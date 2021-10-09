from src.components import create_component, VAEEncoderWrapper
from src.training import TrainingManager
from src.loss import hinge_loss
from src.envs import get_input_shape, get_action_space_size
from src.utils import maybe_update, recursive_dictify, recursive_objectify
from src.evaluation import evaluate_with_planning, latent_evaluate
from src.metrics import *
from src.planners.algos import mc_bfs, selective_mc_bfs, flat_mc_bfs
from src.planners import MCBFSPlanner, FullPlanner, MPCPlanner
from functools import partial


def get_model(model_params, **kwargs):
    name = model_params.name
    if name == 'ppgs':
        return PPGS(model_params=model_params, **kwargs)
    if name == 'latent':
        return LatentModel(model_params=model_params, **kwargs)
    raise Exception('Unknown model name.')


class WorldModel:

    def __init__(self, train_params, eval_params, model_params, optimizer_params, data_params, device):
        self.train_params = train_params
        self.eval_params = eval_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.data_params = data_params
        self.device = device

        self.component_list = []
        self.fav_fast_planner = None
        self.fav_slow_planner = None

        self.start_idx = 0
        self.model_dict = None
        self.optimizer = None
        self.manager = None

        self.lr_dict = self.train_params.lr_dict
        self.loss_weight_dict = self.train_params.loss_weight_dict

    def load(self, checkpoint_path, skip_optimizer=False):
        input_shape = get_input_shape(**self.data_params)
        action_space_size = get_action_space_size(self.data_params.env_name)
        checkpoint = torch.load(open(checkpoint_path, 'rb'), map_location=self.device)
        self.model_dict = {}
        for k in self.component_list:
            self.model_dict[k] = create_component(name=k, args={'input_shape': input_shape, 'device': self.device,
                                                            'action_space_size': action_space_size,
                                                                **checkpoint['model_params']}).to(self.device)
            self.model_dict[k].load_state_dict(checkpoint[f'{k}_state'])
        self.optimizer = None
        if not skip_optimizer:
            self.optimizer = torch.optim.Adam(**checkpoint['optimizer_params'], params=[
                {'params': self.model_dict[k].parameters(), 'lr': self.train_params.lr_dict[k + '_lr']} for k in
                self.component_list])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.start_idx = checkpoint['start_idx']

        self.manager = TrainingManager(self)

    def build(self):
        input_shape = get_input_shape(**self.data_params)
        action_space_size = get_action_space_size(self.data_params.env_name)
        self.model_dict = {k: create_component(name=k, args=dict(input_shape=input_shape, device=self.device,
                                                                 action_space_size=action_space_size,
                                                                 **self.model_params)).to(self.device) for k in self.component_list}
        self.optimizer = torch.optim.Adam(**self.optimizer_params,
                                     params=[{'params': self.model_dict[k].parameters(), 'lr': self.lr_dict[k + '_lr']} for k
                                             in self.component_list])
        self.manager = TrainingManager(self)

    def train(self, ep_idx, train_data_loader):
        self.manager.update(ep_idx)  # ask the manager to update its params according to the current iteration

        ep_metrics = {}  # keeps track of metrics across batches
        count = 0
        for batch in train_data_loader:
            train_dict = self.step(batch, train=True)
            maybe_update(ep_metrics, {k: train_dict[k].item() for k in train_dict})  # record losses and metrics
            count = count + 1
            del batch
            if self.device == 'cpu' and count > 1:  # if running on CPU, going through a single epoch takes too long
                print('Skipping a few batches.')
                break

        avg_ep_metrics = {k: float(sum(ep_metrics[k])/count) for k in ep_metrics}
        return avg_ep_metrics

    def eval(self, test_data_loader):
        ep_metrics, latent_evaluation, evaluation_with_planning = {}, {}, {}  # aggregates losses from each batch
        count = 0
        if self.eval_params.evaluate_losses:
            for batch in test_data_loader:
                val_dict = self.step(batch, train=False)
                maybe_update(ep_metrics, {k: val_dict[k].item() for k in val_dict})
                del batch
                count += 1
                if self.device == 'cpu' and count > 1:  # if running on CPU, going through a single epoch takes too long
                    print('Skipping a few batches.')
                    break
        latent_evaluation = latent_evaluate(model_dict=self.model_dict, eval_params=self.eval_params,
                                            data_params=self.data_params, device=self.device,
                                            planner=self.fav_fast_planner)
        evaluation_with_planning = evaluate_with_planning(model_dict=self.model_dict, eval_params=self.eval_params,
                                                          data_params=self.data_params, device=self.device,
                                                          planner=self.fav_fast_planner)
        val_dict = {**{'val_' + k: float(sum(v) / count) for k, v in ep_metrics.items()},
                **{'val_' + str(k): v for k, v in latent_evaluation.items()},
                **{'val_' + str(k): v for k, v in evaluation_with_planning.items()}}
        return val_dict

    def final_eval(self):
        final_eval_params = recursive_dictify(self.eval_params)
        final_eval_params['eval_at'] = []
        metrics = evaluate_with_planning(self.model_dict, recursive_objectify(final_eval_params), self.data_params,
                                         self.device, self.fav_slow_planner)
        return {'final_' + str(k): v for k, v in metrics.items()}

    def pre_step(self, batch, train):
        for k, v in self.model_dict.items():
            if train:
                v.train()
            else:
                v.eval()
        return batch['state'], batch['action'], batch['anchor']

    def step(self, batch, train=True):
        raise NotImplementedError

    def compute_forward_pred(self, emb_state, anchor, action):
        forward_loss = []
        pred_target_list = []
        action_seq_len = action.shape[1]
        for i in range(action_seq_len):
            emb_trajectory = [emb_state[:, i]]
            for j in range(i, action_seq_len):
                emb_trajectory.append(self.model_dict['forward'].forward(emb_trajectory[-1], anchor, action[:, j]))
                if j == i:
                    pred_target_list.append(emb_trajectory[-1])

            emb_trajectory = emb_trajectory[1:]
            targets = emb_state[:, i + 1:]
            targets = targets.permute(1, 0, 2)
            for pred_target, emb_target in zip(emb_trajectory, targets):
                loss_fn = torch.nn.MSELoss()
                forward_loss.append(loss_fn(pred_target, emb_target))
            del emb_trajectory, targets
        return torch.stack(forward_loss).mean(), pred_target_list

    def backwards_pass(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, checkpoint_path, ep_idx):
        print("Saving model...", end='')
        save_dict = {
            **{f'{k}_state': v.state_dict() for k, v in self.model_dict.items()},
            'optimizer_state': self.optimizer.state_dict(),
            'model_params': recursive_dictify(self.model_params),
            'optimizer_params': recursive_dictify(self.optimizer_params),
            'start_idx': ep_idx
        }
        torch.save(save_dict, open(checkpoint_path, 'wb'))
        print('DONE')


class PPGS(WorldModel):

    def __init__(self, **kwargs):
        super(PPGS, self).__init__(**kwargs)
        self.component_list = ['encoder', 'forward', 'inverse']
        self.fav_fast_planner = partial(MCBFSPlanner, search_algo=mc_bfs)
        self.fav_slow_planner = partial(FullPlanner, search_algo=selective_mc_bfs)

    def step(self, batch, train=True):

        obs, action, anchor = self.pre_step(batch, train)
        batch_size = action.shape[0]
        action_seq_len = action.shape[1]
        emb_state = self.model_dict['encoder'].forward(obs.reshape((-1,) + tuple(obs.shape[2:]))).reshape(
            tuple(obs.shape[:2]) + (-1,))

        forward_loss, pred_target_list = self.compute_forward_pred(emb_state, anchor, action)

        start_state = obs[:, :-1].reshape((batch_size * action_seq_len, -1))
        target_state = obs[:, 1:].reshape((batch_size * action_seq_len, -1))
        emb_start = emb_state[:, :-1].reshape((batch_size * action_seq_len, -1))
        emb_target = emb_state[:, 1:].reshape((batch_size * action_seq_len, -1))
        pred_target = torch.stack(pred_target_list, dim=1).reshape((batch_size * action_seq_len, -1))
        pred_action = self.model_dict['inverse'].forward(emb_start.clone(), emb_target.clone())

        inverse_loss = torch.nn.CrossEntropyLoss()(pred_action, action.reshape(-1))
        h_loss = hinge_loss(emb_start, emb_target, pred_target, margin=self.train_params.margin,
                            hinge_params=self.train_params.hinge_params, device=self.device)

        # weigh losses
        forward_weight, inverse_weight, hinge_weight = self.manager.loss_weight_dict['forward_w'], \
                                                       self.manager.loss_weight_dict['inverse_w'], \
                                                       self.manager.loss_weight_dict['hinge_w']
        total_loss = forward_weight * forward_loss + inverse_weight * inverse_loss + h_loss * hinge_weight

        if train:
            self.backwards_pass(total_loss)

        # metrics
        inverse_accuracy = multiclass_accuracy(pred_action, action)
        adj_forward_loss = adjusted_forward_loss(emb_start, emb_target, pred_target)
        forward_accuracy = embedding_accuracy(pred_target, emb_target, margin=self.train_params.margin)
        fr_of_trans, fr_of_trans_loop, fr_of_trans_non_loop = \
            fraction_of_large_transitions(emb_start, emb_target, pred_target, self.train_params.margin)

        return {"forward_loss": forward_loss,
                "inverse_loss": inverse_loss,
                "hinge_loss": h_loss,
                "total_loss": total_loss,
                "forward_accuracy": forward_accuracy,
                "inverse_accuracy": inverse_accuracy,
                "adj_forward_loss": adj_forward_loss,
                "frac_long_trans": fr_of_trans,
                "frac_long_trans_loop": fr_of_trans_loop,
                "frac_long_trans_non_loop": fr_of_trans_non_loop}


class LatentModel(PPGS):

    def __init__(self, **kwargs):
        super(LatentModel, self).__init__(**kwargs)
        self.component_list = ['encoder', 'latent_forward', 'inverse']
        self.fav_fast_planner = partial(MCBFSPlanner, search_algo=mc_bfs)
        self.fav_slow_planner = partial(FullPlanner, search_algo=selective_mc_bfs)

    def build(self):
        input_shape = get_input_shape(**self.data_params)
        action_space_size = get_action_space_size(self.data_params.env_name)
        self.model_dict = {k: create_component(name=k, args=dict(input_shape=input_shape, device=self.device,
                                                                 action_space_size=action_space_size,
                                                                 **self.model_params)).to(self.device) for k in self.component_list}
        self.model_dict['forward'] = self.model_dict['latent_forward']
        self.model_dict.pop('latent_forward')
        self.optimizer = torch.optim.Adam(**self.optimizer_params,
                                     params=[{'params': self.model_dict[k].parameters(), 'lr': self.lr_dict[k + '_lr']} for k
                                             in self.model_dict.keys()])
        self.manager = TrainingManager(self)

    def load(self, checkpoint_path, skip_optimizer=False):
        input_shape = get_input_shape(**self.data_params)
        action_space_size = get_action_space_size(self.data_params.env_name)
        checkpoint = torch.load(open(checkpoint_path, 'rb'), map_location=self.device)
        self.model_dict = {}
        for k in self.component_list:
            self.model_dict[k] = create_component(name=k, args={'input_shape': input_shape, 'device': self.device,
                                                            'action_space_size': action_space_size,
                                                                **checkpoint['model_params']}).to(self.device)
            self.model_dict[k].load_state_dict(checkpoint[f'{k}_state'])
        self.model_dict['forward'] = self.model_dict['latent_forward']
        self.model_dict.pop('latent_forward')
        self.optimizer = None
        if not skip_optimizer:
            self.optimizer = torch.optim.Adam(**checkpoint['optimizer_params'], params=[
                {'params': self.model_dict[k].parameters(), 'lr': self.train_params.lr_dict[k + '_lr']} for k in
                self.model_dict.keys()])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.start_idx = checkpoint['start_idx']

        self.manager = TrainingManager(self)

    def save(self, checkpoint_path, ep_idx):
        print("Saving model...", end='')
        self.model_dict['latent_forward'] = self.model_dict['forward']
        self.model_dict.pop('forward')
        save_dict = {
            **{f'{k}_state': v.state_dict() for k, v in self.model_dict.items()},
            'optimizer_state': self.optimizer.state_dict(),
            'model_params': recursive_dictify(self.model_params),
            'optimizer_params': recursive_dictify(self.optimizer_params),
            'start_idx': ep_idx
        }
        torch.save(save_dict, open(checkpoint_path, 'wb'))
        self.model_dict['forward'] = self.model_dict['latent_forward']
        self.model_dict.pop('latent_forward')
        print('DONE')
