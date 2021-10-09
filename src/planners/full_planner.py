from src.planners.base import AbstractPlanner
import torch
import numpy as np
from src.data import preprocess_image
from src.envs import env_to_model_action_map, model_to_env_action_map, represent_same_state
from src.utils import efficient_from_numpy
from src.planners.algos import mc_bfs, selective_mc_bfs, get_latent_trajectory
import matplotlib.pyplot as plt


class FullPlanner(AbstractPlanner):

    def plan(self, environment, initial_obs, final_obs, max_depth):
        for k in self.model_dict:
            self.model_dict[k].eval()

        min_similarity = 1 - (self.planner_params.margin**2)/8  # min_similarity for being reideintified
        lookup = LookupForwardModel(self.model_dict['forward'], min_similarity)

        initial_emb = self.model_dict['encoder'].forward(initial_obs)
        goal_emb = self.model_dict['encoder'].forward(final_obs)

        best_action_seq = selective_mc_bfs(start=initial_emb, goal=goal_emb, forbidden=None, anchor=initial_obs,
                                           forward_model=lookup, action_set=self.action_set, max_depth=max_depth,
                                           margin=self.planner_params.margin, early_stop=self.planner_params.early_stop,
                                           batch_size=self.planner_params.batch_size, snap=False, device=self.device)

        # print(f'Initial plan: {best_action_seq}')
        pred_state_traj = get_latent_trajectory(initial_emb, initial_obs, best_action_seq, lookup, self.device)[1:]
        best_action_seq = model_to_env_action_map(np.array(best_action_seq), env_name=self.env_name)
        best_action_seq = [a for a in best_action_seq]
        last_obs = environment.reset()
        last_obs = efficient_from_numpy(preprocess_image(last_obs, **self.data_params), device=self.device).float().unsqueeze(0)
        last_emb = self.model_dict['encoder'].forward(last_obs)
        assert torch.allclose(last_emb, initial_emb), 'Environment does not match.'

        visited = Visited(min_similarity, set=not self.planner_params.penalize_visited)
        visited.append(last_emb.detach().cpu().numpy())
        path_from_start = Visited(min_similarity)
        path_from_start.append(last_emb.detach().cpu().numpy())
        actions_from_start = []
        n_step = 0

        # fig = plt.gcf()
        # fig.show()
        # fig.canvas.draw()

        while len(best_action_seq) > 0 and n_step < self.planner_params.max_steps:  # as long as actions are suggested

            # print(best_action_seq)
            random_step = np.random.uniform(0, 1) < self.planner_params.eps
            action = best_action_seq.pop(0) if not random_step else np.random.choice(model_to_env_action_map(np.array(self.action_set), self.env_name))
            new_obs, rew, _, _ = environment.step(int(action))
            new_obs = new_obs
            n_step += 1

            # if rew >= 10:
            #     return True, n_step

            if represent_same_state(preprocess_image(new_obs, **self.data_params), final_obs[0].cpu().detach().numpy(), **self.data_params):
                # print(n_step)
                return True, n_step

            new_obs = efficient_from_numpy(preprocess_image(new_obs, **self.data_params), device=self.device).float().unsqueeze(0)
            new_emb = self.model_dict['encoder'].forward(new_obs)

            # plt.imshow(new_obs.permute(0, 2, 3, 1)[0, :, :, :3].detach().cpu().numpy())
            # fig.canvas.draw()

            visited.append(new_emb.detach().cpu().numpy())  # add to visited
            if path_from_start.discard_after(new_emb.detach().cpu().numpy()):  # if already been here, update to shortest path
                actions_from_start = actions_from_start[:len(path_from_start)-1]
            else:
                path_from_start.append(new_emb.detach().cpu().numpy())
                actions_from_start.append(action)

            lookup.insert(last_emb, int(env_to_model_action_map(np.array(action), env_name=self.env_name)), new_emb)

            assert path_from_start[-1] @ new_emb.detach().cpu().numpy().T > min_similarity
            pred_emb = pred_state_traj.pop(0)

            if new_emb @ pred_emb.T < min_similarity or len(best_action_seq) == 0 or random_step:
                if new_emb @ pred_emb.T < min_similarity:
                    lookup.insert(last_emb, int(env_to_model_action_map(np.array(action), env_name=self.env_name)), new_emb)
                    # print('Update lookup')
                while True:
                    assert np.any(new_emb.detach().cpu().unsqueeze(0).numpy() @ visited.get_visited().T > min_similarity)
                    best_action_seq = selective_mc_bfs(start=new_emb, goal=goal_emb, forbidden=visited.get_visited(),
                         anchor=new_obs, forward_model=lookup, action_set=self.action_set, max_depth=self.planner_params.replan_horizon,
                         margin=self.planner_params.margin, early_stop=self.planner_params.early_stop,
                         batch_size=self.planner_params.batch_size, snap=self.planner_params.snap, device=self.device)  # try to replan
                    if len(best_action_seq) == 0:
                        if self.planner_params.backtrack:
                            if len(path_from_start) <= 1:
                                break
                            prev_emb = torch.tensor(path_from_start[-2], device=self.device)
                            best_action_seq = mc_bfs(start=new_emb, goal=prev_emb,
                                                     anchor=new_obs, forward_model=lookup, action_set=self.action_set, max_depth=1,
                                                     margin=self.planner_params.margin, early_stop=self.planner_params.early_stop,
                                                     batch_size=self.planner_params.batch_size, device=self.device)
                            break
                        else:
                            visited.append(new_emb.detach().cpu().numpy())
                            n_step += 1
                            if not self.planner_params.penalize_visited:
                                break

                    if len(best_action_seq) > 0 or n_step >= self.planner_params.max_steps:
                        break

                pred_state_traj = get_latent_trajectory(new_emb, new_obs, best_action_seq, lookup, self.device)[1:]
                best_action_seq = model_to_env_action_map(np.array(best_action_seq), env_name=self.env_name)
                best_action_seq = [a for a in best_action_seq]

            last_emb = new_emb
        return False, n_step


class LookupForwardModel:

    def __init__(self, forward_model, min_similarity):
        self.keys = None
        self.values = []
        self.min_similarity = min_similarity
        self.forward_model = forward_model

    def forward(self, z, s, a):
        pred = self.forward_model.forward(z, s, a)
        if self.keys is not None:
            for i, row in enumerate(z @ self.keys.T):
                if row.max() > self.min_similarity and int(a[i]) in self.values[row.argmax()]:
                    pred[i] = self.values[row.argmax()][int(a[i])]
        return pred

    def insert(self, key, a, value):
        # key is 1 * emb_size
        # same for value
        # a is int
        if self.keys is None:
            self.keys = key.detach().clone()
            self.values = [{a: value.detach().clone()}]
            return
        tmp = (key @ self.keys.T > self.min_similarity).reshape(-1)
        if torch.any(tmp):
            ind = torch.where(tmp)[0]
            if len(ind) > 1:
                # print('This should not happen (often).')
                pass
            for idx in ind:
                if a in self.values[int(idx)]:
                    # print('overwrite')
                    pass
                # can happen when open loop prediction is close enough to the new_emb, but there is a third past state close to the new_emb
                # and far from the prediction. The next predicted state would be correct if we used the new_emb, since it would use
                # the lookup. However, the open loop pred does not match naything in the lookup and leads to a mistake, which is then
                # inserted using the new_emb as key, causing overwriting
                self.values[int(idx)][a] = value.detach().clone()
        else:
            self.keys = torch.cat([self.keys, key.detach().clone()], dim=0)
            self.values.append({a: value.detach().clone()})

        # check that all keys are different.
        assert (self.keys @ self.keys.T > self.min_similarity).sum() == self.keys.shape[0], 'Keys are not unique'


class Visited:

    def __init__(self, min_similarity, set=True):
        self.visited = None
        self.min_similarity = min_similarity
        self.set = set

    def contains(self, item):
        bs = item.shape[0]
        if self.visited is None:
            return np.array([False]*bs)
        return np.any(item @ self.visited.T > self.min_similarity, axis=-1)

    def append(self, item):
        # item is 1 x emb_size
        if self.contains(item) and self.set:
            return
        if self.visited is None:
            self.visited = item.copy()
        else:
            self.visited = np.concatenate([self.visited, item], axis=0)
        # check that all keys are different.
        assert (not self.set) or np.sum(self.visited @ self.visited.T > self.min_similarity) == self.visited.shape[0], 'Keys are not unique'

    def discard_after(self, item):
        if self.visited is not None:
            indices = np.where((item @ self.visited.T > self.min_similarity).reshape(-1))
            if len(indices[0]) > 0:
                self.visited = self.visited[:np.amin(indices)+1]
                return True
        return False

    def __len__(self):
        if self.visited is None:
            return 0
        else:
            return self.visited.shape[0]

    def __getitem__(self, item):
        return self.visited[[item]]

    def get_visited(self):
        return self.visited
