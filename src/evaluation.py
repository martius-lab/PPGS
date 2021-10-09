import numpy as np
import torch
from src.envs import make_env, get_env_action_set, env_to_model_action_map
from src.planners import get_planner_class, BaselinePlanner
from src.data import preprocess_image
from src.utils import efficient_from_numpy


def evaluate_with_planning(model_dict, eval_params, data_params, device, planner=None):

    env_name = data_params.env_name
    planner_params = eval_params.planner_params
    for k in model_dict:
        model_dict[k].eval()

    planner_class = get_planner_class(eval_params.planner) if planner is None else planner
    planner = planner_class(model_dict=model_dict, env_name=env_name, planner_params=planner_params,
                            data_params=data_params, device=device)

    metrics = {}
    for seed in eval_params.seeds:
        baseline_planner = BaselinePlanner(model_dict=model_dict, env_name=env_name, planner_params=planner_params,
                                           data_params=data_params, device=device, search_algo=None)
        attempted = {**{str(i): 0 for i in eval_params.eval_at}, 'max': 0}
        solved = {**{str(i): 0 for i in eval_params.eval_at}, 'max': 0}
        baseline_steps, method_steps = [], []
        for i in range(eval_params.num_levels):
            gt_env = make_env(env_name, int(seed)+i, data_params.env_params)
            initial_obs = gt_env.reset()
            initial_obs = efficient_from_numpy(preprocess_image(initial_obs, **data_params), device=device).unsqueeze(0)
            actions = gt_env.get_solution()
            for j, a in enumerate(actions):
                final_obs = gt_env.step(a)[0]
                final_obs = efficient_from_numpy(preprocess_image(final_obs, **data_params), device=device).unsqueeze(0)
                if j+1 in eval_params.eval_at:
                    attempted[str(j+1)] += 1
                    env = make_env(env_name, int(seed)+i, data_params.env_params)
                    success, n_step = planner.plan(env, initial_obs, final_obs, j+1)
                    solved[str(j+1)] += success
            attempted['max'] += 1
            env = make_env(env_name, int(seed)+i, data_params.env_params)
            success, n_step = planner.plan(env, initial_obs, final_obs, len(actions))
            solved['max'] += success
            baseline_env = make_env(env_name, int(seed)+i, data_params.env_params)
            method_steps.append(float(n_step))
            baseline_steps.append(float(baseline_planner.plan(baseline_env, initial_obs, final_obs, len(actions))[1]))

        metrics.update({**{f'plan_acc_{s}_{seed}': solved[str(s)]/(attempted[str(s)]+1e-8) for s in eval_params.eval_at},
                        f'lvl_solved_ratio_{seed}': solved['max']/(attempted['max']+1e-8),
                        f'speedup_{seed}': sum(baseline_steps)/(sum(method_steps)+1e-8)})
    return metrics


def latent_evaluate(model_dict, eval_params, data_params, device, planner=None):

    encoder = model_dict['encoder']
    forward_model = model_dict['forward']
    encoder.eval()
    forward_model.eval()
    env_name = data_params.env_name
    batch_size = eval_params.batch_size
    eval_at = eval_params.latent_eval_at

    np.random.seed(None)
    metrics = {**{f"mrr_{s}": [] for s in eval_at}, **{f"hat1_{s}": [] for s in eval_at}}
    for i in range(batch_size):
        actions = np.random.choice(get_env_action_set(env_name), size=max((max(eval_at) + 10), 40))
        env = make_env(env_name, seed=eval_params.seeds[0]+i, env_params=data_params.env_params)
        obs = [env.reset()]
        for a in actions:
            obs.append(env.step(int(a))[0])  # only save obs
        env.close()

        obs = torch.stack([efficient_from_numpy(preprocess_image(e, **data_params), device=device) for e in obs], dim=0)
        embs = encoder.forward(obs)
        autoregressive = [embs[[0]]]
        for a in efficient_from_numpy(env_to_model_action_map(actions, env_name), device).reshape(-1, 1):
            autoregressive.append(forward_model.forward(autoregressive[-1], obs[[0]], a))

        for s in eval_at:
            distances = torch.nn.PairwiseDistance()(torch.stack([embs[s]]*len(embs), dim=0), embs)
            pred_distance = torch.nn.PairwiseDistance()(embs[s], autoregressive[s])
            rank = torch.sum((distances < pred_distance)*(distances > distances[s])) + 1
            metrics[f'mrr_{s}'].append(1. / rank.float())
            metrics[f'hat1_{s}'].append((rank == 1).float())

    return {k: (sum(v)/float(len(v))).item() for k, v in metrics.items()}
