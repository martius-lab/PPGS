from src.planners.base import AbstractPlanner
import numpy as np
from src.data import preprocess_image
from src.envs import model_to_env_action_map, represent_same_state
from src.utils import efficient_from_numpy
from src.planners.algos import mc_bfs


class MPCPlanner(AbstractPlanner):

    def plan(self, environment, initial_obs, final_obs, max_depth):

        for k in self.model_dict:
            self.model_dict[k].eval()
        environment.reset()
        cumulative_action_seq = []
        start = self.model_dict['encoder'].forward(initial_obs)
        goal = self.model_dict['encoder'].forward(final_obs)
        for i in range(max_depth):
            best_action_seq = self.search_algo(start=start, goal=goal, anchor=initial_obs, forward_model=self.model_dict['forward'],
                                     action_set=self.action_set, max_depth=self.planner_params.replan_horizon,
                                     margin=self.planner_params.margin, early_stop=self.planner_params.early_stop,
                                     batch_size=self.planner_params.batch_size, device=self.device)
            best_action_seq = model_to_env_action_map(np.array(best_action_seq), env_name=self.env_name)

            if len(best_action_seq) == 0:
                return False, len(cumulative_action_seq)

            best_action = best_action_seq[0]
            cumulative_action_seq.append(best_action)
            new_obs = environment.step(best_action)[0]
            new_obs = preprocess_image(new_obs, **self.data_params)

            if represent_same_state(new_obs, final_obs[0].cpu().detach().nupred_final_obs, **self.data_paramsmpy(), **self.data_params):
                return True, len(cumulative_action_seq)

            new_obs = efficient_from_numpy(new_obs, device=self.device).float().unsqueeze(0)
            start = self.model_dict['encoder'].forward(new_obs)

        return False, len(cumulative_action_seq)
