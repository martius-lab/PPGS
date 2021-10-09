from src.planners.base import AbstractPlanner
import numpy as np
from src.data import preprocess_image
from src.envs import model_to_env_action_map, represent_same_state
from src.planners.algos import mc_bfs


class MCBFSPlanner(AbstractPlanner):

    def plan(self, environment, initial_obs, final_obs, max_depth):
        for k in self.model_dict:
            self.model_dict[k].eval()

        best_action_seq = self.search_algo(start=self.model_dict['encoder'].forward(initial_obs), goal=self.model_dict['encoder'].forward(final_obs),
                                 anchor=initial_obs, forward_model=self.model_dict['forward'], action_set=self.action_set, max_depth=max_depth,
                                 margin=self.planner_params.margin, early_stop=self.planner_params.early_stop,
                                 batch_size=self.planner_params.batch_size, device=self.device)
        best_action_seq = model_to_env_action_map(np.array(best_action_seq), env_name=self.env_name)

        pred_final_obs = environment.reset()
        for a in best_action_seq:
            pred_final_obs = environment.step(int(a))[0]

        if represent_same_state(preprocess_image(pred_final_obs, **self.data_params), final_obs[0].cpu().detach().numpy(), **self.data_params):
            return True, len(best_action_seq)
        return False, len(best_action_seq)
