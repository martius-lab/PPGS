import numpy as np
from src.data import preprocess_image
from src.envs import get_env_action_set, represent_same_state
from src.planners.base import AbstractPlanner
import networkx


class BaselinePlanner(AbstractPlanner):
    def plan(self, environment, initial_obs, final_obs, max_depth):

        n_step = 0
        new_obs = environment.reset()
        if represent_same_state(preprocess_image(new_obs, **self.data_params), final_obs[0].cpu().detach().numpy(), **self.data_params):
            return True, n_step

        new_obs = new_obs.reshape(1, -1)
        all_obs = new_obs
        curr_state = np.where(np.sum(np.abs(all_obs - new_obs), axis=-1) < 1e-4)[0][0]
        g = networkx.DiGraph()
        g.add_node(curr_state)
        used_actions = {curr_state: []}
        trans = []

        while True:
            if len(used_actions[curr_state]) < 5:
                new_act = np.random.choice([i for i in get_env_action_set(self.env_name) if i not in used_actions[curr_state]]).item()
            else:
                closest_states = networkx.shortest_path_length(g, source=curr_state)
                closest_states = {k: v for k, v in closest_states.items() if len(used_actions[k]) < 5}
                if len(closest_states) == 0:
                    # this only happens when the mouse gets the cheese but observations are slightly different
                    # or in other environments, when you get stuck in a funnel state
                    return False, self.planner_params.max_steps
                best_state = min(closest_states, key=closest_states.get)

                if closest_states[best_state] > 100000:
                    raise Exception('All levels should be solvable')
                new_act = g[curr_state][networkx.shortest_path(g, source=curr_state, target=best_state)[1]]['action']

            new_obs = environment.step(new_act)[0]
            n_step += 1

            # print(new_act)
            if represent_same_state(preprocess_image(new_obs, **self.data_params), final_obs[0].cpu().detach().numpy(),
                                    **self.data_params):
                # print("n_steps: ", n_step)
                return True, n_step
            new_obs = new_obs.reshape(1, -1)
            if len(np.where(np.sum(np.abs(all_obs - new_obs), axis=-1) < 1e-4)[0]) == 0:
                # print('New state')
                all_obs = np.concatenate([all_obs, new_obs], axis=0)
                new_state = np.where(np.sum(np.abs(all_obs - new_obs), axis=-1) < 1e-4)[0][0]
                g.add_node(new_state)
                used_actions[new_state] = []

            new_state = np.where(np.sum(np.abs(all_obs - new_obs), axis=-1) < 1e-4)[0][0]
            if new_act not in used_actions[curr_state]:
                used_actions[curr_state].append(new_act)
                g.add_edge(curr_state, new_state)
                g[curr_state][new_state]['action'] = new_act
                trans.append((curr_state, new_state, new_act))

            curr_state = new_state
            if n_step >= self.planner_params.max_steps:
                break
        return False, n_step