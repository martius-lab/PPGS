import abc
from src.envs import get_model_action_set


class AbstractPlanner(abc.ABC):

    def __init__(self, model_dict, env_name, planner_params, data_params, search_algo, device):
        super(AbstractPlanner, self).__init__()
        self.model_dict = model_dict
        self.env_name = env_name
        self.planner_params = planner_params
        self.data_params = data_params
        self.search_algo = search_algo
        self.action_set = get_model_action_set(env_name)
        self.device = device

    @abc.abstractmethod
    def plan(self, environment, initial_obs, final_obs, max_depth):
        pass
