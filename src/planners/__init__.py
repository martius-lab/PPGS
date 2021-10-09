from .one_shot_planner import MCBFSPlanner
from .full_planner import FullPlanner
from .mpc_planner import MPCPlanner
from.baseline_planner import BaselinePlanner


def get_planner_class(name):
    if name == 'mcbfs':
        return MCBFSPlanner
    if name == 'mpc':
        return MPCPlanner
    if name == 'memory':
        return FullPlanner
    if name == 'baseline':
        return BaselinePlanner
    raise Exception('Unknown planner.')
