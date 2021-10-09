import numpy as np
from puzzlegen import IceSlider, DigitJump


def get_input_shape(env_name, encode_position, **kwargs):
    """
    Returns the input shape of the encoder for particular environments.
    """
    shape_dict = {'ice_slider': (3, 64, 64),
                  'digit_jump': (3, 64, 64),
                  }

    shape = shape_dict[env_name]
    return (shape[0]+2,) + shape[1:] if encode_position else shape


def get_no_op(env_name):
    """
    Returns the no-op key for a particular environment.
    """
    no_opt_dict = {'ice_slider': 4, 'digit_jump': 4}
    return no_opt_dict[env_name]


def get_model_to_env_action_dict(env_name):
    """
    Returns the map from action keys used in the model (sequential integers starting from 0) to action keys used in the
    environment.
    """
    d = {
        'ice_slider': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        'digit_jump': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},

    }
    return d[env_name]


def get_env_to_model_action_dict(env_name):
    """
    Returns the map from action keys used in the environment to action keys used in the model (sequential integers
    starting from 0).
    """
    return {v: k for k, v in get_model_to_env_action_dict(env_name).items()}


def get_model_action_set(env_name):
    """
    Returns all action keys that can be given as input to the model.
    """
    return [k for k, v in get_model_to_env_action_dict(env_name).items()]


def get_action_space_size(env_name):
    return len(get_model_action_set(env_name))


def get_env_action_set(env_name):
    """
    Returns all action keys the environment accepts.
    """
    return [v for k, v in get_model_to_env_action_dict(env_name).items()]


def env_to_model_action_map(a, env_name):
    """
    Maps from procgen action space to the restricted action space used by the model.
    """
    original_shape = a.shape
    flat_a = a.copy().reshape(-1)
    d = get_env_to_model_action_dict(env_name)
    for i in range(len(flat_a)):
        flat_a[i] = d[flat_a[i]]
    return flat_a.reshape(original_shape)


def model_to_env_action_map(a, env_name):
    """
    Maps from the restricted model action space to procgen action space.
    """
    original_shape = a.shape
    flat_a = a.copy().reshape(-1)
    d = get_model_to_env_action_dict(env_name)
    for i in range(len(flat_a)):
        flat_a[i] = d[flat_a[i]]
    return flat_a.reshape(original_shape)


def represent_same_state(a, b, env_name, encode_position, **kwargs):
    if env_name == 'maze':
        return True if np.count_nonzero(np.count_nonzero(a - b, axis=0)) <= 1 else False
    if env_name in ['digit_jump', 'ice_slider']:
        return np.allclose(a, b)
    raise NotImplementedError('Unknown environment.')


def make_env(env_name, seed, env_params=None):
    puzzle_fn = {'ice_slider': IceSlider, 'digit_jump': DigitJump}
    if env_name in ['ice_slider', 'digit_jump']:
        return puzzle_fn[env_name](seed=seed, **({} if env_params is None else env_params))
    raise Exception('Unknown environment.')
