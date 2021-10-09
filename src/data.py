import numpy as np
import torch
from src.utils import efficient_from_numpy
from src.envs import env_to_model_action_map


def preprocess_image(rgb, normalize, encode_position, **kwargs):
    """
    Preprocesses an 8bit RGB observation by (optionally) normalizing it and adding two positional channels.
    """
    rgb = rgb.astype(np.float32)
    rgb = rgb / 255. if normalize else rgb
    if encode_position:
        h, v = rgb.shape[-3:-1]
        pos_b = np.zeros((*rgb.shape[:-1], 2))
        pos_b[..., 0] = np.broadcast_to(np.arange(0, h).reshape([1]*(len(pos_b.shape)-2)+[h]), (*pos_b.shape[:-2], h))
        pos_b[..., 1] = np.broadcast_to(np.arange(0, v).reshape([1]*(len(pos_b.shape)-3)+[h, 1]), (*pos_b.shape[:-3], h, v))
        pos_b = pos_b / 64. if normalize else pos_b
        rgb = np.concatenate([rgb, pos_b], axis=-1)  # 64x64x?
    return np.moveaxis(rgb, -1, -3).astype(np.float32)


class EnvDataset(torch.utils.data.Dataset):
    """
    Dataset interface for Procgen episodes (and other environments hopefully).
    """

    def __init__(self, path, data_params, device):
        self.path = path
        self.seq_len = data_params.seq_len
        self.normalize = data_params.normalize
        self.encode_position = data_params.encode_position
        self.env_name = data_params.env_name
        self.device = device
        self.data = None
        self.n_episodes = None
        self.total_length = None
        self.episode_lengths = None
        self.id_map = None
        self.sample_count = None
        self.load_data()

    def load_data(self):
        """
        Loads data from .npy file, organizes it into rollouts of varying length.
        """
        if self.data is None:
            self.total_length = 0
            self.episode_lengths = {}
            full_dataset = np.load(self.path, allow_pickle=True).tolist()
            print(f'Loaded data from {self.path}')

            # throw out all unnecessary information and save observations and actions in state
            clean_dataset = []
            episode_id = 0
            for i, key in enumerate(full_dataset):
                observations = np.array([o for o in full_dataset[key]['obs']], dtype=np.uint8)
                actions = env_to_model_action_map(np.array(full_dataset[key]['actions']), self.env_name).reshape((-1))
                length = len(actions)
                if length < self.seq_len:
                    continue  # discard episodes that are too short
                clean_dataset.append(dict(observations=observations, actions=actions, seed=key))
                self.episode_lengths[episode_id] = length
                self.total_length += length
                episode_id += 1
            self.n_episodes = episode_id
            self.data = np.array(clean_dataset, dtype=object)
            del full_dataset

            # maps an id to a pair
            self.id_map = {}
            i = 0
            for key, value in self.episode_lengths.items():
                for step in range(value - self.seq_len + 2):
                    self.id_map[i] = (key, step)
                    i += 1
            self.sample_count = i

    def __getitem__(self, item):
        """
        Returns a single rollout of length self.seg_len. 'state' contains the preprocessed observation of shape
        (seq_len x n_channels x 64 x 64), 'action' contains the actions embedded according to the model's input
        of shape (seq_len-1), 'anchor' contains a random observation from the same rollout of shape (64x64xn_channels),
        while seed contains the seed of the level. If the agent's position is available, it is embedded into a 3d hyper-
        shpere and contained in 'position', of shape (seq_len x 3).
        """
        episode, step = self.id_map[item]
        anchor_step = np.random.choice(self.episode_lengths[episode]+1)
        return_dict = dict(state=efficient_from_numpy(self._preprocess(self.data[episode]['observations'][step:step + self.seq_len]), self.device),
                           action=efficient_from_numpy(self.data[episode]['actions'][step:step + self.seq_len - 1], self.device),
                           anchor=efficient_from_numpy(self._preprocess(self.data[episode]['observations'][anchor_step]), self.device),
                           seed=self.data[episode]['seed'])
        return return_dict

    def __len__(self):
        return self.sample_count

    def _preprocess(self, rgb):
        return preprocess_image(rgb, normalize=self.normalize, encode_position=self.encode_position)
