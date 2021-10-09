import torch

class TrainingManager(object):
    """
    Modifies training weights/learning rates, freezes and unfreezes models.
    It extracts a discrete schedule from the config and updates hyperparameters at corresponding epochs.
    """

    def __init__(self, model):
        self.component_list = model.component_list
        self.model_dict = model.model_dict
        self.lr_dict = model.lr_dict
        self.loss_weight_dict = model.loss_weight_dict
        self.optimizer = model.optimizer
        self.decay = model.train_params.lr_decay
        self.schedule = {}
        for e in model.train_params.schedule:  # calculate epochs that require updates
            self.schedule[int(e['progress'] * model.train_params.epochs)] = e

    @staticmethod
    def freeze(model, verbose=True):
        for param in model.parameters():
            param.requires_grad = False
        if verbose:
            print(f'{model.__class__.__name__} is frozen.')
        model.is_frozen = True

    @staticmethod
    def unfreeze(model, verbose=True):
        for param in model.parameters():
            param.requires_grad = True
        if verbose:
            print(f'{model.__class__.__name__} is not frozen.')
        model.is_frozen = False

    def update(self, epoch_id, verbose=True):

        if epoch_id > 0 and self.decay < 1:  # lr decay
            state_dict = self.optimizer.state_dict()
            for i in range(len(state_dict['param_groups'])):
                state_dict['param_groups'][i]['lr'] = state_dict['param_groups'][i]['lr'] * self.decay
            self.optimizer.load_state_dict(state_dict)

        if epoch_id in self.schedule.keys():
            for param_dict in [self.lr_dict, self.loss_weight_dict]:
                for k in param_dict:
                    if k in self.schedule[epoch_id].keys():
                        param_dict[k] = self.schedule[epoch_id][k]
            for k in self.component_list:
                if 'freeze_' + k in self.schedule[epoch_id].keys():
                    if self.schedule[epoch_id]['freeze_'+k]:
                        self.freeze(self.model_dict[k])
                    else:
                        self.unfreeze(self.model_dict[k])
            if verbose:
                print(f'Weights updated to: {self.loss_weight_dict}.')
                print(f'Learning rates updated to: {self.lr_dict}.')

            state_dict = self.optimizer.state_dict()
            for i, k in enumerate(self.component_list):
                state_dict['param_groups'][i]['lr'] = self.lr_dict[k+'_lr']
            self.optimizer.load_state_dict(state_dict)
