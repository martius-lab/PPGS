import os
import time
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from src.data import EnvDataset
from src.model import get_model
from src.utils import set_seed, recursive_objectify, recursive_dictify


def main(train_params, eval_params, model_params, optimizer_params, data_params, working_dir, **kwargs):

    if train_params.seed is not None:
        set_seed(train_params.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('Training on CPU is only suitable for debugging. Most batches will be skipped to reduce training time. Please use a CUDA device to reproduce results.')
    epochs, batch_size = train_params.epochs, train_params.batch_size
    logger = SummaryWriter(working_dir)

    # load and preprocess datasets of pairs of consecutive frames
    train_dataset = EnvDataset(path=data_params.train_path, data_params=data_params, device=device)
    test_dataset = EnvDataset(path=data_params.test_path, data_params=data_params, device=device)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=data_params.shuffle)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=data_params.shuffle)

    checkpoint_path = os.path.join(working_dir, 'latest.tor')   # create model and load weights
    model = get_model(model_params=model_params, train_params=train_params, eval_params=eval_params,
                      optimizer_params=optimizer_params, data_params=data_params, device=device)
    if os.path.exists(checkpoint_path) and model_params.load_model:
        print(f'Loading checkpoint from {checkpoint_path}...', end='')
        model.load(checkpoint_path)
    else:
        print('Creating fresh model...', end='')
        model.build()
    print('DONE')

    metrics = {}  # contains the metrics of the current epoch and is passed to cluster_utils
    for ep_idx in range(model.start_idx, epochs):
        print(f'Epoch: {ep_idx} - training...')

        avg_ep_metrics = model.train(ep_idx, train_data_loader)
        metrics.update({**avg_ep_metrics, "step": ep_idx, **model.manager.loss_weight_dict, **model.manager.lr_dict})

        if ep_idx % train_params.val_every_n_epochs == 0:
            print(f'Epoch: {ep_idx} - evaluating...')
            with torch.no_grad():
                val_dict = model.eval(test_data_loader)
            metrics.update(val_dict)

        for k, v in metrics.items():
            logger.add_scalar(k, v, ep_idx)

        info_str = " ".join(['- '+str(k)+' = '+str(v)+'\n\t' for (k, v) in metrics.items()])
        print(f"Epoch: {ep_idx} {info_str}", end='\r\n')

        if ep_idx % train_params.save_every_n_epochs == 0:  # save checkpoint
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
            model.save(checkpoint_path, ep_idx+1)

    print('Final evaluation.')
    metrics.update(model.final_eval())
    print('Finished.')
    print(metrics)


if __name__ == '__main__':
    main(**recursive_objectify(json.load(open('default_params.json'))))
