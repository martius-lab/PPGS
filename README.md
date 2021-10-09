# PPGS: Planning from Pixels in Environments with Combinatorially Hard Search Spaces

### Environment Setup

- We recommend pipenv for creating and managing virtual environments (dependencies for other environment managers can be found in `Pipfile`)

```
git clone https://github.com/martius-lab/PPGS
cd ppgs
pipenv install
pipenv shell
```

- For simplicity, this codebase is ready for training on two of the three environments (IceSlider and DigitJump). They are part of the *puzzlegen* package, which we provide [here](https://github.com/martius-lab/puzzlegen), and can be simply installed with
```
pip install -e https://github.com/martius-lab/puzzlegen
```

- Offline datasets can be generated for training and validation. In the case of IceSlider we can use

```
python -m puzzlegen.extract_trajectories --record-dir /path/to/train_data --env-name ice_slider --start-level 0 --number-levels 1000 --max-steps 20 --n-repeat 20 --random 1
python -m puzzlegen.extract_trajectories --record-dir /path/to/test_data --env-name ice_slider --start-level 1000 --number-levels 1000 --max-steps 20 --n-repeat 5 --random 1
```

- Finally, we can add the paths to the extracted datasets in `default_params.json` as data_params.train_path and data_params.test_path. We should also set the name
  of the environment for validation in data_params.env_name ("ice_slider" for IceSlider or "digit_jump" for DigitJump).
  
- Training and evaluation are performed sequentially by running
```
python main.py
```

## Configuration

All settings can be handled by editing `default_config.json`.

| Param         | Default           | Info  |
| ------------- | ------------- | -----:|
| optimizer_params.eps | 1e-05 | epsilon for Adam |
| train_params.seed | null | seed for training |
| train_params.epochs | 40 | # of training epochs |
| train_params.batch_size | 128 | batch size for training |
| train_params.save_every_n_epochs | 5 | how often to save models |
| train_params.val_every_n_epochs | 2 | how often to perform validation |
| train_params.lr_dict | - | dictionary of learning rates for each component |
| train_params.loss_weight_dict | - | dictionary of weights for the three loss functions |
| train_params.margin | 0.1 | latent margin epsilon |
| train_params.hinge_params | - | hyperparameters for margin loss |
| train_params.schedule | [] | learning rate schedule |
| model_params.name | 'ppgs' | name of the model to train in ['ppgs', 'latent']
| model_params.load_model | true | whether to load saved model if present |
| model_params.filters | [64, 128, 256, 512] | encoder filters |
| model_params.embedding_size | 16 | dimensionality of latent space |
| model_params.normalize | true | whether to normalize embeddings |
| model_params.forward_layers | 3 | layers in MLP forward model for 'latent' world model |
| model_params.forward_units | 256 | units in MLP forward model for 'latent' world model |
| model_params.forward_ln | true | layer normalization in MLP forward model for 'latent' world model |
| model_params.inverse_layers | 1 | layers in MLP inverse model |
| model_params.inverse_units | 32 | units in MLP inverse model |
| model_params.inverse_ln | true | layer normalization in MLP inverse model |
| data_params.train_path | '' | path to training dataset |
| data_params.test_path | '' | path to validation dataset |
| data_params.env_name | 'ice_slider' | name of environment ('ice_slider' for IceSlider, 'digit_jump' for DigitJump |
| data_params.seq_len | 2 | number of steps for multi-step loss |
| data_params.shuffle | true | whether to shuffle datasets |
| data_params.normalize | true | whether to normalize observations |
| data_params.encode_position | false | enables positional encoding |
| data_params.env_params | {} | params to pass to environment |
| eval_params.evaluate_losses | true | whether to compute evaluation losses |
| eval_params.evaluate_rollouts | true | whether to compute solution rates |
| eval_params.eval_at | [1,3,4] | # of steps to evaluate at |
| eval_params.latent_eval_at | [1,5,10] | K for latent metrics |
| eval_params.seeds | [2000] | starting seed for evaluation levels |
| eval_params.num_levels | 100 | # evaluation levels |
| eval_params.batch_size | 128 | batch size for latent metrics evaluation |
| eval_params.planner_params.batch_size | 256 | cutoff for graph search |
| eval_params.planner_params.margin | 0.1 | latent margin for reidentification |
| eval_params.planner_params.early_stop | true | whether to stop when goal is found |
| eval_params.planner_params.backtrack | false | enables backtracking algorithm |
| eval_params.planner_params.penalize_visited | false | penalizes visited vertices in graph search |
| eval_params.planner_params.eps | 0 | enables epsilon greedy action selection |
| eval_params.planner_params.max_steps | 256 | maximal solution length |
| eval_params.planner_params.replan horizon | 10 | T_max for full planner |
| eval_params.planner_params.snap | false | snaps new vertices to visited ones |
| working_dir |  "results/ppgs" | directory for checkpoints and results |


