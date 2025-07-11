# RAPID
## Usage

### Setup

First, setup the python environment setup. We use [Poetry](https://python-poetry.org/) for this:

```
poetry install
```

Then you could run a test configuration to see if everything works:

```
poetry run python -m ocl.train tests/configs/test_dummy_image.yml
```

Second, to download the datasets used in this work, follow the instructions in [data/README.md](data/README.md).
By default, datasets are expected to be contained in the folder `./data`.

### Training

Run one of the configurations in `configs/image`, for example:

```
poetry run python -m ocl.train configs/image/coco_base14_rapid.yml
```

The results are stored in a folder created under the log root folder (by defaults `./logs`, changeable by the argument `--log-dir`).
If you want to continue training from a previous run, you can use the `--continue` argument, like in the following command:

```
poetry run python -m ocl.train --continue <path_to_log_dir_or_checkpoint_file> configs/image/coco_base14_rapid.yml
```

### Inference
If you want to run one of the released checkpoints (see below) on your own video you can use inference script with corresponding config file:

```
poetry run python -m ocl.inference --config configs/inference/coco.yml
```
in the released config, please change `checkpoint: path/to/rapid-movi-c.ckpt` to the real path to your checkpoint.
For different video formats you would need to modify corresponding transformations in `build_inference_transform` function.

# Acknowledgement
The code uses resources from [VideoSAUR](https://github.com/martius-lab/videosaur/tree/main). We thank authors of this wonderful project for open-sourcing their work.