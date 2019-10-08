# ACFlow

Code to reproduce most of the results in our work [Flow Models for Arbitrary Conditional Likelihoods](https://arxiv.org/abs/1909.06319).

## Download datasets
* NOTE: you need to change the path in the DataLoaders (inside /datasets/ folder)

## Training scripts
python scripts/train.py --cfg_file config_file_path &

* One example configuration for training on celeba is provided in /exp/celeba/test/param.json

## Quantitative evaluation
python scripts/test.py --cfg_file config_file_path &

## Sampling
python scripts/sample.py --cfg_file config_file_path &


