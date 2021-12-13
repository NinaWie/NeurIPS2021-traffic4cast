# Traffic forecasting on traffic movie snippets

This repo contains all code to reproduce our approach to the IARAI [Traffic4cast 2021](https://www.iarai.ac.at/traffic4cast/) challenge. Our approach was presented at the NeurIPS competition workshop 2021 and the slides of our presentation can be found in this [PDF](presentation_patch_based_approach.pdf). The talk can also be found on [Youtube](https://www.youtube.com/watch?v=YHW70ZAO7b0) (at around 1:45h).

In the challenge, traffic data is provided in movie format, i.e. a rasterised map with volume and average speed values evolving over time.
The code is based on (and forked from) the code provided by the competition organizers, which can be found [here](https://github.com/iarai/NeurIPS2021-traffic4cast). 
For further information on the data and the challenge we also refer to the competition [Website](https://www.iarai.ac.at/traffic4cast/) or [GitHub](https://github.com/iarai/NeurIPS2021-traffic4cast).

### Installation and setup

To install the repository and all required packages, run

```bash
git clone https://github.com/NinaWie/NeurIPS2021-traffic4cast.git
cd NeurIPS2021-traffic4cast

conda env update -f environment.yaml
conda activate t4c

export PYTHONPATH="$PYTHONPATH:$PWD"
```

Instructions on installation with GPU support can be found in the [yaml file](environment.yaml). 


To reproduce the results and train or test on the original data, download the [data](https://www.iarai.ac.at/traffic4cast/forums/forum/competition/competition-2021/) and extract it to the subfolder `data/raw`.

## Test model

Download the weights of our best model [here](https://polybox.ethz.ch/index.php/s/aBvfKzOFkSsSUQv) and put it in a new folder named `trained_model` in the main directory. The path to the checkpoint should now be `NeurIPS2021-traffic4cast/trained_models/ckpt_upp_patch_d100.pt`. 

To create a submission on the test data, run

```
DEVICE=cpu
DATA_RAW_PATH="data/raw"
STRIDE=10

python baselines/baselines_cli.py --model_str=up_patch --resume_checkpoint='trained_models/ckpt_upp_patch_d100.pt' --radius=50 --stride=$STRIDE --epochs=0 --batch_size=1 --num_workers=0 --data_raw_path=$DATA_RAW_PATH --device=$DEVICE --submit
```

**Notes:**
* For our best submission (score 59.93) a stride of 10 is used. This means that patches are extracted from the test data in a very densely overlapping manner. However, much more patches per sample have to be predicted and the runtime thus increases significantly. We thus recommend to use a stride of 50 for testing (score 60.13 on leaderboard).
* In our paper, we define *d* as the side length of each patch. In this codebase we set a *radius* instead. The best performing model was trained with radius 50 corresponding to *d=100*. 
* The `--submit`-flag was added to the arguments to be called whenever a submission should be created.

## Train

To train a model from scratch with our approach, run

```
DEVICE=cpu
DATA_RAW_PATH="data/raw"

python baselines/baselines_cli.py --model_str=up_patch --radius=50 --epochs=1000 --limit=100 --val_limit=10 --batch_size=8 --checkpoint_name='_upp_50_retrained' --num_workers=0 --data_raw_path=$DATA_RAW_PATH --device=$DEVICE
```
**Notes:**
* The model will be saved in a folder called `ckpt_upp_50_retrained`, as specified with the `checkpoint_name` argument. The checkpoints will be saved every 50 epochs **and** whenever a better validation score is achieved (`best.pt`). Later, training can be resumed (or the model can be tested) by setting `--resume_checkpoint='ckpt_upp_50_retrained/best.pt'`.
* No submission will be created after the run. Add the flag `--submit` in order to create a submission
* The stride argument is not necessary for training, since it is only relevant for test data. The validation MSE is computed on the patches, not a full city.
* In order to use our dataset, the number of workers must be set to 0. Otherwise, the random seed will be set such that the same files are loaded for every epoch. This is due to the setup of the `PatchT4CDataset`, where files are randomly loaded every epoch and then kept in memory.

## Reproduce experiments

In our short paper, further experiments comparing model architectures and different strides are shown. To reproduce the experiment on stride values, execute the following steps:
* Run `python baselines/naive_shifted_stats.py` to create artifical test data from the city Antwerp
* Adapt the paths in the [script](test_script.py)
* Run `python test_script.py`
* Analyse the output csv file results_test_script.csv

For the other experiments, we regularly write training and validation losses to a file `results.json` during training (file is stored in the same folder as the checkpoints).


## Other approaches

* In [naive_shifted_stats](baselines/naive_shifted_stats.py) we have implemented a naive approach to the temporal challenge, namely using averages of the previous year and adapting the values to 2020 with a simple factor dependent on the shift of the input hour. The statistics however first have to be computed for each city.
* In the [configs](baselines/baselines_configs.py) file further options were added, for example `u_patch` which is the normal U-Net with patching, and models from the `segmentation_models_pytorch (smp)` PyPI package. For the latter, smp must be installed with `pip install segmentation_models_pytorch`. 