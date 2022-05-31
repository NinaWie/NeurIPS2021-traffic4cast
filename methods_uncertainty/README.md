# Aleatoric uncertainty estimation in movie-based traffic forecasting

We employ different uncertainty quantification (UQ) approaches and investigates their application to movie-based traffic speed prediction. Specifically, we use our patch-based
approach as an aleatoric UQ method based on the variance of patch-wise predictions. We further compare the patch-based UQ to other methods, namely loss attenuation, test time augmentation (TTA), and a combination of patches+TTA.

The main script to compute and evaluate uncertainty estimates can be found in the main folder, as [uncertainty_evaluation.py](../uncertainty_evaluation.py). Individual UQ methods are implemented in the classes in this folder. 

### Installation:

Please follow the installation instructions in the [main README](../README.md). 

The only package that is required additionally is `torchvision`. Install with
```
pip install torchvision
```

### Download pretrained models & other input data

The pretrained models and necessary input data can be downloaded from [here](https://polybox.ethz.ch/index.php/s/c4JkFaiYhxwAjX1). Download and unzip the folder `unc_data_folder.zip`. It contains metadata to reproduce our experiments, as well as the trained models.

We provide pretrained models for the three implemented UQ methods. All models are trained only on the 2019 data from all cites except for Antwerp, and they were validated on the first half of the 2020 data and tested on the second half.
There are three models:

* `patch.pt`: This model was trained on patches of size 100 x 100. It is used for patch-based UQ.
* `bayes_unet.pt`: This model was trained with loss attenuation, i.e. a negative log likelihood loss, to output the predicted mean and variance. The output is therefore twice as large as for the other models.
* `base_unet.pt`: This is just a standard U-Net model trained on the whole-city input. It is used for test time augmentation.

Apart from the provided data in `unc_data_folder`, you also need the data from Antwerp and Bangkok to reproduce our results. For that purpose, download the Traffic4cast data [here](https://www.iarai.ac.at/traffic4cast/2021-competition/challenge/#data), and move the folders `ANTWERP` and `BANGKOK` into the folder `unc_data_folder`. This way, all inputs and pretrained models are located in this `unc_data_folder`.

## Run and evaluate uncertainty estimation

Our main evaluation script has the following options:

```
usage: uncertainty_evaluation.py [-h] [-u UQ_METHOD] [-d DATA_PATH] [-o OUT_NAME] [-c CITY] [-a METACITY] [-r RADIUS] [-s STRIDE] [--device DEVICE] [--calibrate]

optional arguments:
  -h, --help            show this help message and exit
  -u UQ_METHOD, --uq_method UQ_METHOD
  -d DATA_PATH, --data_path DATA_PATH
  -o OUT_NAME, --out_name OUT_NAME
  -c CITY, --city CITY
  -a METACITY, --metacity METACITY
  -r RADIUS, --radius RADIUS
  -s STRIDE, --stride STRIDE
  --device DEVICE
  --calibrate
```

Our experiments have two parts: 1) The calibration of the prediction interval quantiles, and 2) the actual evaluation on the test set. **For each UQ method the quantiles must first be calibrated!** . 

### Step 1: Calibration on validation set:
For example, to calibrate the patch based approach, run the following:

``` 
python uncertainty_evaluation.py --data_path unc_data_folder -uq_method patch --city BANGKOK --out_name unc_eval --calibrate
```
As explanation: You must specify the path to the `unc_data_folder`, then select the right UQ method (here `patch`), then the city used for testing, and the name of the output folder. The output folder will also be created within the `unc_data_folder`!
Executing this command will create the folder `unc_data_folder/unc_eval` and save the calibration results there as `speed_quantiles.npy` and `vol_quantiles.npy`.

### Step 2: Evaluating on the test set:

After calibration is done, execute the same command without the `--calibrate` flag, for example
``` 
python uncertainty_evaluation.py --data_path unc_data_folder -uq_method patch --city BANGKOK --out_name unc_eval
```
This command will now create two subfolders `unc_data_folder/unc_eval/speed` and `unc_data_folder/unc_eval/vol` and all results will be saved there.

The same steps can be executed in the same way for the other UQ methods, with the flags `--uq_method=attenuation` for loss attenuation, or `--uq_method=tta` for TTA. All results will be saved in the `unc_eval` folder.

## Reproduce figures

We use the results in the `unc_eval` folder to produce the figures in our paper. To skip the steps above, you can also download our results [here](https://polybox.ethz.ch/index.php/s/JTTPZ058dgySiN2). Our code to create the figures and tables is provided in [this notebook](uncertainty_evaluation_notebook.ipynb)