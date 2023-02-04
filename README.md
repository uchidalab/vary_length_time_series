# On mini-batch training with varying length time series

This is a repository for the paper *On mini-batch training with varying length time series* submitted to ICASSP 2022.

## Requires

This was built using Python and Numpy

### Python libraries

```
pip install numpy==1.19.5 matplotlib==2.2.2 tqdm
```

### Data preparation

This repository was made with the 2018 UCR Time Series Archive datasets in mind. The datasets are 1D time series, 11 of which have varying lengths. To install the datasets, download the .zip file from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ and extract the contents into the `data` folder.

Like the UCR Time Series Archive, we assume the input is stored in the numpy array with a fixed length, but with NaN at the end of each time series.

Example:

Two sequences, one with 3 time steps and one with 2 time steps.

```
0.1, 0.2, 0.3, NaN, NaN
0.1, 0.2, 0.3, 0.4, 0.5
```

## Example Usage

```
import numpy as np
import utils.vary as vy

# Load UCR data
dataset = "GestureMidAirD1"
data_train = np.genfromtxt(os.path.join("data", dataset, "%s_TRAIN.tsv"%dataset), delimiter=" ")
data_test = np.genfromtxt(os.path.join("data", dataset, "%s_TEST.tsv"%dataset), delimiter=" ")

# Separate labels
x_data_train = data_train[:,1:]
x_data_test = data_test[:,1:]

# Run
x_data_train, x_data_test = vy.zero_pad_pre(x_data_train, x_data_test)
```

# Citation
```
@inproceedings{iwana2022on,
  title={On Mini-Batch Training with Varying Length Time Series},
  author={Iwana, Brian Kenji},
  booktitle={International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year={2022},
  doi={10.1109/ICASSP43922.2022.9746542},
}
```

