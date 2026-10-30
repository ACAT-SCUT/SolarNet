# SolarNet

Welcome to the official repository of the SolarNet paper: "SolarNet: Modeling Periodic Patterns Conditioned on Solar Geometry for Photovoltaic Power Forecasting".

## Getting Started

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n SolarNet python=3.8
conda activate SolarNet
pip install -r requirements.txt
```


### Data Preparation

The four multi-station PV datasets are available at [Here](https://drive.google.com/drive/folders/1AAvyRbvK38ED09p5AFVn5jEMXnHaxBpK).
Create a separate folder named ```./dataset``` and place all the CSV files in this directory.

### Training Example

You can easily reproduce the results from the paper by running the provided script command. For instance, to reproduce the main results, execute:

```
sh run_main.sh
```

## Citation

If you find this repo useful, please cite our paper.

```
@article{solarnet,
  title   = {SolarNet: Modeling Periodic Patterns Conditioned on Solar Geometry for Photovoltaic Power Forecasting},
  author  = {},
  journal = {},
  year    = {2026},
}
```


