# Fengshun
---
Fengshun system, an AI-driven global subseasonal-to-seasonal prediction system developed by the CMA. Fengshun features a flow-dependent intelligent perturbation generation module in the latent space of the model, incorporates air-sea interaction, and offers interpretability. Trained on CRA-40 reanalysis data, CODAS SST data, and FY-3E OLR data, it daily generates 100 ensemble members for 60-day forecasts of global basic elements and circulation.

# Version
---
CMA-AIM-S2S-Fengshun-V1.0

# Data and Model Download
---
The downloaded files shall be organized as the following hierarchy:

```
├── root
│   ├── data
│   │   ├── mask.nc
│   │   ├── sample
│   │   │   ├── geopotential.nc
│   │   │   ├── temperature.nc
│   │   │   ├── ......
│   │   │   └── total_precipitation.nc
│   │
│   ├── model
│   │   └── Fengshun_v1.0.onnx
│   │
│   ├── inference.py
│   └── data_util.py
```

# Quick Start
---
1. Install xarray  
   `conda install -c conda-forge xarray dask netCDF4 bottleneck`

2. Install pytorch  
   `conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia`

3. Usage
```
python inference.py
    --model model/fuxi_s2s.onnx
    --input data/input.nc
    --total_step 42
    --total_member 11
    --save_dir output
```

5. Input preparation
The input.nc file contains preprocessed data from the origin ERA5 files.
The file has a shape of (2, 76, 121, 240), where the first dimension represents two time steps. The second dimension represents all variable and level combinations, named in the following exact order:
```python
['z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300',
 'z250', 'z200', 'z150', 'z100', 'z50', 't1000', 't925', 't850',
 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150',
 't100', 't50', 'u1000', 'u925', 'u850', 'u700', 'u600', 'u500',
 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50', 'v1000',
 'v925', 'v850', 'v700', 'v600', 'v500', 'v400', 'v300', 'v250',
 'v200', 'v150', 'v100', 'v50', 'q1000', 'q925', 'q850', 'q700',
 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100',
 'q50', 't2m', 'd2m', 'sst', 'ttr', '10u', '10v', '100u', '100v',
 'msl', 'tcwv', 'tp']
```
