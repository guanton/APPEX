# APPEX: Alternating Projection Parameter Estimation from X0

APPEX is a Python library for estimating the drift, diffusion, and causal graph of Stochastic Differential Equations (SDEs) from observed temporal marginals. The code is based on our paper: 

## Key Features and General Usage
Currently, only time-homogeneous linear additive noise are supported.
- **Data Generation**: Generate synthetic temporal marginals from SDEs. Relevant functions are found in `data_generation.py` 
- **Visualization**: Using `plot_time_marginals.py`, you can make fun gifs of the evolutions of the temporal marginals of various SDEs to visualize identifiability/non-identifiability.
- **Experimentation**: Run experiments for parameter estimation and causal discovery using our APPEX algorithm in `experiments.py`. Plot and interpret them with `plot_experiment_results.py` 

## Installation
Clone the repository and install dependencies via:
```bash
git clone https://github.com/guanton/APPEX.git
cd APPEX
pip install -r requirements.txt
```








