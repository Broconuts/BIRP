# Grid-Based Break-In Risk Prediction
This repository contains the source code for the Grid-Based Break-In Risk Prediction (GB-BIRP).

## Setup
This project was developed using Python 3.9.9. Please install requirements:

```
pip install -r requirements.txt
```

Individual experiment runs are collected in the ``experiments`` subdirectory. Experiment notebooks contain redundant code that was not abstracted into more reusable functions for convenience. Moreover, if sample weighting is to be applied in experiment 5, make sure to remove the hashtag in training function calls which is currently commenting out the passing of sample weights to the function.


## Training/Evaluation Data and Model Weights
Please note that this repository is missing some of the data required to train GB-BIRP. More specifically, the break-in data was omitted due to its sensitive nature. This means that experiments cannot be run locally unless you have access to the WED dataset provided by the Bremen Police department. 

Event and social data was collected and aggregated into the currently used format manually.

Weather data was acquired through [Meteostat](https://meteostat.net/en/).