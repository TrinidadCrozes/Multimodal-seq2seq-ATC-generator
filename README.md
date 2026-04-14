# A multimodal sequence-to-sequence model for automatic assignment of ATC codes in drug discovery and repurposing. 

## Requirements
- Python=3.12.4
- pandas=2.2.2
- numpy=1.26.4
- torch=2.7.1+cu118
- keras=3.6.0
- tensorflow=2.17.0
- scikit-learn=1.5.2
- matplotlib=3.10.5
- tqdm=4.66.4
- mordredcommunity[full]=2.0.6

## Data

Our dataset is available in the [data](https://github.com/TrinidadCrozes/Multimodal-seq2seq-ATC-generator/tree/main/data) directory.

The complete list of molecular descriptors is available in the [descriptors_calc/3203_descriptors.csv](https://github.com/TrinidadCrozes/Multimodal-seq2seq-ATC-generator/blob/main/descriptors_calc/3203_descriptors.csv) file.

## Experiments

The experiments in this directory are divided in:

### Scenario 1: Prediction of ATC codes for new compounds
The directory [experiments/new_compounds](https://github.com/TrinidadCrozes/Multimodal-seq2seq-ATC-generator/tree/main/experiments/new_compounds) contains:
- The data partitions used to train, validate and test the models in this scenario (available in the Datasets folder).
- A folder for each baseline and proposed method, containing:
  -  the source code for the hyperparameter optimization,
  -  the source code for model training and evaluation, and
  -  a CSV file with the experimental results.

**Meta-model**

The implementation of the meta-model and a CSV file containing the metric results for this scenario can be found in the [experiments/meta-model](https://github.com/TrinidadCrozes/Multimodal-seq2seq-ATC-generator/tree/main/experiments/meta-model) directory.

**Variation of the number of predicted ATC codes from one to ten**

This folder follows a similar structure as in the new_compounds directory. 
The [experiments/varying_#predictions](https://github.com/TrinidadCrozes/Multimodal-seq2seq-ATC-generator/tree/main/experiments/varying_%23predictions) directory contains:
- A folder for each baseline and proposed method, containing the source code for model evaluation when predicting from one to ten ATC codes,
- A script used to generate the plots for each metric, and
- A `figs` folder containing the resulting figures.

### Scenario 2: Prediction of ATC codes for drug repurposing
The directory [experiments/drug_repurposing](https://github.com/TrinidadCrozes/Multimodal-seq2seq-ATC-generator/tree/main/experiments/drug_repurposing) contains:
- The data partitions used to train, validate and test the models in this scenario (available in the Datasets folder).
- A folder for each baseline and proposed method, containing:
  -  the source code for the hyperparameter optimization,
  -  the source code for model training and evaluation, and
  -  a CSV file with the experimental results.

**Meta-model**

The implementation of the meta-model and a CSV file containing the metric results for this scenario can be found in the [experiments/meta-model_repurposing](https://github.com/TrinidadCrozes/Multimodal-seq2seq-ATC-generator/tree/main/experiments/meta-model_repurposing) directory.

**Variation of the number of predicted ATC codes from one to ten**

The [experiments/varying_#predictions_repurposing](https://github.com/TrinidadCrozes/Multimodal-seq2seq-ATC-generator/tree/main/experiments/varying_%23predictions_repurposing) directory contains:
- A folder for each baseline and proposed method, containing the source code for model evaluation when predicting from one to ten ATC codes,
- A script used to generate the plots for each metric, and
- A `figs` folder containing the resulting figures.

## Seq2seq

The source code in this package was taken and modified from the [pytorch_beam_search](https://github.com/jarobyte91/pytorch_beam_search) package.

The original implementation was adapted to support ATC code generation and we added the multimodal architecture proposed in our work and the source code for the performance metrics.
