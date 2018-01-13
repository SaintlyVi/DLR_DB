# SA Domestic Load Research Bayesian Network Inference of Customer Classes

This repository is a submodule of the [DLR Intelligent System](https://github.com/SaintlyVi/DLR_DB). It contains modules and functions for constructing Bayesian Network models from DLR data.

## Data Models

An experimental model has been set up to demonstrate the processing and inference steps.

### Experiment 1 - Simple Naive Bayes model
The model contains the following random variables:
`["monthly_income", "water_access", "roof_material", "wall_material", "cb_size", "floor_area", "geyser_nr"]`

#### Model Setup

1. _Feature Extraction:_ Specify BN nodes and search terms in `DLR_DB/clmod/bnevidence.py` and run evidence_exp1() for all years of interest (from 2000 onwards)
2. _Model Construction:_ Define BN structure, variables and probability tables in `exp1_bn.txt`
3. _Class Inference:_ Open the `DLR NOTEBOOK Class Inference` jupyter notebook from the [main module](https://github.com/SaintlyVi/DLR_DB) and follow the steps outlined to construct evidence files, run inference over the network and save output files that specify the customer class of each AnswerID.

## Repository Structure
All data input and output files are saved in a subdirectory named after the experiment.

#### Data input 
`/evidence/exp1`

.txt files with json-formatted evidence for each AnswerID

#### Data output 
`/out/exp1`

.csv files with the maximum probability inferred class assigned to each AnswerID

#### Support functions
contained in module `bnsupport`

**ensure that you apply the 2to3 fix to bntextutils.py**