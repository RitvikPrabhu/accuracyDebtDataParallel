#!/bin/bash

batchSize=10000
nRanks=4
ensembleSize=10
nEvents=1000000
nSamples=100
mainDir="/Users/daniellersch/Desktop/SciDAC/multi_datasets/mg_problem/"
coreName="results_test_N4_baseline_set2"
# mainDir=""
# coreName="results_test_N2_multidata"
ensembleLoc=$mainDir$coreName
resultLoc=$mainDir"ana_"$coreName


CMD=" eval_mg.py"
CMD=$CMD" environment.parser.alpha=[[0.9,0.1,0.0],[0.25,0.75,0.0]] environment.parser.n_events=$nEvents"
CMD=$CMD" environment.alpha=[[0.9,0.1,0.0],[0.25,0.75,0.0]] environment.n_samples=$nSamples"
CMD=$CMD" batch_size=$batchSize n_ranks=$nRanks ensemble_size=$ensembleSize"
CMD=$CMD" ensemble_loc=$ensembleLoc logdir=$resultLoc"

python $CMD