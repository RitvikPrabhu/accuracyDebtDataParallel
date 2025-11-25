#!/bin/bash

# This script lets you run the workflow on your local machine. 

# Basic settings: 
mainDir="/Users/daniellersch/Desktop/SciDAC/multi_datasets/quantom-ips/examples/dalitz_analysis"
dataLoc="/Users/daniellersch/Desktop/SciDAC/sample_data/dalitz_v0"
maskLoc="$dataLoc/dalitz_ortho"
nRanks=6
mode="multidata"
nVars=2
outputLoc="results_test_N$nRanks"_$mode

nEpochs=500
batchSize=100
freq=10
gradFreq=50
modelFreq=50
printFreq=100
outerFreq=20
gridSize=50
nSamples=1000


CMD="python $mainDir/run_dalitz_analysis.py"
CMD=$CMD" hydra.run.dir=$outputLoc"
CMD=$CMD" environment.mode=$mode"
CMD=$CMD" environment.parser.paths=[[$dataLoc/dalitz_data_set0.npy],[$dataLoc/dalitz_data_set1.npy],[$dataLoc/dalitz_data_set2.npy]]"
CMD=$CMD" environment.theory.ratios=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]] environment.n_samples=$nSamples"
CMD=$CMD" environment.theory.grid_size=$gridSize environment.theory.mask_dir=$maskLoc"
CMD=$CMD" environment.objective.layers.LinearStack.config.n_inputs=$nVars"
CMD=$CMD" trainer.batch_size=300 trainer.n_epochs=500 trainer.outer_update_epochs=20"
CMD=$CMD" analysis.frequency=10 analysis.gradient_snapshot_frequency=50 analysis.model_snapshot_frequency=50 analysis.printout_frequency=100"

mpirun -n $nRanks $CMD