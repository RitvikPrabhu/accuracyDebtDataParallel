#!/bin/bash

batchSize=1000
nRanks=8
nReps=50
mainDir="/Users/daniellersch/Desktop/SciDAC/multi_datasets"
extension="n8_mcmc"
extensionF="n8_concrete_l4_long_temp0"
coreDataLoc="$mainDir/migan_ensemble_$extension"
resultLoc="$coreDataLoc/results_migan_$extensionF"
logDir="$coreDataLoc/eval_results_migan_$extensionF"_T50

CMD=" run_image_eval.py mode=mc"
CMD=$CMD" n_ranks=$nRanks n_repetitions=$nReps"
CMD=$CMD" batch_size=$batchSize device=mps result_loc=$resultLoc logdir=$logDir"
CMD=$CMD" environment.parser.alphas=[[0.75,0.25],[0.25,0.75]] environment.parser.relative_offset=0.1"
CMD=$CMD" environment.alphas=[[0.75,0.25],[0.25,0.75]]"
# CMD=$CMD" optimizer.dropouts=[0.0,0.0,0.0,0.02,0.02] environment.objective.dropouts=[0.0,0.0,0.0,0.02,0.02]"
# CMD=$CMD" optimizer.ngfs=[1042,512,128,64,2] environment.objective.ndfs=[64,128,512,1042,1]"
CMD=$CMD" optimizer.concrete_dropout_weight_regs=[1e-6,1e-6,1e-6,1e-6,0.0] optimizer.concrete_dropout_drop_regs=[1e-5,1e-5,1e-5,1e-5,0.0]"
CMD=$CMD" optimizer.concrete_temp=0.5"

python $CMD

rm -r outputs