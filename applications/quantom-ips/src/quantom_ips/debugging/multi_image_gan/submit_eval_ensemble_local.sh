#!/bin/bash

batchSize=1000
nRanks=8
ensembleSize=5
mainDir="/Users/daniellersch/Desktop/SciDAC/multi_datasets"
extension="n8_pcgrad"
extensionF="n8_align"
coreDataLoc="$mainDir/migan_ensemble_$extension"
resultLoc="$coreDataLoc/results_migan_$extensionF"
logDir="$coreDataLoc/eval_results_migan_$extensionF"

CMD=" run_image_eval.py"
#CMD=" run_image_eval.py mode=single single_idx=3"
CMD=$CMD" n_ranks=$nRanks ensemble_size=$ensembleSize"
CMD=$CMD" batch_size=$batchSize device=mps result_loc=$resultLoc logdir=$logDir"
CMD=$CMD" environment.parser.alphas=[[0.75,0.25],[0.25,0.75]] environment.parser.relative_offset=0.1"
CMD=$CMD" environment.alphas=[[0.75,0.25],[0.25,0.75]]"

python $CMD

rm -r outputs