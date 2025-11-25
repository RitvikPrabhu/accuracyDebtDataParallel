#!/bin/bash

batchSize=1024
nEpochs=2000 # 5000
readFrequency=25 # 25
printFrequency=400 # 1000 
outerGroupFrequency=50 # 50
snapshotFrequency=100
lrGen=1e-4
lrDisc=1e-4
nRanks=2
nEvents=1000000
nSamples=100
mode="multidata"
resultLoc="results_test_N$nRanks"_"$mode"_v0

CMD=" python run_mg.py"
CMD=$CMD" trainer.batch_size=$batchSize trainer.n_epochs=$nEpochs trainer.gen_lr=$lrGen trainer.disc_lr=$lrDisc trainer.outer_update_epochs=$outerGroupFrequency"
CMD=$CMD" analysis.logdir=$resultLoc analysis.frequency=$readFrequency analysis.printout_frequency=$printFrequency analysis.snapshot_frequency=$snapshotFrequency"
CMD=$CMD" environment.parser.alpha=[[1.0,0.0,0.0],[0.0,1.0,0.0]] environment.parser.n_events=$nEvents"
CMD=$CMD" environment.alpha=[[1.0,0.0,0.0],[0.0,1.0,0.0]] environment.n_samples=$nSamples"
CMD=$CMD" environment.mode=$mode"

mpirun -n $nRanks $CMD

rm -r outputs