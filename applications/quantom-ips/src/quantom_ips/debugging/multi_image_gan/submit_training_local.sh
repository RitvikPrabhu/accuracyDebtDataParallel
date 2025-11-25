#!/bin/bash

batchSize=10
nEpochs=1000 # 5000
readFrequency=25 # 25
printFrequency=200 # 1000 
outerGroupFrequency=50 # 50
lrGen=1e-4
lrDisc=1e-5
alphaMax='[0.9,0.6,0.3]'
Ewarmup='[50,40,30]'
Erampup='[30,30,30]'
Ehold='[50,50,50]'
Erampdown='[10,10,10]'
resultLoc="results_test_v1"

CMD=" python run_multi_image_gan.py"
CMD=$CMD" trainer.batch_size=$batchSize trainer.n_epochs=$nEpochs trainer.gen_lr=$lrGen trainer.disc_lr=$lrDisc trainer.outer_update_epochs=$outerGroupFrequency"
CMD=$CMD" analysis.logdir=$resultLoc analysis.frequency=$readFrequency analysis.printout_frequency=$printFrequency"
CMD=$CMD" environment.parser.alphas=[[0.75,0.25],[0.25,0.75]] environment.parser.relative_offset=0.1"
CMD=$CMD" environment.alphas=[[0.75,0.25],[0.25,0.75]]"
CMD=$CMD" trainer.gradient_transport.pcgrad_fraction=0.5"
CMD=$CMD" trainer.alpha_max=$alphaMax trainer.E_warm_up=$Ewarmup trainer.E_ramp_up=$Erampup trainer.E_hold=$Ehold trainer.E_ramp_down=$Erampdown"
#CMD=$CMD" optimizer.concrete_dropout_weight_regs=[1e-6,1e-6,1e-6,1e-6,0.0] optimizer.concrete_dropout_drop_regs=[1e-5,1e-5,1e-5,1e-5,0.0]"
#CMD=$CMD" trainer.gradient_transport.pcgrad_threshold=-0.02 trainer.gradient_transport.loss_ema_beta=0.9"
mpirun -n 2 $CMD

rm -r outputs