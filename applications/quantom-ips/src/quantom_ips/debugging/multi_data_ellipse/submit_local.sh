#!/bin/bash

nEpochs=10000
batchSize=1000
readFrequency=50
printFrequency=1000
outerGroupFrequency=50
groupSize=2
lossScale=0.0
emaALPHA=-2.0
logDir="results_ellipse_multidata_v1"
nNeuronsDisc=100
drDisc=0.0
lrGen=1e-5
lrDisc=1e-4

CMD=" python run_multi_data_ellipse.py n_epochs=$nEpochs batch_size=$batchSize read_frequency=$readFrequency print_frequency=$printFrequency outer_group_frequency=$outerGroupFrequency"
CMD=$CMD" group_size=$groupSize std_loss_scale=$lossScale logdir=$logDir ema_alpha=$emaALPHA"
CMD=$CMD" dropouts_discriminator=$drDisc lr_generator=$lrGen lr_discriminator=$lrDisc n_neurons_discriminator=$nNeuronsDisc"

mpirun -n 4 $CMD