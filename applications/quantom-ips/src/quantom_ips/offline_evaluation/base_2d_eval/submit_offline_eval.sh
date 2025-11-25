#!/bin/bash

# This script lets you run the offline analysis on your local machine. 

# Basic settings: 
mainDir="/Users/daniellersch/Desktop/SciDAC/multi_datasets/quantom-ips/src/quantom_ips"
driver=$mainDir"/offline_evaluation/base_2d_eval/run_eval.py"
dataLoc="/Users/daniellersch/Desktop/SciDAC/sample_data/proxy2d_app_4datasets"
# resultLoc="/Users/daniellersch/Desktop/SciDAC/multi_datasets/results_proxy2d_v31"
# outputLoc="/Users/daniellersch/Desktop/SciDAC/multi_datasets/eval_proxy2d_ifarm_v31_new2"
resultLoc="/Users/daniellersch/Desktop/SciDAC/multi_datasets/multi_dataset_local_test"
outputLoc="/Users/daniellersch/Desktop/SciDAC/multi_datasets/eval_multi_dataset_local_test_t1"
gridSize=100
nSamples=100000
latentDim=100
batchSize=100
ensembleSize=2

CMD=" environment/theory=proxy2d_v2 environment.parser.paths=[[$dataLoc/events_2dproxy_data0.npy],[$dataLoc/events_2dproxy_data1.npy],[$dataLoc/events_2dproxy_data2.npy],[$dataLoc/events_2dproxy_data3.npy]]"
CMD=$CMD" data_locs=[$resultLoc] data_names=[proxy2d_data] environment.parser.keep_data_off_device=True"
CMD=$CMD" environment.theory.a_min=[0.0,0.0,0.0] environment.theory.a_max=[0.1,0.15,0.01] environment.theory.ratios=[0.78,0.65,0.87,0.38]"
CMD=$CMD" optimizer.layers.Upsample.config.size=[$gridSize,$gridSize]"
CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.out_channels=3"
CMD=$CMD" environment.n_samples=$nSamples n_densities=3 batch_size=$batchSize"
CMD=$CMD" hydra.run.dir=$outputLoc ensemble_size=$ensembleSize"

CMD=$CMD" optimizer.layers.Linear_1.config.in_features=$latentDim optimizer.input_shape=[0,$latentDim]"
#CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.kernel_size=4 optimizer.layers.ConvTranspose2dStack.config.n_layers=5"
CMD=$CMD" n_ranks=6 devices=mps"

python $driver $CMD