#!/bin/bash

# This script lets you run the offline analysis on your local machine. 

# Basic settings: 
mainDir="/Users/daniellersch/Desktop/SciDAC/multi_datasets/quantom-ips/src/quantom_ips"
driver=$mainDir"/offline_evaluation/duke_and_owens_eval/compare_runs.py"
dataLoc="/Users/daniellersch/Desktop/SciDAC/sample_data/duke_and_owens_toy_data"
resultLoc="/Users/daniellersch/Desktop/SciDAC/multi_datasets/results_duke_and_owens_wgan_v4"
#resultLoc="/Users/daniellersch/Desktop/SciDAC/multi_datasets/quantom-ips/src/quantom_ips/utils/job_submissions/multi_dataset_local_test"
outputLoc="/Users/daniellersch/Desktop/SciDAC/multi_datasets/eval_new_results_wgan_ifarm_v4"
gridSize=100
nSamples=100000
latentDim=100

CMD="environment.parser.paths=[[$dataLoc/events_duke_and_owens_proton.npy],[$dataLoc/events_duke_and_owens_neutron.npy]]"
CMD=$CMD" data_locs=[$resultLoc] data_names=[duke_and_owens]"
CMD=$CMD" environment.sampler.log_space=True environment.theory.fixed_density_index=-1 environment.theory.a_min=[0.0,0.0] environment.theory.a_max=[20000,35000]"
CMD=$CMD" optimizer.layers.Upsample.config.size=[$gridSize,$gridSize]"
CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.out_channels=2"
CMD=$CMD" environment.n_samples=$nSamples"
CMD=$CMD" hydra.run.dir=$outputLoc"

CMD=$CMD" optimizer.layers.Linear_1.config.in_features=$latentDim optimizer.input_shape=[0,$latentDim]"
#CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.kernel_size=4 optimizer.layers.ConvTranspose2dStack.config.n_layers=5"
#CMD=$CMD" n_ranks=4"

python $driver $CMD