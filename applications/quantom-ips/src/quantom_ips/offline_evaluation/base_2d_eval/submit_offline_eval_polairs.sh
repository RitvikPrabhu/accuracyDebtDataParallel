#!/bin/bash -l                                                                                                                                                                                    
#PBS -l select=1:ncpus=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:grand
#PBS -q preemptable
#PBS -A EE-ECP 
# This script lets you run the workflow on Polaris 
# Set your environment:
module use /soft/modulefiles
module load conda
source /home/dlersch/qm_ips_env/bin/activate

# This script lets you run the offline analysis on your local machine. 

# Basic settings: 
mainDir="/home/dlersch/SCIDAC/ips_workflow/quantom-ips/src/quantom_ips"
driver=$mainDir"/offline_evaluation/base_2d_eval/compare_runs.py"
dataLoc="/grand/EE-ECP/quantom_scidac/sample_data/proxy2d_app_4datasets"
resultLoc="/grand/EE-ECP/quantom_scidac/results/results_4datasets_v0"
outputLoc="/grand/EE-ECP/quantom_scidac/results/eval_results_4datasets_v0"
gridSize=100
nSamples=100000
latentDim=100

CMD=" environment/theory=proxy2d_v2 environment.parser.paths=[[$dataLoc/events_2dproxy_data0.npy],[$dataLoc/events_2dproxy_data1.npy],[$dataLoc/events_2dproxy_data2.npy],[$dataLoc/events_2dproxy_data3.npy]]"
CMD=$CMD" data_locs=[$resultLoc] data_names=[proxy2d_data]"
CMD=$CMD" environment.theory.a_min=[0.0,0.0,0.0] environment.theory.a_max=[0.1,0.15,0.01] environment.theory.ratios=[0.78,0.65,0.87,0.38]"
CMD=$CMD" optimizer.layers.Upsample.config.size=[$gridSize,$gridSize]"
CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.out_channels=3"
CMD=$CMD" environment.n_samples=$nSamples"
CMD=$CMD" hydra.run.dir=$outputLoc"

CMD=$CMD" optimizer.layers.Linear_1.config.in_features=$latentDim optimizer.input_shape=[0,$latentDim]"
CMD=$CMD" n_ranks=60 devices=cuda"

python $driver $CMD
