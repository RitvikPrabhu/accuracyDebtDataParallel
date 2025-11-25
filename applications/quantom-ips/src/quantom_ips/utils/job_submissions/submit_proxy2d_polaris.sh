#!/bin/bash -l
#PBS -l select=10:ncpus=4:ngpus=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q prod
#PBS -A EE-ECP

# This script lets you run the workflow on Polaris 

# Set your environment:                                                                                                                                                                                                                                    
module use /soft/modulefiles
module load conda
source /home/dlersch/qm_ips_env/bin/activate

# Basic settings: 
mainDir="/home/dlersch/SCIDAC/ips_workflow/quantom-ips/src/quantom_ips"
driver=$mainDir"/drivers/distributed_multi_data_training_workflow.py"
dataLoc="/grand/EE-ECP/quantom_scidac/sample_data/proxy2d_app_4datasets"
outputLoc="/grand/EE-ECP/quantom_scidac/results/results_4datasets_v0"

batchSize=1000
nEpochs=100000
readEpochs=500
printEpochs=10000
snapEpochs=5000
gradEpochs=5000
outerEpochs=1000
nSamples=10000
gridSize=100

CMD="environment.parser.paths=[[$dataLoc/events_2dproxy_data0.npy],[$dataLoc/events_2dproxy_data1.npy],[$dataLoc/events_2dproxy_data2.npy],[$dataLoc/events_2dproxy_data3.npy]]"
CMD=$CMD" environment/theory=proxy2d_v2 environment.theory.a_min=[0.0,0.0,0.0] environment.theory.a_max=[0.1,0.15,0.01] environment.theory,ratios=[0.78,0.65,0.87,0.38]"
CMD=$CMD" environment.n_samples=$nSamples optimizer.layers.Upsample.config.size=[$gridSize,$gridSize]"
CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.out_channels=3"
CMD=$CMD" trainer.outer_update_epochs=$outerEpochs"
CMD=$CMD" trainer.batch_size=$batchSize"
CMD=$CMD" trainer.n_epochs=$nEpochs analysis.frequency=$readEpochs analysis.gradient_snapshot_frequency=$gradEpochs analysis.model_snapshot_frequency=$snapEpochs analysis.printout_frequency=$printEpochs"
CMD=$CMD" hydra.run.dir=$outputLoc"

mpiexec -n 40 -ppn 4 python $driver $CMD
