#!/bin/bash -l
#PBS -l select=2:ncpus=4:ngpus=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A EE-ECP

# This script lets you run the workflow on Polaris 

# Set your environment:                                                                                                                                                                                                                                    
module use /soft/modulefiles
module load conda
source /home/dlersch/qm_ips_env/bin/activate

# Basic settings: 
mainDir="/home/dlersch/SCIDAC/ips_workflow/quantom-ips/src/quantom_ips"
driver=$mainDir"/drivers/distributed_multi_data_training_workflow.py"
dataLoc="/grand/EE-ECP/quantom_scidac/sample_data/duke_and_owens_toy_data"
outputLoc="/grand/EE-ECP/quantom_scidac/results/multi_dataset_polaris_test"

batchSize=1000
nEpochs=500
readEpochs=5
printEpochs=100
snapEpochs=50
gradEpochs=50
nSamples=50000
gridSize=50

CMD="environment.parser.paths=[[$dataLoc/events_duke_and_owens_proton.npy],[$dataLoc/events_duke_and_owens_neutron.npy]]"
CMD=$CMD" environment.parser.n_randomly_chosen_sets=1 environment.parser.chosen_data_set_idx=[]"
CMD=$CMD" environment.sampler.log_space=True environment.theory.fixed_density_index=-1 environment.theory.a_min=0.0 environment.theory.a_max=17000"
CMD=$CMD" environment.n_samples=$nSamples optimizer.layers.Upsample.config.size=[$gridSize,$gridSize]"
CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.out_channels=2"
CMD=$CMD" trainer.batch_size=$batchSize environment.objective.normalize_inputs=True"
CMD=$CMD" trainer.n_epochs=$nEpochs analysis.frequency=$readEpochs analysis.gradient_snapshot_frequency=$gradEpochs analysis.model_snapshot_frequency=$snapEpochs analysis.printout_frequency=$printEpochs"
CMD=$CMD" hydra.run.dir=$outputLoc"

mpiexec -n 8 -ppn 4 python $driver $CMD
