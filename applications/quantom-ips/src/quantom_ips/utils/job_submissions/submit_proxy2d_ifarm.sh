#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:4
#SBATCH --time=22:00:00
#SBATCH --mem=300GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --output=./ana_2dproxy_v0.log 
#SBATCH --job-name=QM2Dapp
# This line is very important --> Make shell aware of your conda env
source /etc/profile.d/conda.sh 
# Activate your conda env
conda activate qm_env

# Basic settings: 
mainDir="/work/data_science/dlersch/SciDAC/ips_workflow/quantom-ips/src/quantom_ips"
driver=$mainDir"/drivers/distributed_multi_data_training_workflow.py"
dataLoc="/work/data_science/quantom/data/proxy2d_data"
resultLoc="/work/data_science/dlersch/SciDAC/ips_workflow/results_proxy2d_v0"

nEpochs=40000
batchSize=1000
readEpochs=100
printEpochs=4000
gradEpochs=2000
snapEpochs=2000
nevs=100000
gridSize=100

CMD="environment.parser.paths=[[$dataLoc/events_2dproxy_data0.npy],[$dataLoc/events_2dproxy_data1.npy]]"
CMD=$CMD" environment.parser.n_randomly_chosen_sets=1 environment.parser.chosen_data_set_idx=[]"
CMD=$CMD" environment/theory=proxy2d environment.sampler.log_space=True"
CMD=$CMD" environment.n_samples=$nevs optimizer.layers.Upsample.config.size=[$gridSize,$gridSize]"
CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.out_channels=2"
CMD=$CMD" trainer.batch_size=$batchSize"
CMD=$CMD" trainer.n_epochs=$nEpochs analysis.frequency=$readEpochs analysis.gradient_snapshot_frequency=$gradEpochs analysis.model_snapshot_frequency=$snapEpochs analysis.printout_frequency=$printEpochs"
CMD=$CMD" hydra.run.dir=$resultLoc"
CMD=$CMD" trainer.generator_update_frequency=5 trainer=distributed_wgan_trainer trainer.gradient_penalty_scale=10.0 trainer.disc_lr=0.00001"

mpirun --mca btl_tcp_port_min_v4 32768 --mca btl_tcp_port_range_v4 28230 python $driver $CMD
