#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A800:2
#SBATCH --time=22:00:00
#SBATCH --mem=300GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=./ana_image_gan_v0.log 
#SBATCH --job-name=QM2Dapp
# This line is very important --> Make shell aware of your conda env
source /etc/profile.d/conda.sh 
# Activate your conda env
conda activate qm_env

# Basic settings: 
mainDir="/work/data_science/dlersch/SciDAC/ips_workflow/quantom-ips/src/quantom_ips"
driver=$mainDir"/debugging/multi_image_gan/run_multi_image_gan.py"
resultLoc="/work/data_science/dlersch/SciDAC/ips_workflow/results_migan_n2_v0"

batchSize=1000
nEpochs=20000 
readFrequency=200
printFrequency=2000  
outerGroupFrequency=100 
lrGen=1e-4
lrDisc=1e-5

CMD=$CMD" trainer.batch_size=$batchSize trainer.n_epochs=$nEpochs trainer.gen_lr=$lrGen trainer.disc_lr=$lrDisc trainer.outer_update_epochs=$outerGroupFrequency"
CMD=$CMD" analysis.logdir=$resultLoc analysis.frequency=$readFrequency analysis.printout_frequency=$printFrequency"
CMD=$CMD" environment.parser.ratios=[0.75,0.75]  environment.parser.relative_offset=0.1"

mpirun --mca btl_tcp_port_min_v4 32768 --mca btl_tcp_port_range_v4 28230 python $driver $CMD
