#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A800:8
#SBATCH --time=22:00:00
#SBATCH --mem=300GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=./ana_mg_v0.log 
#SBATCH --job-name=MG
# This line is very important --> Make shell aware of your conda env
source /etc/profile.d/conda.sh 
# Activate your conda env
conda activate qm_env

# Basic settings: 
mainDir="/work/data_science/dlersch/SciDAC/ips_workflow/quantom-ips/"
driver=$mainDir"/examples/multi_gaussian/run_mg.py"

batchSize=512
nEpochs=2000 # 5000
readFrequency=25 # 25
printFrequency=400 # 1000 
outerGroupFrequency=50 # 50
snapshotFrequency=50
lrGen=1e-4
lrDisc=1e-5
nRanks=4
nEvents=1000000
nSamples=100
mode="baseline"
resultLoc="/work/data_science/dlersch/SciDAC/ips_workflow/mg_problem/results_test_N$nRanks"_"$mode"_v0

CMD=$CMD" trainer.batch_size=$batchSize trainer.n_epochs=$nEpochs trainer.gen_lr=$lrGen trainer.disc_lr=$lrDisc trainer.outer_update_epochs=$outerGroupFrequency"
CMD=$CMD" analysis.logdir=$resultLoc analysis.frequency=$readFrequency analysis.printout_frequency=$printFrequency analysis.snapshot_frequency=$snapshotFrequency"
CMD=$CMD" environment.parser.alpha=[[1.0,0.0,0.0],[0.0,1.0,0.0]] environment.parser.n_events=$nEvents"
CMD=$CMD" environment.alpha=[[1.0,0.0,0.0],[0.0,1.0,0.0]] environment.n_samples=$nSamples"
CMD=$CMD" environment.mode=$mode"

mpirun --mca btl_tcp_port_min_v4 32768 --mca btl_tcp_port_range_v4 28230 python $driver $CMD
