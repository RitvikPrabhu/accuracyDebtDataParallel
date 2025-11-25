#/bin/bash -l
#PBS -l select=30:ncpus=4:ngpus=4:system=polaris                          
#PBS -l place=scatter
#PBS -l walltime=03:00:00      
#PBS -l filesystems=home:grand    
#PBS -q prod                                                                                                                                            
#PBS -A quantom_scidac                                                                                                                        
# Polaris has ~560 nodes in total, with 4 GPUs each. Here, we just use two GPUs for testing  
# Load conda and set your quantom environment:

module use /soft/modulefiles
module load conda
source /home/dlersch/qm_ips_env/bin/activate

# The following lines are a bit awkward, but they are needed, in case we wish use torch DDP:                                             
# Begin awkwardness:                                                                                                                          
export AWS_DIR=/soft/libraries/aws-ofi-nccl/v1.6.0/
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=$AWS_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH

export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_REQ_BUF_SIZE=16MB

export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000
export FI_CXI_RDZV_THRESHOLD=2000

masterNode=$(head -1 $PBS_NODEFILE)
masterPort=$(( RANDOM % (30000 - 11000 + 1 ) + 11000 ))

# End awkwardness....                                        

# Basic definitions / arguments to run the workflow   
ensembleSize=15
nNodes=2
offset=0
step=$((nNodes - 1))

storeDir="/grand/EE-ECP/quantom_scidac/results"
driver="/home/dlersch/SCIDAC/ips_workflow/quantom-ips/src/quantom_ips/drivers/distributed_training_workflow.py" # ADJUST to your working directory
dataLoc="/grand/EE-ECP/quantom_scidac/sample_data/events_2d_proxy_app_v0.npy"

nEpochs=100000
printEpochs=10000
recordEpochs=2000
snapEpochs=10000
gradEpochs=2000
batchSize=1000
gridSize=50
nevs=100000
updateEpochs=500
logDir=$storeDir"/results_2D_proxy_app_bs"$batchSize"_gs"$gridSize"_ep100k_nevs100k_8ranks"

# Define hydra commands:
envCMD="environment.n_samples=$nevs environment.parser.path=[$dataLoc] environment.sampler.a_max=0.1"
trainerCMD="trainer.n_epochs=$nEpochs trainer.batch_size=$batchSize trainer.outer_update_epochs=$updateEpochs trainer.master_ip=$masterNode trainer.master_port=$masterPort trainerr.backend=nccl"
optimizerCMD="optimizer.layers.Upsample.config.size=[$gridSize,$gridSize]"
anaCMD="analysis.frequency=$recordEpochs analysis.printout_frequency=$printEpochs analysis.model_snapshot_frequency=$snapEpochs analysis.gradient_snapshot_frequency=$gradEpochs"

for m in $(seq 0 $((ensembleSize-1)));
do
  
  a=$((m * nNodes + 1))
  b=$((a + step))
  idx=$((m + offset))
  
  node_config= sed -n "${a},${b}p" < $PBS_NODEFILE
  echo $node_config > newhost

  cat newhost

  resultDir=$logDir"_v"$idx
  echo "submit job "$idx
  echo "driver: "$driver
  mpiexec -n 8 -ppn 4 -hostfile newhost python $driver hydra.run.dir=$resultDir $envCMD $trainerCMD $optimizerCMD $anaCMD
  python -c "print('something is running here')"
  echo "job submitted"
  sleep 1
  rm newhost
  
done
