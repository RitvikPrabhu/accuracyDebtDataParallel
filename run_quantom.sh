# python -m quantom_ips.drivers.training_workflow "environment.parser.path=[applications/quantom-ips/data/data.npy]"

PMIX_MCA_gds=hash mpirun -np 2 python -m quantom_ips.drivers.distributed_training_workflow "environment.parser.path=[applications/quantom-ips/data/data.npy]" 


torchrun --nproc_per_node=2 -m quantom_ips.drivers.ddp_training_workflow "environment.parser.path=[applications/quantom-ips/data/data.npy]" 

PMIX_MCA_gds=hash horovodrun -np 2 python -m quantom_ips.drivers.hvd_training_workflow "environment.parser.path=[applications/quantom-ips/data/data.npy]"

PMIX_MCA_gds=hash mpirun -np 2 python -m quantom_ips.drivers.ddp_mpi_training_workflow "environment.parser.path=[applications/quantom-ips/data/data.npy]"