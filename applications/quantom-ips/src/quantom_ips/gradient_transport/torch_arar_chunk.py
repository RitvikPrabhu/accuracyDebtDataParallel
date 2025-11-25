from mpi4py import MPI
import numpy as np
import torch
import logging
from dataclasses import dataclass
from quantom_ips.utils.registration import register_with_hydra
from quantom_ips.gradient_transport.torch_arar import TorchARARDefaults, TorchARAR

logger = logging.getLogger(__name__)


@dataclass
class TorchARARPCGradDefaults(TorchARARDefaults):
    id: str = "TorchARARChunk"


@register_with_hydra(
    group="trainer/gradient_transport", defaults=TorchARARPCGradDefaults, name="torch_arar_chunk"
)
class TorchARARChunk(TorchARAR):
    """
    Torch Asynchronous Ring All Reduce (ARAR) with gradient chunking
    """

    # Initialize:
    # ****************************
    def __init__(
        self, config, mpi_comm=None, default_group_size=4, dtype=torch.float32
    ):  
        # Init all necessary variables:
        super().__init__(config, mpi_comm, default_group_size, dtype)
    # ****************************
    
    # Rewrite ring-all-reduce for a single tensor:
    # ****************************
    def ring_allreduce_single_tensor(self, data, mpi_comm, num_ranks, neighbours, op_is_sum=False):
        # Get original shape:
        orig_shape = data.shape
        # Get data chunks:
        data_chunks = np.array_split(np.copy(data).ravel(),num_ranks,0)

        # Get the current rank:
        rank = mpi_comm.Get_rank()

        # Scatter reduce, i.e. send around individual chunks:
        # ++++++++++++++++++
        for i in range(num_ranks - 1):
            send_chunk_idx = (rank - i) % num_ranks
            recv_chunk_idx = (rank - i - 1) % num_ranks

            # Receive data: 
            recv_data = np.zeros_like(data_chunks[recv_chunk_idx])
            recv_req = mpi_comm.Irecv(recv_data, neighbours[0])
            
            # Send data:
            send_req = mpi_comm.Isend(data_chunks[send_chunk_idx], neighbours[1])
            
            recv_req.wait()
            send_req.wait()
            
            # Accumulate chunks:
            data_chunks[recv_chunk_idx][:] += recv_data[:]
        # ++++++++++++++++++


        # Allgather:
        # Copy data so that the original chunks are not altered
        final_chunks = list(data_chunks)
        # Delete from meory:
        del data_chunks
        # ++++++++++++++++++
        for i in range(num_ranks - 1):
            send_chunk_idx = (rank - i + 1) % num_ranks
            recv_chunk_idx = (rank - i) % num_ranks

            # Receive data: 
            recv_data = np.zeros_like(final_chunks[recv_chunk_idx])
            recv_req = mpi_comm.Irecv(recv_data, neighbours[0])
            # Send data:
            send_req = mpi_comm.Isend(final_chunks[send_chunk_idx], neighbours[1]) 
            recv_req.wait()
            send_req.wait()

            final_chunks[recv_chunk_idx] = recv_data
        # ++++++++++++++++++
       
        if op_is_sum:
            return np.concatenate(final_chunks).reshape(orig_shape)
        return np.concatenate(final_chunks).reshape(orig_shape) / num_ranks
    # ****************************
    
    # Ring all-reduce for full tensor:
    # ****************************
    def ring_allreduce(self, currrent_gradients, mpi_comm, num_ranks, neighbours):
        new_gradients = {}
        # +++++++++++++++++++++++
        for key in currrent_gradients:
            if currrent_gradients[key].size > num_ranks:
               new_gradients[key] = self.ring_allreduce_single_tensor(
                  currrent_gradients[key], mpi_comm, num_ranks, neighbours
               )
            else:
                new_gradients[key] = super().ring_allreduce_single_tensor(
                  currrent_gradients[key], mpi_comm, num_ranks, neighbours
                )
        # +++++++++++++++++++++++

        return new_gradients
    # ****************************