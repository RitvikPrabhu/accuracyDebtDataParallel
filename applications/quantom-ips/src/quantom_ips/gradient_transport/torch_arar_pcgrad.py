from mpi4py import MPI
import numpy as np
import torch
import logging
from typing import Optional
from dataclasses import dataclass
from quantom_ips.utils.registration import register_with_hydra
from quantom_ips.gradient_transport.torch_arar import TorchARARDefaults, TorchARAR


@dataclass
class TorchARARPCGradDefaults(TorchARARDefaults):
    id: str = "TorchARARPCGrad"
    pcgrad_th: float = -0.01
    pcgrad_severity_range: float = 0.5
    pcgrad_fraction: float = 0.75
    pcgrad_scale_max: float = 1.0
    min_norm: float = 1e-9
    epsilon: float = 1e-12


@register_with_hydra(
    group="trainer/gradient_transport", defaults=TorchARARPCGradDefaults, name="torch_arar_pcgrad"
)
class TorchARARPCGrad(TorchARAR):
    """
    Torch Asynchronous Ring All Reduce (ARAR) with PCGrad option
    """

    # Initialize:
    # ****************************
    def __init__(
        self, config, mpi_comm=None, default_group_size=4, dtype=torch.float32
    ):  
        # Init all necessary variables:
        super().__init__(config, mpi_comm, default_group_size, dtype)
        # Init PCGrad related functions:
        self.pcgrad_th = config.pcgrad_th
        self.pcgrad_severity_range = config.pcgrad_severity_range
        self.pcgrad_fraction = config.pcgrad_fraction
        self.pcgrad_scale_max = config.pcgrad_scale_max
        self.min_norm = config.min_norm
        self.epsilon = config.epsilon

        self._ws = {}
    # ****************************

    def _get_workspace(self, shape, dtype):
      """Return a dict of reusable buffers for a given shape and dtype."""
      key = (tuple(shape), np.dtype(dtype).str)
      ws = getattr(self, "_ws", None)
      if ws is None:
        self._ws = {}
        ws = self._ws
      if key not in ws:
        # allocate once
        ws[key] = {
            "send": np.empty(shape, dtype=dtype),
            "recv": np.empty(shape, dtype=dtype),
            "accum": np.empty(shape, dtype=dtype),
            "disagree": np.zeros(shape, dtype=dtype),
        }
      return ws[key]
    
    # Normalize tensor, i.e. get unit-tensor:
    # ****************************
    def get_unit_tensor(self,tensor):
        return tensor / (self.epsilon + np.linalg.norm(tensor))
    # ****************************
    
    # Ring allreduce for a single tensor, but we also monitor the aligment between gradients:
    # ****************************
    def ring_allreduce_single_tensor(self,data, mpi_comm, num_ranks, neighbours, skip_alignment_check=False):
        # Run 'traditional' ring-allreduce, if we are not interested in alignment or alignment checks:
        if skip_alignment_check:
            return super().ring_allreduce_single_tensor(data, mpi_comm, num_ranks, neighbours, False)
        
        send_data = np.copy(data)
        recv_data = np.empty_like(data)
        accum_data = np.copy(data)
        disagreement_data = np.zeros_like(data)

        accum_cosine_similarity = 0.0
        mon_cosine_similarity = np.zeros(2)
        counts = np.zeros(dtype=np.int32,shape=2)
        n_disagreements = 0
        local_norm = np.linalg.norm(data) + 1e-12
        
        # ++++++++++++++++++++++++
        for i in range(num_ranks-1):
            # Receive data from "left" neighbour:
            recv_req = mpi_comm.Irecv(recv_data, source=neighbours[0],tag=i)
            # Send data to "right" neighbour:
            send_req = mpi_comm.Isend(send_data, dest=neighbours[1],tag=i)
            # Make sure we receive the incoming data so that we can continue with our computation:
            recv_req.wait()
            accum_data[:] += recv_data
            # Copy recv_data just to avoid any delay in the communication
            copy_recv_data = np.copy(recv_data)
            # Compute cosine similarity and collect data based on misalignment:
            recv_norm = np.linalg.norm(copy_recv_data) + 1e-12
            copy_recv_data = np.copy(copy_recv_data)
            cos = np.dot(data, copy_recv_data) / (local_norm * recv_norm)
            if cos >= 0.0:
                mon_cosine_similarity[0] += cos
                counts[0] += 1
            if cos < 0.0:
                mon_cosine_similarity[1] += cos
                counts[1] += 1
            if cos < self.pcgrad_th:
                accum_cosine_similarity += cos
                n_disagreements += 1
                np.add(disagreement_data, copy_recv_data, out=disagreement_data)
            # Wait for the current data to be sent, before updating it:
            send_req.wait()
            send_data[:] = recv_data
            
        # ++++++++++++++++++++++++

        mon_cosine_similarity[0] = mon_cosine_similarity[0] / (self.epsilon + counts[0])
        mon_cosine_similarity[1] = mon_cosine_similarity[1] / (self.epsilon + counts[1])

        # Average accumulated cosine similarity over disagreements:
        avg_cosine_similarity = accum_cosine_similarity / (self.epsilon + n_disagreements)

        # Get projection flag: Incorporate safety measures, so that we do not accidentally 
        # correct a good gradient:
        # 1.) Make sure that we have "enough" disagreements that justify a change:
        trigger_1 = n_disagreements >= int(self.pcgrad_fraction*num_ranks)
        # 2.) Make sure that we do not have a very small projection gradient 'disagreement_data'
        # --> Do not want to introduce additional noise:
        trigger_2 = np.linalg.norm(disagreement_data) > self.min_norm
        # 3.) Formulate the trigger:
        projection_trigger = trigger_1 & trigger_2

        return accum_data/num_ranks, disagreement_data/(self.epsilon + n_disagreements), avg_cosine_similarity, projection_trigger, mon_cosine_similarity 
    # ****************************
    
    # Run projection on local gradients:
    # ****************************
    def project(self,data, disagreement_data, cosine_similarity, trigger, scale):
        if trigger and scale > 0.0:
          # Get norm from disagreement tenor:
          disagreement_norm = np.linalg.norm(disagreement_data) + 1e-12
          # Get the severity:
          raw_severity = (self.pcgrad_th - cosine_similarity) / (self.epsilon + self.pcgrad_severity_range)
          severity = np.clip(raw_severity,0.0,1.0)
          # Get the projection:
          proj = np.dot(data,disagreement_data) * disagreement_data / (disagreement_norm * disagreement_norm)
          # Realign local tensor:
          proj_scale = np.clip(scale*severity,0.0,self.pcgrad_scale_max)
          projected_data = data - proj_scale * proj
          # Renormalize the projected data, in order to preserve original normalization
          norm_corr = np.linalg.norm(data) / (self.epsilon + np.linalg.norm(projected_data))
          return projected_data * norm_corr
        
        return data
    # ****************************
    
    # Overwrite ring-allreduce:
    # ****************************
    def ring_allreduce(self, currrent_gradients, mpi_comm, num_ranks, neighbours, scale, skip_alignment):
        new_gradients = {}
        uncorr_cos_sim = np.zeros(2)
        corr_cos_sim = np.zeros(2)
        # +++++++++++++++++++++++
        for key in currrent_gradients:
            # Store shape:
            current_shape = currrent_gradients[key].shape
            # Do not bother with any alignemnt and just do regular ring-allreduce: 
            if skip_alignment:
                data = self.ring_allreduce_single_tensor(
                   currrent_gradients[key].ravel(), mpi_comm, num_ranks, neighbours
                )
                uncorr_cos_sim += data[-1]
                new_gradients[key] = data[0].reshape(current_shape)
            else:
              
              # Check alignment:
              data = self.ring_allreduce_single_tensor(
                   currrent_gradients[key].ravel(), mpi_comm, num_ranks, neighbours
               )
              disagreement = data[1]
              cosine_similarity = data[2]
              trigger = data[3]
              uncorr_cos_sim += data[4]
              # Get projected local grads:
              local_projected_grads = self.project(
                currrent_gradients[key].ravel(), disagreement, cosine_similarity, trigger, scale
              ).astype(self.np_dtype)
              # Run ring-allreduce:
              new_data = self.ring_allreduce_single_tensor(
                local_projected_grads, mpi_comm, num_ranks, neighbours
              )
              # Reshape to get original shape back:
              new_gradients[key] = new_data[0].reshape(current_shape)
              # Record similarity after updates:
              corr_cos_sim += new_data[4]
        # +++++++++++++++++++++++

        return new_gradients, uncorr_cos_sim/len(new_gradients), corr_cos_sim/len(new_gradients)
    # ****************************

    # Overwrite forward, to include new functionality:
    # ****************************
    def forward(self, model, use_outer_group_communication, gradient_scale=1.0, pcgrad_scale=1.0, skip_alignment=False):
        # First, we need to get the gradients from the model:
        gradients = self.get_model_gradients(model)
        synced_grads = None
        avg_uncorr_cos_sim = np.zeros(shape=2)
        avg_corr_cos_sim = np.zeros(shape=2)
       
        # If the model is trained as ensemble on the outer ranks, then no outer grou communication will happen:
        outer_group_communication_active = use_outer_group_communication
        if self.train_as_ensemble:
            outer_group_communication_active = False

        # Use grouping:
        if self.use_grouping == True:

            # Inner group updates, i.e. ranks on the same node:
            if outer_group_communication_active == False:

                # Check if RMA is requested:
                if self.use_rma == True:
                    try:
                        synced_grads = self.rma_ring_allreduce(
                            gradients,
                            self.inner_comm,
                            self.n_inner_ranks,
                            self.inner_rank,
                            self.inner_neighbours[0],
                            self.rma_win_inner,
                        )
                    except:
                        if self.rma_win_inner is None:
                            logging.error(
                                f">>> {self.name}: RMA window not defined! Please make sure that you run sync_model() first, before calling this function <<<"
                            )

                # "Regular" ring all reduce
                else:
                    synced_grads, avg_uncorr_cos_sim, avg_corr_cos_sim = self.ring_allreduce(
                        gradients,
                        self.inner_comm,
                        self.n_inner_ranks,
                        self.inner_neighbours,
                        pcgrad_scale,
                        skip_alignment
                    )

            # Run outer group update, i.e. accross nodes:
            else:
                if self.n_outer_ranks > 1 and self.outer_rank >= 0:
                    synced_grads, avg_uncorr_cos_sim, avg_corr_cos_sim = self.ring_allreduce(
                        gradients,
                        self.outer_comm,
                        self.n_outer_ranks,
                        self.outer_neighbours,
                        pcgrad_scale,
                        skip_alignment
                    )

        # No grouping, i.e. conventional ARAR:
        else:
            synced_grads, avg_uncorr_cos_sim, avg_corr_cos_sim = self.ring_allreduce(
                gradients, self.comm, self.n_ranks, self.neighbours, pcgrad_scale, skip_alignment
            )
           
        # Computing gradients is expensive, so if there are no updated gradients (because the current rank is NOT part of the sync)
        # we simply use the original ones
        if synced_grads is None:
            self.set_model_gradients(model, gradients, gradient_scale=gradient_scale)
            return False, avg_uncorr_cos_sim, avg_corr_cos_sim
             
        self.set_model_gradients(model, synced_grads, gradient_scale=gradient_scale)
        return True, avg_uncorr_cos_sim, avg_corr_cos_sim
    # ****************************

