from mpi4py import MPI
import numpy as np
import torch
import logging
from dataclasses import dataclass
from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)

@dataclass
class TorchARARDefaults:
    id: str = "TorchARAR"
    master_rank: int = 0
    force_rma_rank_synchronization: bool = False
    group_size: int = 4
    gradient_sync_mode: str = "arar"
    use_weight_grad_only: bool = False
    train_as_ensemble: bool = False
    
@register_with_hydra(
    group="trainer/gradient_transport", defaults=TorchARARDefaults, name="torch_arar"
)
class TorchARAR:
    """
    Torch Asynchronous Ring All Reduce (ARAR)
    """

    # Initialize:
    # ****************************
    def __init__(
        self, config, mpi_comm=None, default_group_size=4, dtype=torch.float32
    ):
        # Module name for logging:
        self.name = config.id

        # Collect basic information:
        self.master_rank = config.master_rank
        self.force_rma_rank_synchronization = config.force_rma_rank_synchronization
        group_size = config.group_size
        gradient_sync_mode = config.gradient_sync_mode
        self.use_weight_grad_only = config.use_weight_grad_only
        if self.use_weight_grad_only:
            logger.warning("Using weight gradients only, this means that bias gradients are not shared. Please watch out for model drifts!")
        self.train_as_ensemble = config.train_as_ensemble
        self.dtype = dtype
        # Get the dtype for numpy:
        self.np_dtype = str(self.dtype).split(".")[1]
        
        # Initialize important properties for grouping, gradient sync and RMA:
        self.use_grouping = False
        self.use_rma = False
        self.gradient_sync_mode = None
        self.group_size = -1
        self.rma_win_inner = None

        # Get info from the mpi communicator:

        # Overall communication:

        # Initialize own comm, if user did not provide one:
        # This allows a bit more flexibility, in case: (a) someone is already working within an mpi-environment, or: (b) if this is the only software that require mpi and no other rank communication is required
        if mpi_comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = mpi_comm

        self.rank = self.comm.Get_rank()
        self.n_ranks = self.comm.Get_size()
        self.neighbours = None

        # Inner group communication:
        self.inner_comm = None
        self.inner_rank = -1
        self.n_inner_ranks = -1
        self.inner_neighbours = None

        # Outer group communication:
        self.outer_comm = None
        self.outer_rank = -1
        self.n_outer_ranks = -1
        self.outer_neighbours = None

        # Assign device to rank:
        self.assign_device_to_rank()
        # Bundle ranks into groups:
        self.group_ranks(gradient_sync_mode, group_size, default_group_size)

    # ****************************

    # Assign device (GPU/CPU) to rank:
    # ****************************
    def assign_device_to_rank(self):
        # CPU:
        dev = "cpu"
        self.device_is_cpu = True
        self.device_is_cuda = False
        self.device_is_mps = False

        self.torch_device_id = self.rank

        # CUDA:
        if torch.cuda.is_available():
            self.torch_device = "cuda"

            self.device_is_cpu = False
            self.device_is_cuda = True
            self.device_is_mps = False

            n_cudas = torch.cuda.device_count()
            accept_rank = False
            while accept_rank == False:

                if self.torch_device_id < n_cudas:
                    accept_rank = True
                else:
                    self.torch_device_id -= n_cudas

            dev = "cuda:" + str(self.torch_device_id)
        # MPS:
        elif torch.backends.mps.is_available():
            dev = "mps"
            self.device_is_cpu = False
            self.device_is_cuda = False
            self.device_is_mps = True

        self.torch_device = torch.device(dev)
        self.devices = dev
        logger.info(
            f"Rank {self.rank} uses torch device: {self.torch_device} and is on processor: {MPI.Get_processor_name()}"
        )

        if self.use_grouping == True:
            logger.info(
                f"Inner Rank: {self.inner_rank} and Outer Rank {self.outer_rank} use torch device: {self.torch_device} and is on processor: {MPI.Get_processor_name()}"
            )

    # ****************************

    # Handle the grouping:
    # ****************************
    # Get the left and right "neighbour" of the current rank:
    def get_neighbours(self, current_rank, num_ranks):
        left = ((current_rank - 1) + num_ranks) % num_ranks
        right = (current_rank + 1) % num_ranks

        return left, right

    # ---------------------------

    # Group ranks:
    def group_ranks(self, gradient_sync_mode, group_size, default_group_size):
        # Determine the gradient synchronization mode and if grouping is active or not:
        # Modes with grouping:
        if (
            gradient_sync_mode.lower() == "arar"
            or gradient_sync_mode.lower() == "rma_arar"
        ):
            self.use_grouping = True
            self.gradient_sync_mode = gradient_sync_mode.lower()
            if self.gradient_sync_mode == "rma_arar":
                self.use_rma = True

            # Make sure that group size is properly set:
            if group_size < 1:
                logger.warning(
                    f"The group size you provided {group_size} is < 1. Going to set it to default: {default_group_size}"
                )
                self.group_size = default_group_size
            else:
                logger.warning(
                    f"Running with group size: {group_size}"
                )
                self.group_size = group_size

            # Inner:
            self.inner_comm = self.comm.Split(color=self.rank / self.group_size)
            self.inner_rank = self.inner_comm.Get_rank()
            self.n_inner_ranks = self.inner_comm.Get_size()
            # Get inner group neighbours for (RMA) ring_allreduce:
            self.inner_neighbours = self.get_neighbours(
                self.inner_rank, self.n_inner_ranks
            )

            # Outer:
            outer_rank_list = [0]  # --> We just collect rank 0 from all inner groups:
            f = 1
            # ++++++++++++++++
            for r in range(self.n_ranks):
                if r == f * self.group_size:
                    outer_rank_list.append(r)
                    f += 1
            # ++++++++++++++++

            outer_group = self.comm.group.Incl(outer_rank_list)
            self.outer_comm = self.comm.Create_group(outer_group)
            self.outer_rank = outer_group.Get_rank()
            self.n_outer_ranks = outer_group.Get_size()
            # Get outer group neighbours for ring_allreduce:
            self.outer_neighbours = self.get_neighbours(
                self.outer_rank, self.n_outer_ranks
            )

        # Mode without grouping:
        elif gradient_sync_mode.lower() == "conv_arar":
            self.use_grouping = False
            # Get overall neighbours for ring_allreduce:
            self.neighbours = self.get_neighbours(self.rank, self.n_ranks)

        else:
            logger.warning(
                f"The gradient synchronization mode you provided {gradient_sync_mode} is not implemented. Please check your settings."
            )

    # ****************************

    # Synchronize model and its optimizer:
    # ****************************
    # Generic function to pass state dict around:
    def sync_state_dict(self, torch_object, current_rank, current_comm, master):
        if current_rank == master:
            ref_states = torch_object.state_dict()
        else:
            ref_states = None

        ref_states = current_comm.bcast(ref_states, root=master)
        torch_object.load_state_dict(ref_states)

    # ------------------------------------------

    # Synchronize the model and its optimizer:
    def sync_model(self, model, optimizer):
        # Distributed the optimizer / model states accross all ranks:

        # Training as ensemble on the outer group, i.e. ensemble accross nodes:
        # This means we need to synchronie the model/optimizer accross members of the inner group only.
        if self.train_as_ensemble:
            self.sync_state_dict(
                model, self.inner_rank, self.inner_comm, self.master_rank
            )
            self.sync_state_dict(
                optimizer, self.inner_rank, self.inner_comm, self.master_rank
            )
        else:  # Training with gradient transfer accross all ranks:
            # We have to synchronize model/optimzier accross ALL ranks:
            self.sync_state_dict(model, self.rank, self.comm, self.master_rank)
            self.sync_state_dict(optimizer, self.rank, self.comm, self.master_rank)

        # Now that we have the weights, we can determine the number of trainable model parameters, that we need for the RMA window:
        if self.use_rma == True and self.use_grouping == True:
            n_total_params = 0
            # ++++++++++++++++++++++++++++++++++++++
            for name, params in model.named_parameters():
                if "weight" in name:
                    n_total_params += np.prod(params.size())

                # Add bias to the gradient transfer
                if self.use_weight_grad_only == False and "bias" in name:
                    n_total_params += np.prod(params.size())
            # ++++++++++++++++++++++++++++++++++++++

            self.rma_win_inner = MPI.Win.Allocate(
                n_total_params * MPI.DOUBLE.Get_size(), comm=self.inner_comm
            )

    # ****************************

    # Get and set gradients of model:
    # ****************************
    # Get the gradients:
    def get_model_gradients(self, model):
        model_grad_dict = {}
        # +++++++++++++++++++++++++++++++
        for i, (name, params) in enumerate(model.named_parameters()):
            if self.use_weight_grad_only == True:  # Consider weights only:
                if "weight" in name:
                    model_grad_dict[name] = (
                        params.grad.detach().cpu().numpy().astype(self.np_dtype)
                    )
            else:
                if "weight" in name or "bias" in name:
                    model_grad_dict[name] = (
                        params.grad.detach().cpu().numpy().astype(self.np_dtype)
                    )
        # +++++++++++++++++++++++++++++++

        return model_grad_dict

    # ------------------------

    # Set the gradients:
    def set_model_gradients(self, model, new_gradients, gradient_scale):
        gradients_set = False  # --> Make sure that we actually have new gradients...

        if bool(new_gradients) == True:
            # +++++++++++++++++++++++++++++++
            for name, params in model.named_parameters():
                if self.use_weight_grad_only == True:  # Consider weights only:
                    if "weight" in name:
                        params.grad = torch.as_tensor(
                            new_gradients[name] * gradient_scale,
                            dtype=self.dtype,
                            device=self.torch_device,
                        )
                        gradients_set = True
                else:
                    if "weight" in name or "bias" in name:
                        params.grad = torch.as_tensor(
                            new_gradients[name] * gradient_scale,
                            dtype=self.dtype,
                            device=self.torch_device,
                        )
                        gradients_set = True
            # +++++++++++++++++++++++++++++++

        return gradients_set

    # ****************************
    
    # Define 'conventional' ring-allreduce: Taken from: https://pytorch.org/tutorials/intermediate/dist_tuto.html#
    # The conventional ring-allreduce is acting on the outer communicationr, i.e. accross all nodes
    # ****************************
    # Define ring_allreduce for a tensor:
    def ring_allreduce_single_tensor(self, data, mpi_comm, num_ranks, neighbours, op_is_sum=False):
        send_data = np.copy(data)
        recv_data = np.empty_like(data)
        accum_data = np.copy(data)
        
        # +++++++++++++++++++++++
        for _ in range(num_ranks - 1):
            # Receive data from "left" neighbour:
            recv_req = mpi_comm.Irecv(recv_data, neighbours[0])
            # Send data to "right" neighbour:
            send_req = mpi_comm.Isend(send_data, neighbours[1])

            # Make sure we locally receive the incoming data and accumulate it:
            recv_req.wait()
            accum_data[:] += recv_data

            # Once data is sent away, we update the send data to what we received:
            send_req.wait()
            send_data[:] = recv_data
        # +++++++++++++++++++++++
        
        if op_is_sum:
            return accum_data
        
        return accum_data / num_ranks
    # -----------------------------
    
    # Apply it to the entire model:
    def ring_allreduce(self, currrent_gradients, mpi_comm, num_ranks, neighbours):
        new_gradients = {}
        # +++++++++++++++++++++++
        for key in currrent_gradients:
            new_gradients[key] = self.ring_allreduce_single_tensor(
                currrent_gradients[key], mpi_comm, num_ranks, neighbours
            )
        # +++++++++++++++++++++++

        return new_gradients
    # ****************************
    
    # RMA ring-allreduce:
    # ****************************
    def rma_ring_allreduce(
        self,
        current_gradients,
        mpi_comm,
        n_comm_cycles,
        current_rank,
        prev_rank,
        rma_win_dict,
    ):
        # Preparations for gradient transfer:
        send_grad_data = {}
        accum = {}
        # +++++++++++++++++++++++++++++++++
        for key in current_gradients:
            send_grad_data[key] = (
                current_gradients[key] / n_comm_cycles
            )  # --> Normalize the gradients here, so that we just need to add them later on
            accum[key] = current_gradients[key] / n_comm_cycles
        # +++++++++++++++++++++++++++++++++

        # Now transfer gradients:
        # +++++++++++++++++++++++++++++++++
        for _ in range(n_comm_cycles - 1):
            # Dump gradients into memory:
            rma_win_dict.Lock(rank=current_rank)
            # ++++++++++++++++++++++
            for key in send_grad_data:
                rma_win_dict.Put(send_grad_data[key], target_rank=current_rank)
            # ++++++++++++++++++++++
            rma_win_dict.Unlock(rank=current_rank)

            # Wait for everyone to finish:
            if self.force_rank_synchronization == True:
                mpi_comm.Barrier()

            # Receive gradients and accumulate them:
            rma_win_dict.Lock(rank=prev_rank)
            # ++++++++++++++++++++++
            for key in current_gradients:
                recv_grad = np.zeros(current_gradients[key].shape, dtype=np.float32)
                rma_win_dict.Get(recv_grad, target_rank=prev_rank)

                if np.isfinite(np.sum(recv_grad)) == True:
                    accum[key] += recv_grad
                    send_grad_data[key] = recv_grad
            # ++++++++++++++++++++++
            rma_win_dict.Unlock(rank=prev_rank)
        # +++++++++++++++++++++++++++++++++

        return accum

    # ****************************
    
    # Synchronize / transport gradients:
    # *************************
    def forward(self, model, use_outer_group_communication, gradient_scale=1.0):
        # First, we need to get the gradients from the model:
        gradients = self.get_model_gradients(model)
        synced_grads = None
       
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
                    synced_grads = self.ring_allreduce(
                        gradients,
                        self.inner_comm,
                        self.n_inner_ranks,
                        self.inner_neighbours
                    )

            # Run outer group update, i.e. accross nodes:
            else:
                if self.n_outer_ranks > 1 and self.outer_rank >= 0:
                    synced_grads = self.ring_allreduce(
                        gradients,
                        self.outer_comm,
                        self.n_outer_ranks,
                        self.outer_neighbours
                    )

        # No grouping, i.e. conventional ARAR:
        else:
            synced_grads = self.ring_allreduce(
                gradients, self.comm, self.n_ranks, self.neighbours
            )
           
        # Computing gradients is expensive, so if there are no updated gradients (because the current rank is NOT part of the sync)
        # we simply use the original ones
        if synced_grads is None:
            self.set_model_gradients(model, gradients, gradient_scale=gradient_scale)
            return False
             
        self.set_model_gradients(model, synced_grads, gradient_scale=gradient_scale)
        return True

    # *************************

    # Clear RMA window, when we are done with everything,
    # and reset the pc trigger:
    # *************************
    def clear(self):
        if self.rma_win_inner is not None:
            self.rma_win_inner.Free()
    # *************************
        
