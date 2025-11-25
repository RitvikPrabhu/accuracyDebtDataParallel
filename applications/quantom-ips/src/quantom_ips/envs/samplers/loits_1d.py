import torch
import concurrent.futures
from dataclasses import dataclass, field
from quantom_ips.utils.loits_utils import (
    empty_cache,
    calculate_norm,
    calculate_weight_tensor,
    calculate_grid_indices,
    linear_interpolation,
)
from quantom_ips.utils.registration import register_with_hydra


# Define defaults for this sampler:
@dataclass
class LOITS1DDefaults:
    average: bool = True
    a_min: float = 0.0
    a_max: float = 1.0
    n_interpolations_x: int = 10
    use_threading: bool = False
    vmap_randomness: str = "different"
    log_space: bool = False
    id: str = "LOInverseTransformSampler1D"


# Define LOITS for 1D:
class TorchLOITS1D:
    """
    Stand-alone PyTorch implementation of the Local inverse transform sampler (LOITS), i.e. no pre-processing of the theory-outputs.
    """

    # Initialize:
    # ***************************
    def __init__(
        self,
        use_threading,
        vmap_randomness,
        devices="cpu",
        dtype=torch.float32,
    ):
        self.devices = devices
        self.dtype = dtype

        self.vmap_randomness = vmap_randomness
        self.use_threading = use_threading

    # ***************************

    # Components to calculate the CDF in x and Q2 coordinates:
    # ***************************
    # Determine the density: rho = xsec / norm
    def calc_rho(self, bins, xsec):
        # First, we need the norm:
        norm = torch.vmap(lambda b, s: calculate_norm(b, s), in_dims=0)(bins, xsec)
        # and get rho:
        rho = xsec / norm[:, None]

        return rho

    # --------------------------

    # Now compute the CDF itself:
    def calc_cdf(self, bins, xsec):
        rho = self.calc_rho(bins, xsec)

        # Define an empty CDF tensor and overwrite it with the cumulative sum --> Steven Goldenberg worked on the lines below:
        cdf = torch.zeros_like(rho)
        cdf[:, 1:] = torch.cumulative_trapezoid(y=rho, x=bins)

        return cdf

    # ***************************

    # Generate events, based on the provided observables, the acceptance matrix, weight tensors and grid index tensors:
    # ***************************
    def gen_events(
        self,
        obs_bins,
        obs_xsec,
        weight_tensor,
        grid_index_tensor,
        n_max,
    ):
        # Compute the CDF:
        cdf_obs = self.calc_cdf(obs_bins, obs_xsec)

        # First generate u:
        u_obs = torch.rand(
            (grid_index_tensor.size()[0], n_max),
            device=self.devices,
            dtype=torch.float32,
        )

        # Use the cdf in obs that we just computed and combine them with the index tensor,
        # i.e. assign the proper values to the grid position:
        cdf_obs_flat = cdf_obs[grid_index_tensor[:, 0]]
        bin_obs_flat = obs_bins[grid_index_tensor[:, 0]]

        # Now utilize the linear interpolation function and compute x:
        # Also, we can now leverage that the indices are flat and run a vectorization:
        obs_gen = (
            torch.vmap(
                linear_interpolation, in_dims=0, randomness=self.vmap_randomness
            )(u_obs, cdf_obs_flat, bin_obs_flat)
            * weight_tensor
        )
        return obs_gen.flatten()[:, None]

    # ***************************

    # Define a forward pass:
    # ***************************
    # Forward pass for a single parameter sample:
    def forward_single_sample(
        self,
        x_bins,
        xsec_x,
        weight_tensor,
        grid_idx,
        max_n,
    ):
        # Generate events:
        evts = self.gen_events(x_bins, xsec_x, weight_tensor, grid_idx, max_n)

        # Assume that we have 1D predictions firstL
        cond = evts[:, 0] > 0.0
        events = evts[cond]

        # Detach everything that we do not need --> Free the GPU memory:
        del cond
        del evts

        # Free cache on the device we are using
        empty_cache(self.devices)

        return events

    # --------------------------------------------

    # Formulate the entire forward pass:
    def forward(self, theory_outputs, n_events):
        # Get the information from theory: We are expecting 4 items:
        # i) bins and crossections = 2 variables
        # ii) cross section weights and phase space acceptance = 2 variables
        x_bins, xsec_x, weights = theory_outputs[:3]

        # Compute flat grid-indices which help to avoid another for-loop over the grid itself:
        npy_grid_idx = calculate_grid_indices(weights[0].size()[0], 1)
        torch_grid_idx = torch.as_tensor(
            npy_grid_idx, device=self.devices, dtype=torch.int
        )

        # Get batch size from theory outputs:
        batch_size = weights.size()[0]
        data_list = []
        stats_list = []

        # Generate data, using one thread per prediction, where #events_generated = n_events
        if self.use_threading == True:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_size
            ) as executor:
                threads = []

                # +++++++++++++++++++++++++++++++
                for i in range(batch_size):
                    n_low = i * n_events
                    n_high = (i + 1) * n_events

                    # Run this on CPU:
                    detached_weights = weights[i].detach().cpu()

                    n = torch.abs(detached_weights * n_high).to(torch.int).reshape(
                        -1, 1
                    ) - torch.abs(detached_weights * n_low).to(torch.int).reshape(-1, 1)
                    max_n = torch.max(n)

                    weight_tensor_cpu = calculate_weight_tensor(n)
                    weight_tensor_gpu = torch.as_tensor(
                        weight_tensor_cpu, device=self.devices, dtype=torch.float32
                    )

                    threads.append(
                        executor.submit(
                            self.forward_single_sample,
                            x_bins[i],
                            xsec_x[i],
                            weight_tensor_gpu,
                            torch_grid_idx,
                            max_n,
                        )
                    )
                # +++++++++++++++++++++++++++++++

                # +++++++++++++++++++++++++++++++
                for thread in concurrent.futures.as_completed(threads):
                    evts = thread.result()
                    data_list.append(evts)
                    stats_list.append(evts.size()[0])
                # +++++++++++++++++++++++++++++++

        else:
            # Or run everything sequentially and generate n_events for each prediction, where #events_generated = batch_size * n_events
            # +++++++++++++++++++++++++++++++
            for i in range(batch_size):
                # Run this on CPU:
                detached_weights = weights[i].detach().cpu()
                n = torch.abs(detached_weights * n_events).to(torch.int).reshape(-1, 1)
                max_n = torch.max(n)

                weight_tensor_cpu = calculate_weight_tensor(n)
                weight_tensor_gpu = torch.as_tensor(
                    weight_tensor_cpu, device=self.devices, dtype=torch.float32
                )

                current_evs = self.forward_single_sample(
                    x_bins[i],
                    xsec_x[i],
                    weight_tensor_gpu,
                    torch_grid_idx,
                    max_n,
                )

                data_list.append(current_evs)
                stats_list.append(current_evs.size()[0])
            # +++++++++++++++++++++++++++++++

        # Determine smallest number of events collected:
        n_evts_min = min(stats_list)
        # Make sure that all batches have the same number of events:
        data_list = [d[:n_evts_min] for d in data_list]
        return torch.stack(data_list, dim=0)

    # ***************************


# Full LOITS implementation
@register_with_hydra(
    group="environment/sampler", defaults=LOITS1DDefaults, name="LOITS_1D"
)
class LOInverseTransformSampler1D:
    """
    Base implementation of 1D LOITS, including a pre-processing of the theory outputs:
    - Inputs: density, grid_axes
    - Outputs: samples
    """

    # Initialize:
    # *************************
    def __init__(self, config, devices="cpu", dtype=torch.float32):
        # Info for torch:
        self.devices = devices
        self.dtype = dtype

        # Important settings:
        self.average = config.average
        self.a_min = config.a_min
        self.a_max = config.a_max
        self.n_interpolations_x = config.n_interpolations_x
        self.log_space = config.log_space

        # Get LOITS:
        self.loits = TorchLOITS1D(
            config.use_threading, config.vmap_randomness, devices, dtype
        )

        # Compute a few quantities that are frequently need:

        # Density scalers:
        self.a_min = torch.as_tensor(self.a_min, dtype=self.dtype, device=self.devices)
        self.a_max = torch.as_tensor(self.a_max, dtype=self.dtype, device=self.devices)

        # Interpolating A w.r.t x:
        self.u_x = torch.linspace(
            0, 1, self.n_interpolations_x, device=self.devices, dtype=self.dtype
        )

    # *************************

    # Get all the quantities that we are going to pass to the sampler which will then produce x/y pairs:
    # *************************
    # First, we need a linear interpolation of A in x space:
    def linear_interpolatation_1d(self, A, u):
        A_min = A[:-1]
        dA = A[1:] - A_min
        return A_min.reshape(-1, 1) + u * dA.reshape(-1, 1)

    # ----------------------------

    # Second, compute weights that are passed to LOITS 1D for sampling:
    def compute_weights(self, A, x_bins):
        # Scale A to the desired range:
        A_scaled = (self.a_max - self.a_min) * A + self.a_min

        # Interpolate w.r.t x:
        A_interpol_x = self.linear_interpolatation_1d(A_scaled, self.u_x)

        # Compute differnetial and total x-section, weights and acceptance:
        weights = torch.trapezoid(A_interpol_x, x_bins)
        sum_weights = weights.sum()
        weights = weights.squeeze() / sum_weights
        # Return everything for the sampler:
        return (weights, sum_weights, A_interpol_x)

    # *************************

    # Define the forward pass:
    # *************************
    # Single target:
    def forward_single_target(self, A, grid_axes, n_events):
        # Compute bins that match the dimensions of A:

        # x-Bins:
        xv = grid_axes[0]
        xv_min = xv[:-1]
        dxv = xv[1:] - xv_min
        x_bins = xv_min.reshape(-1, 1) + self.u_x * dxv.reshape(-1, 1)
        if self.log_space:
            x_bins = torch.exp(x_bins)

        # Make sure that the dimensions of A make sense:
        if len(A.size()) == 1:
            A = A[None, :]

        x_bins = torch.repeat_interleave(x_bins[None, :], repeats=A.size()[0], dim=0)
        # Get the weights and interpolated densities:
        weights, sum_weights, A_interpol_x = torch.vmap(
            lambda a, x: self.compute_weights(a, x), in_dims=0, randomness="same"
        )(A, x_bins)

        # Average everything if requested by the user:
        if self.average == True:
            A_interpol_x = torch.mean(A_interpol_x, 0)[None, :]
            weights = torch.mean(weights, 0)[None, :]
        else:
            x_bins = torch.repeat_interleave(x_bins, A.size()[0], dim=0)

        # Now generate samples using LOITS:
        theory_outputs = (
            x_bins,
            A_interpol_x,
            weights,
            sum_weights,
        )
        return self.loits.forward(theory_outputs, n_events)

    # ----------------------------

    # All targets:
    def forward(self, A, grid_axes, n_events, chosen_targets=None):
        n_targets = A.size()[1]
        events = []
        stats = []

        if chosen_targets is None:
          chosen_targets = [p for p in range(n_targets)]
          
        # +++++++++++++++++++++++
        for p in chosen_targets:
              samples = self.forward_single_target(A[:, p, :, :], grid_axes, n_events)
              events.append(samples)
              stats.append(samples.size()[1])
        # +++++++++++++++++++++++
        min_n_events = min(stats)

        # +++++++++++++++++++++++
        for i in range(len(events)):
            ev = events[i]
            idx = torch.randperm(ev.size()[1], device=self.devices)[:min_n_events]
            events[i] = ev[:, idx, :]
            del idx
        # +++++++++++++++++++++++

        return torch.stack(events, dim=1)

    # *************************
