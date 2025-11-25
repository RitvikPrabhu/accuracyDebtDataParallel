import numpy as np
import torch
from quantom_ips.utils.registration import register_with_hydra
from dataclasses import dataclass, field
import quantom_ips.utils.params as PAR
from quantom_ips.utils.pixelated_densities import (
    DukeAndOwensDensities,
)

from dataclasses import dataclass


@dataclass
class DukeAndOwensTheoryV1Defaults:
    id: str = "DukeAndOwensTheory"
    a_min: list = field(default_factory=lambda: [0.0,0.0])
    a_max: list = field(default_factory=lambda: [1.0,1.0])
    xsec_min: float = 0.0
    xsec_max: float = 1.0
    average: bool = True
    fixed_density_index: int = -1
    acceptance_epsilon: float = 1e-11


@register_with_hydra(
    group="environment/theory",
    defaults=DukeAndOwensTheoryV1Defaults,
    name="duke_and_owens",
)
class DukeAndOwensTheory:
    """
    Theory module that ingests a 2D array A mimicing a density and produces x-sections parameterized in 2 dimenstions (e.g. x and Q^2).
    Most of the code here was initially designed by Nobuo Sato.
    """

    # Initialize:
    # *************************
    def __init__(self, config, devices="cpu", dtype=torch.float32):
        self.devices = devices
        self.dtype = dtype

        # Basic settings in config:
        self.a_min = config.a_min
        self.a_max = config.a_max
        self.xsec_min = config.xsec_min
        self.xsec_max = config.xsec_max
        self.average = config.average
        self.acceptance_epsilon = config.acceptance_epsilon

        # Initialze tool to get true densities:
        self.density_util = DukeAndOwensDensities(devices=self.devices)

        # Get the fixed density index that should be either 0 (up quark) or 1 (down quark)
        self.fixed_density_index = config.fixed_density_index
        self.fixed_A = [None, None]

        # Theory specific constants that are needed to properly compute the x-section:
        self.Q02 = 1.0
        self.lam2 = 0.4
        self.rs = 140.0
        self.W2min = 10.0

        # Density scalers:
        self.a_min = torch.as_tensor(self.a_min, dtype=self.dtype, device=self.devices)
        self.a_max = torch.as_tensor(self.a_max, dtype=self.dtype, device=self.devices)

        # Determine the x / Q^2 axes that are required for the theory output:
        self.Q2_min = PAR.mc2
        self.Q2_max = self.rs**2 - PAR.M2
        self.x_min = self.Q2_min / self.Q2_max
        self.x_max = 1.0

    # *************************

    # Tanslate densities to x-sections:
    # *************************
    def translate_density_to_xsection(self, A, A_fix, x, Q2):
        # Fix one of the densities, if required:
        u_density, d_density = A_fix
        if u_density is None:
            u_density = A[0]*(self.a_max[0] - self.a_min[0]) + self.a_min[0]
        if d_density is None:
            d_density = A[1]*(self.a_max[1] - self.a_min[1]) + self.a_min[1]

        y = Q2 / self.Q2_max / x
        Yp = 1 + (1 - y) ** 2
        Ym = 1 - (1 - y) ** 2
        K2 = Yp + 2 * x**2 * y**2 * PAR.M2 / Q2
        KL = -(y**2)
        K3 = Ym * x
        alfa = 1 / 137
        norm = 2 * np.pi * alfa**2 / x / y / Q2 * y / Q2

        xu = x * u_density
        xd = x * d_density

        eU2 = 4.0 / 9.0
        eD2 = 1.0 / 9.0

        # x-section for proton target:
        F2_p = eU2 * xu + eD2 * xd

        FL_p = 0
        F3_p = 0
        xsec_p = norm * (K2 * F2_p + KL * FL_p + K3 * F3_p)

        # x-section for neutron target:
        F2_n = eU2 * xd + eD2 * xu
        FL_n = 0
        F3_n = 0
        xsec_n = norm * (K2 * F2_n + KL * FL_n + K3 * F3_n)

        return torch.stack([xsec_p, xsec_n], dim=0)

    # *************************

    # Define the forward pass:
    # *************************
    def forward(self, A):
        # Get the dimensions from the predictions:
        n_points_x = A.size()[2]
        n_points_Q2 = A.size()[3]

        # Handle true densities for debugging:
        true_densities = self.density_util.get_pixelated_densities(
            n_points_x, n_points_Q2
        )[0]

        if self.fixed_density_index >= 0 and self.fixed_density_index < 2:
            self.fixed_A[self.fixed_density_index] = true_densities[
                self.fixed_density_index, :, :
            ]
        if (
            self.fixed_density_index == 2
        ):  # --> A sort of debug mode, just to make sure that the outputs of this module make sense
            self.fixed_A[0] = true_densities[0, :, :]
            self.fixed_A[1] = true_densities[1, :, :]

        # Pre-calculations for the module:
        self.x_axis = torch.linspace(
            np.log(self.x_min),
            np.log(self.x_max),
            n_points_x,
            device=self.devices,
            dtype=self.dtype,
        )

        self.Q2_axis = torch.linspace(
            np.log(self.Q2_min),
            np.log(self.Q2_max),
            n_points_Q2,
            device=self.devices,
            dtype=self.dtype,
        )

        # Create a grid from the axis --> Needed for x-section computation:
        self.xv_grid, self.Q2v_grid = torch.meshgrid(
            self.x_axis, self.Q2_axis, indexing="ij"
        )
        self.xv_grid = torch.exp(self.xv_grid)
        self.Q2v_grid = torch.exp(self.Q2v_grid)

        # Get the acceptance, i.e. which bins are physically relevant and which are not
        # 1st condition:
        x_c = self.xv_grid
        Q2_c = self.Q2v_grid
        cond1 = self.Q2_max * x_c >= Q2_c
        # 2nd condition:
        W2_c2 = PAR.M2 + Q2_c / x_c - Q2_c
        cond2 = W2_c2 >= self.W2min
        # Now get the acceptance:
        acceptance = torch.where((cond1 * cond2), 1.0, self.acceptance_epsilon)

        # Scale the differential x-section (dQ^2dx)
        xsec_scaling = (self.xv_grid * self.Q2v_grid) * acceptance

        # Provide loss as one output argument:
        zero_loss = torch.zeros(
            size=(1,), dtype=self.dtype, device=self.devices, requires_grad=True
        )

        # Get x-sections:
        xsec = torch.vmap(
            lambda a: self.translate_density_to_xsection(
                a, self.fixed_A, self.xv_grid, self.Q2v_grid
            ),
            in_dims=0
        )(A)

        # Gather the axes:
        axes = [self.x_axis, self.Q2_axis]

        # Determine xsection after acceptance correction and clamping:
        corr_xsec = torch.clamp(xsec*xsec_scaling,min=self.xsec_min,max=self.xsec_max)

        # Make it all available:
        if self.average:
            return corr_xsec.mean(dim=0)[None, :], axes, zero_loss
        return corr_xsec, axes, zero_loss

    # *************************
