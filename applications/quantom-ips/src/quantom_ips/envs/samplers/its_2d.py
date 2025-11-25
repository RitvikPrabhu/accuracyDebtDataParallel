import torch
import logging
from quantom_ips.utils.registration import register_with_hydra
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ITS2DDefaults:
    id: str = "InverseTransformSampler2D"
    interp_pts: int = 10


@register_with_hydra(group="environment/sampler", defaults=ITS2DDefaults, name="ITS_2D")
class InverseTransformSampler2D:
    def __init__(self, config, devices, dtype=torch.float32):

        self.interp_pts = config.interp_pts

        self.devices = devices
        self.dtype = dtype

        self.conv2d_kernel = (
            torch.ones(size=(1, 1, 2, 2), device=self.devices, dtype=self.dtype) / 4
        )

        self.eps = torch.finfo(self.dtype).eps

    def normalize_A(self, A, xpts, ypts):
        assert A.dim() == 2
        A = torch.nn.functional.interpolate(
            A[None, None, :, :],
            size=(xpts, ypts),
            mode="bilinear",
            align_corners=True,
        ).squeeze(dim=(0, 1))

        area = torch.nn.functional.conv2d(
            A[None, None, :, :], self.conv2d_kernel
        ).squeeze(dim=(0, 1))
        total = area.sum()

        # We should probably just return completely random numbers if total == 0.0,
        # but for now we will just assert that it's > 0.0
        assert total > 0

        pdf_points = A / total

        return pdf_points

    def interpolate_grid(self, pts):
        n_pts = pts.shape[0]
        n_pts_out = (n_pts - 1) * (self.interp_pts + 1) + 1

        pts = torch.nn.functional.interpolate(
            pts[None, None, :], size=(n_pts_out), mode="linear", align_corners=True
        ).squeeze()

        return pts

    def particle_forward(self, A, grid_list, n):
        output = torch.stack(
            [self.sample(A_particle, grid_list, n) for A_particle in A]
        )
        return output

    def forward(self, A, grid_list, n):
        output = torch.stack(
            [self.particle_forward(A_batch, grid_list, n) for A_batch in A]
        )
        return output

    def sample(self, A, grid_list, n):
        # Get grid information
        assert len(grid_list) == 2
        xpts, ypts = [self.interpolate_grid(grid) for grid in grid_list]
        xpts_out = xpts.shape[0]
        ypts_out = ypts.shape[0]

        pdf_points = self.normalize_A(A, xpts_out, ypts_out)

        all_probs = torch.nn.functional.conv2d(
            pdf_points[None, None, :, :], self.conv2d_kernel
        ).squeeze(dim=(0, 1))
        xprobs = all_probs.sum(dim=1).cumsum(dim=0)

        # if total > 0, then xprobs[-1] should always be very close to 1.0...
        # xprobs[-1] = torch.where(xprobs[-1] == 0.0, 1.0, xprobs[-1])
        xprobs[-1] += self.eps

        u_x = torch.rand(n, device=self.devices, dtype=self.dtype)
        xinds = (u_x[:, None] > xprobs[None, :-1]).sum(dim=1)

        yprobs = all_probs.cumsum(dim=1)
        # These cumulative sums may be 0.0, so we add eps to
        # the final entry before normalizing
        yprobs[:, -1] += self.eps

        # By adding eps to the last yprob values, we shouldn't need to check for 0.0's
        # yprobs[:, -1:] = torch.where(yprobs[:, -1:] == 0.0, 1.0, yprobs[:, -1:])
        yprobs = yprobs / yprobs[:, -1:]

        u_y = torch.rand(n, device=self.devices, dtype=self.dtype)
        yinds = (u_y[:, None] > yprobs[xinds, :-1]).sum(dim=1)

        xprobs = torch.concatenate(
            [torch.zeros(1, device=self.devices, dtype=self.dtype), xprobs]
        )
        yprobs = torch.concatenate(
            [
                torch.zeros(xpts_out - 1, 1, device=self.devices, dtype=self.dtype),
                yprobs,
            ],
            dim=1,
        )

        xsamples = (u_x - xprobs[xinds]) / (
            xprobs[xinds + 1] - xprobs[xinds] + self.eps
        )
        ysamples = (u_y - yprobs[xinds, yinds]) / (
            yprobs[xinds, yinds + 1] - yprobs[xinds, yinds] + self.eps
        )

        xval = xpts[xinds] + xsamples * (xpts[xinds + 1] - xpts[xinds])
        yval = ypts[yinds] + ysamples * (ypts[yinds + 1] - ypts[yinds])

        return torch.stack([xval, yval], dim=1)
