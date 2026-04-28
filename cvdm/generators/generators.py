import numpy as np
import matplotlib.pyplot as plt
from .base import *

class Uniform2D(Generator):
    def __init__(self,size):
        self.size = size
        super().__init__(size,size)
    def forward(
        self,
        nspots,
        sigma=0.92,
        texp=1.0,
        N0_min=500.0,
        N0_max=1000.0,
        eta=1.0,
        gain=1.0,
        B0=None,
        nframes=1,
        offset=100.0,
        var=5.0,
        show=False,
        halo_alpha: float = 0.0,
        halo_sigma: float = 0.0,
        grf_alpha: float = 0.0,
        grf_sigma: float = 0.0,
        grf_seed: int | None = None,
    ):
        density = Uniform(self.size)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        N0 = np.random.uniform(N0_min,N0_max,nspots)
        theta[0,:] = x; theta[1,:] = y
        theta[2,:] = sigma; theta[3,:] = N0
        adu,spikes = self.sample_frames(
            theta,
            nframes,
            texp,
            eta,
            B0,
            gain,
            offset,
            var,
            show=show,
            halo_alpha=halo_alpha,
            halo_sigma=halo_sigma,
            grf_alpha=grf_alpha,
            grf_sigma=grf_sigma,
            grf_seed=grf_seed,
        )
        return adu,spikes,theta


class Nanoruler2D(Generator):
    def __init__(self, size):
        self.size = size
        super().__init__(size, size)

    def forward(
        self,
        nspots,
        sigma=0.92,
        texp=1.0,
        N0_min=500.0,
        N0_max=1000.0,
        eta=1.0,
        gain=1.0,
        B0=None,
        nframes=1,
        offset=100.0,
        var=5.0,
        show=False,
        spacing_px=4.0,
        spacing_nm=None,
        pixel_size_nm=None,
        edgew=5.0,
        position_sigma=0.0,
        pattern: str = "uniform",
        parent_rate: float | None = None,
        parent_count: int | None = None,
        children_sigma: float = 1.0,
        children_min: int = 0,
        children_pmf: list[float] | None = None,
        burst_prob: float | None = None,
        halo_alpha: float = 0.0,
        halo_sigma: float = 0.0,
        grf_alpha: float = 0.0,
        grf_sigma: float = 0.0,
        grf_seed: int | None = None,
    ):
        if spacing_nm is not None and pixel_size_nm is not None:
            spacing_px = spacing_nm / pixel_size_nm
        half_spacing = spacing_px / 2.0
        margin = edgew + half_spacing

        if pattern == "thomas":
            if parent_count is None:
                rate = 0.0 if parent_rate is None else float(parent_rate)
                parent_count = np.random.poisson(rate)
            if parent_count < 1:
                parent_count = 1
            parents_x = np.random.uniform(margin, self.size - margin, parent_count)
            parents_y = np.random.uniform(margin, self.size - margin, parent_count)
            if children_pmf is not None:
                pmf = np.array(children_pmf, dtype=float)
                if pmf.ndim != 1 or pmf.size == 0:
                    raise ValueError("children_pmf must be a non-empty 1D list")
                pmf_sum = float(pmf.sum())
                if pmf_sum <= 0:
                    raise ValueError("children_pmf must sum to a positive value")
                pmf = pmf / pmf_sum
                counts = np.arange(1, pmf.size + 1)
                if burst_prob is None:
                    child_counts = np.random.choice(counts, size=parent_count, p=pmf)
                else:
                    bprob = float(burst_prob)
                    bprob = min(max(bprob, 0.0), 1.0)
                    bursts = np.random.rand(parent_count) < bprob
                    extra = np.random.choice(counts, size=parent_count, p=pmf)
                    child_counts = 1 + bursts.astype(int) * extra
            else:
                raise ValueError("children_pmf must be set for pattern='thomas'")
            if children_min > 0:
                child_counts = np.maximum(child_counts, int(children_min))
            total_children = int(child_counts.sum())
            if total_children < 1:
                total_children = 1
                child_counts = np.zeros(parent_count, dtype=int)
                child_counts[0] = 1
            child_x = np.repeat(parents_x, child_counts)
            child_y = np.repeat(parents_y, child_counts)
            child_x = child_x + np.random.normal(0.0, children_sigma, size=child_x.shape)
            child_y = child_y + np.random.normal(0.0, children_sigma, size=child_y.shape)
            child_x = np.clip(child_x, margin, self.size - margin)
            child_y = np.clip(child_y, margin, self.size - margin)
            x = child_x
            y = child_y
            nspots = len(x)
        else:
            x = np.random.uniform(margin, self.size - margin, nspots)
            y = np.random.uniform(margin, self.size - margin, nspots)
        angles = np.random.uniform(0, 2 * np.pi, nspots)

        dx = half_spacing * np.cos(angles)
        dy = half_spacing * np.sin(angles)

        x1 = x + dx
        y1 = y + dy
        x2 = x - dx
        y2 = y - dy

        if position_sigma and position_sigma > 0:
            x1 = x1 + np.random.normal(0.0, position_sigma, size=x1.shape)
            y1 = y1 + np.random.normal(0.0, position_sigma, size=y1.shape)
            x2 = x2 + np.random.normal(0.0, position_sigma, size=x2.shape)
            y2 = y2 + np.random.normal(0.0, position_sigma, size=y2.shape)

        nlocalizations = nspots * 2
        theta = np.zeros((4, nlocalizations))
        theta[0, 0::2] = x1
        theta[1, 0::2] = y1
        theta[0, 1::2] = x2
        theta[1, 1::2] = y2

        N0 = np.random.uniform(N0_min, N0_max, nlocalizations)
        theta[2, :] = sigma
        theta[3, :] = N0

        adu, spikes = self.sample_frames(
            theta,
            nframes,
            texp,
            eta,
            B0,
            gain,
            offset,
            var,
            show=show,
            halo_alpha=halo_alpha,
            halo_sigma=halo_sigma,
            grf_alpha=grf_alpha,
            grf_sigma=grf_sigma,
            grf_seed=grf_seed,
        )
        return adu, spikes, theta


