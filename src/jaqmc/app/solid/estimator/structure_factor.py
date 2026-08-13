r"""Structure factor estimators for periodic solid state systems.

Computes the charge ($S_\rho(q)$) and spin ($S_\sigma(q)$) structure factors
by evaluating the Fourier components of the electron density $n_q$.
"""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.app.solid.data import SolidData
from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData
from jaqmc.estimator.base import Estimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class StructureFactor(Estimator):
    r"""Spin and Charge Structure Factor estimator."""

    # Needs to know the number of spins to split the electrons array
    nspins: tuple[int, int] = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    # 1D Grid definition (matches momentum_distr.py)
    N: int = runtime_dep(default=10)  # Number of atoms/sites
    a: float = runtime_dep(default=2.0)  # Interatomic distance
    name: str = "StructureFactor"

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.n_up = self.nspins[0]
        self.n_down = self.nspins[1]

    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData[SolidData],
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        del prev_walker_stats, state, rngs  # Not used

        def single_walker_math(walker_data):
            electrons = walker_data[self.data_field]  # shape: (nelec, 3)

            # Split into up and down spins
            r_up = electrons[: self.n_up]
            r_down = electrons[self.n_up :]

            # Extract x-coordinates for 1D structure factor
            rx_up = r_up[:, 0]
            rx_down = r_down[:, 0]

            # q = 2 * pi * n / (N * a)
            n_vals = jnp.arange(-self.N // 2, self.N // 2 + 1)
            q_vals = 2.0 * jnp.pi * n_vals / (self.N * self.a)

            # Compute sum_{i} e^{i q r_i} for up and down spins
            # Outer product gives shape (n_q, n_electrons_up/down)
            exp_iqr_up = jnp.exp(1j * jnp.outer(q_vals, rx_up))
            exp_iqr_down = jnp.exp(1j * jnp.outer(q_vals, rx_down))

            n_q_up = jnp.sum(exp_iqr_up, axis=1)  # shape (n_q,)
            n_q_down = jnp.sum(exp_iqr_down, axis=1)  # shape (n_q,)

            n_q_total = n_q_up + n_q_down
            n_q_diff = n_q_up - n_q_down

            # Return instantaneous values and squared magnitudes.
            # The connected part (<AB> - <A><B>) is computed in finalize_stats
            return {
                "n_q_total": n_q_total,
                "n_q_diff": n_q_diff,
                "n_q_total_sq": jnp.abs(n_q_total) ** 2,
                "n_q_diff_sq": jnp.abs(n_q_diff) ** 2,
            }

        walker_stats = jax.vmap(single_walker_math, in_axes=(batched_data.vmap_axis,))(
            batched_data.data
        )

        return walker_stats, None

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(walker_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        """Compute the connected part of the structure factors."""
        # 1. Average over MCMC steps (axis 0)
        mean_stats = {k: jnp.nanmean(v, axis=0) for k, v in batched_stats.items()}

        # S_rho(q) = < |n_q_up + n_q_down|^2 > - |<n_q_up + n_q_down>|^2
        S_rho = mean_stats["n_q_total_sq"] - jnp.abs(mean_stats["n_q_total"]) ** 2

        # S_sigma(q) = < |n_q_up - n_q_down|^2 > - |<n_q_up - n_q_down>|^2
        S_sigma = mean_stats["n_q_diff_sq"] - jnp.abs(mean_stats["n_q_diff"]) ** 2

        return {
            "S_rho": S_rho,
            "S_sigma": S_sigma,
            "n_q_total": mean_stats["n_q_total"],
            "n_q_diff": mean_stats["n_q_diff"],
        }
