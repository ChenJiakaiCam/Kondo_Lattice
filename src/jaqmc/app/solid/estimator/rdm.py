r"""Reduced density matrix (RDM) estimator for periodic solid state systems.

Computes the 1-RDM and 2-RDM using meta-Lowdin localized orbitals from PySCF
and MCMC importance sampling of the auxiliary coordinate $r'$.
"""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp
from pyscf import lo

from jaqmc.array_types import Params, PRNGKey
from jaqmc.app.solid.data import SolidData
from jaqmc.data import BatchedData
from jaqmc.estimator.base import Estimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import WavefunctionEvaluate
from jaqmc.utils.atomic.scf import PeriodicSCF
from jaqmc.utils.atomic.gto import PBCAtomicOrbitalEvaluator
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.geometry.pbc import make_pbc_gaussian_proposal, wrap_positions


@configurable_dataclass
class OneAndTwoRDM(Estimator):
    r"""One- and Two-body reduced density matrix.

    Computes the 1-RDM and 2-RDM using meta-Lowdin localized orbitals.
    Uses an MCMC importance sampling over $f(r') = \sum_i |\varphi_i(r')|^2$.
    Returns spin-separated RDMs (alpha/beta blocks).
    """

    phase_logpsi: WavefunctionEvaluate = runtime_dep()
    scf: PeriodicSCF = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    # one-body sampling MCMC settings
    # These can be set in the config yaml file, e.g., sc_h_chain_rdm.yml
    n_sweeps: int = 1
    aux_steps: int = 10

    # Batching parameters needed for the auxiliary pool
    batch_size: int = 2048
    ratio_naux_nbatch: float = 2.0

    def init(self, data: SolidData, rngs: PRNGKey) -> dict[str, Any]:
        """
        Initialize the auxiliary electron coordinates for 1-RDM and 2-RDM sampling.
        """
        self.n_up = self.scf._cell.nelec[0]
        self.n_down = self.scf._cell.nelec[1]

        cell = self.scf._cell
        self._lattice_vectors = jnp.array(cell.lattice_vectors())
        self._mo_coeff = jnp.array(lo.orth_ao(cell, 'meta-lowdin'))
        self._ao_evaluator = PBCAtomicOrbitalEvaluator.from_pyscf(cell)
        self._kpts = jnp.asarray(self.scf.get_orbital_kpoints())

        key_init, key_burn = jax.random.split(rngs)

        r_prime_pool = self._init_electrons(key=key_init, num_samples=int(self.batch_size * self.ratio_naux_nbatch))

        self._aux_sampler = MCMCSampler(
            steps=self.aux_steps,
            initial_width=0.5,
            adapt_frequency=10,
            sampling_proposal=make_pbc_gaussian_proposal(self._lattice_vectors)
        )

        sampler_state = self._aux_sampler.init(r_prime_pool, rngs)

        def burn_in_step(i, val):
            rp, state, key = val
            key, subkey = jax.random.split(key)
            rp, _, state = self._aux_sampler.step(
                batch_log_prob=self._log_fsum,
                data=rp,
                state=state,
                rngs=subkey
            )
            return rp, state, key

        num_burn_in_steps = 400
        r_prime_pool, sampler_state, _ = jax.lax.fori_loop(
            0,
            num_burn_in_steps,
            burn_in_step,
            (r_prime_pool, sampler_state, key_burn)
        )

        return {"r_prime_pool": r_prime_pool, "sampler_state": sampler_state}

    def _init_electrons(self, key: PRNGKey, num_samples: int) -> jnp.ndarray:
        """Initialize auxiliary electron coordinates for a 1D periodic chain."""
        key_x, key_yz = jax.random.split(key)

        Lx = self._lattice_vectors[0, 0]
        x_init = jax.random.uniform(key_x, (num_samples, 1), minval=0.0, maxval=Lx)
        yz_init = jax.random.normal(key_yz, (num_samples, 2)) * 1.0
        r_prime_pool = jnp.concatenate([x_init, yz_init], axis=1)
        r_prime_pool = wrap_positions(r_prime_pool, self._lattice_vectors)

        return r_prime_pool

    def _evaluate_mo(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Evaluate localized MOs at positions."""
        aos = self._ao_evaluator(positions, self._kpts)
        return jnp.dot(aos[0], self._mo_coeff)

    def _fsum(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Evaluate f(r) = sum_i |phi_i(r)|^2."""
        mo = self._evaluate_mo(positions)
        return jnp.sum(jnp.abs(mo)**2, axis=-1)

    def _log_fsum(self, data: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log(f(r')) for MCMC sampling."""
        return jnp.log(self._fsum(data))

    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData[SolidData],
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        del prev_walker_stats
        r_prime_pool = state["r_prime_pool"]
        sampler_state = state["sampler_state"]

        r_prime_pool, aux_stats, sampler_state = self._aux_sampler.step(
            batch_log_prob=self._log_fsum,
            data=r_prime_pool,
            state=sampler_state,
            rngs=rngs
        )

        rngs, subkey_1rdm, subkey_2rdm = jax.random.split(rngs, 3)
        sweep_keys_1rdm = jax.random.split(subkey_1rdm, self.n_sweeps)
        sweep_keys_2rdm = jax.random.split(subkey_2rdm, self.n_sweeps)

        # 1-RDM Sampling
        def draw_single_sweep_1rdm(key):
            return jax.random.choice(
                key,
                r_prime_pool,
                shape=(batched_data.batch_size,),
                replace=False,
                axis=0
            )

        r_prime_sweeps_1rdm = jax.vmap(draw_single_sweep_1rdm)(sweep_keys_1rdm)
        r_prime_per_walker_1rdm = jnp.swapaxes(r_prime_sweeps_1rdm, 0, 1)  # Shape: (batch_size, n_sweeps, 3)

        # 2-RDM Sampling
        def draw_single_sweep_2rdm(key):
            rp = jax.random.shuffle(key, r_prime_pool)
            rp = rp[:2 * batched_data.batch_size, :]
            return rp.reshape((2, batched_data.batch_size, 3))

        r_prime_sweeps_2rdm = jax.vmap(draw_single_sweep_2rdm)(sweep_keys_2rdm)
        r_prime_per_walker_2rdm = jnp.transpose(r_prime_sweeps_2rdm, (2, 0, 1, 3))  # Shape: (batch_size, n_sweeps, 2, 3)

        # SINGLE WALKER MATH
        def single_walker_math(walker_data, walker_rp_1rdm, walker_rp_2rdm):
            electrons = walker_data[self.data_field]
            nelec = electrons.shape[0]

            phase, log_mag = self.phase_logpsi(params, walker_data)
            varphi_r = self._evaluate_mo(electrons)

            # 1-RDM
            varphi_rp_1rdm = self._evaluate_mo(walker_rp_1rdm)
            fsum_rp_1rdm = self._fsum(walker_rp_1rdm)
            varphi_rp_1rdm_over_f = varphi_rp_1rdm / fsum_rp_1rdm[:, None]

            unnorm_Ni_sq = jnp.abs(varphi_rp_1rdm)**2 / fsum_rp_1rdm[:, None]
            Ni_sq = jnp.mean(unnorm_Ni_sq, axis=0)

            def displace_one(a, rp):
                displaced = electrons.at[a].set(rp)
                return self.phase_logpsi(params, walker_data.merge({self.data_field: displaced}))

            vmap_displace_one = jax.vmap(jax.vmap(displace_one, in_axes=(None, 0)), in_axes=(0, None))

            idx_up = jnp.arange(self.n_up)
            idx_down = jnp.arange(self.n_up, nelec)

            phase_prime_up, log_mag_prime_up = vmap_displace_one(idx_up, walker_rp_1rdm)
            phase_prime_down, log_mag_prime_down = vmap_displace_one(idx_down, walker_rp_1rdm)

            wf_ratio_up = (phase_prime_up / phase) * jnp.exp(log_mag_prime_up - log_mag)
            wf_ratio_down = (phase_prime_down / phase) * jnp.exp(log_mag_prime_down - log_mag)

            one_rdm_up = jnp.einsum(
                "aA,ai,Aj->ij", jnp.conj(wf_ratio_up), jnp.conj(varphi_r[:self.n_up]), varphi_rp_1rdm_over_f
            ) / self.n_sweeps

            one_rdm_down = jnp.einsum(
                "aA,ai,Aj->ij", jnp.conj(wf_ratio_down), jnp.conj(varphi_r[self.n_up:]), varphi_rp_1rdm_over_f
            ) / self.n_sweeps

            # 2-RDM
            rp1_2rdm = walker_rp_2rdm[:, 0, :]
            rp2_2rdm = walker_rp_2rdm[:, 1, :]

            varphi_rp1_2rdm = self._evaluate_mo(rp1_2rdm)
            varphi_rp2_2rdm = self._evaluate_mo(rp2_2rdm)
            fsum_rp1_2rdm = self._fsum(rp1_2rdm)
            fsum_rp2_2rdm = self._fsum(rp2_2rdm)

            varphi_rp1_2rdm_over_f = varphi_rp1_2rdm / fsum_rp1_2rdm[:, None]
            varphi_rp2_2rdm_over_f = varphi_rp2_2rdm / fsum_rp2_2rdm[:, None]

            def displace_two(a, b, r1, r2):
                displaced = electrons.at[a].set(r1).at[b].set(r2)
                return self.phase_logpsi(params, walker_data.merge({self.data_field: displaced}))

            vmap_sweeps_2rdm = jax.vmap(displace_two, in_axes=(None, None, 0, 0))
            vmap_b_2rdm = jax.vmap(vmap_sweeps_2rdm, in_axes=(None, 0, None, None))
            vmap_a_2rdm = jax.vmap(vmap_b_2rdm, in_axes=(0, None, None, None))

            phase_prime_uu, log_mag_prime_uu = vmap_a_2rdm(idx_up, idx_up, rp1_2rdm, rp2_2rdm)
            phase_prime_dd, log_mag_prime_dd = vmap_a_2rdm(idx_down, idx_down, rp1_2rdm, rp2_2rdm)
            phase_prime_ud, log_mag_prime_ud = vmap_a_2rdm(idx_up, idx_down, rp1_2rdm, rp2_2rdm)

            wf_ratio_uu = (phase_prime_uu / phase) * jnp.exp(log_mag_prime_uu - log_mag)
            wf_ratio_dd = (phase_prime_dd / phase) * jnp.exp(log_mag_prime_dd - log_mag)
            wf_ratio_ud = (phase_prime_ud / phase) * jnp.exp(log_mag_prime_ud - log_mag)

            mask_uu = (1.0 - jnp.eye(self.n_up))[:, :, None]
            mask_dd = (1.0 - jnp.eye(self.n_down))[:, :, None]

            wf_ratio_uu = wf_ratio_uu * mask_uu
            wf_ratio_dd = wf_ratio_dd * mask_dd

            unnorm_two_rdm_uu = jnp.einsum(
                "abs,ai,bj,sk,sl->ijkl",
                jnp.conj(wf_ratio_uu), jnp.conj(varphi_r[:self.n_up]), jnp.conj(varphi_r[:self.n_up]),
                varphi_rp1_2rdm_over_f, varphi_rp2_2rdm_over_f
            ) / self.n_sweeps

            unnorm_two_rdm_dd = jnp.einsum(
                "abs,ai,bj,sk,sl->ijkl",
                jnp.conj(wf_ratio_dd), jnp.conj(varphi_r[self.n_up:]), jnp.conj(varphi_r[self.n_up:]),
                varphi_rp1_2rdm_over_f, varphi_rp2_2rdm_over_f
            ) / self.n_sweeps

            unnorm_two_rdm_ud = jnp.einsum(
                "abs,ai,bj,sk,sl->ijkl",
                jnp.conj(wf_ratio_ud), jnp.conj(varphi_r[:self.n_up]), jnp.conj(varphi_r[self.n_up:]),
                varphi_rp1_2rdm_over_f, varphi_rp2_2rdm_over_f
            ) / self.n_sweeps

            return {
                "unnorm_one_rdm_up": one_rdm_up,
                "unnorm_one_rdm_down": one_rdm_down,
                "unnorm_two_rdm_up_up": unnorm_two_rdm_uu,
                "unnorm_two_rdm_down_down": unnorm_two_rdm_dd,
                "unnorm_two_rdm_up_down": unnorm_two_rdm_ud,
                "Ni_sq": Ni_sq,
            }

        walker_stats = jax.vmap(single_walker_math, in_axes=(batched_data.vmap_axis, 0, 0))(
            batched_data.data,
            r_prime_per_walker_1rdm,
            r_prime_per_walker_2rdm
        )

        walker_stats["aux_pmove"] = jnp.broadcast_to(aux_stats["pmove"], (batched_data.batch_size,))
        new_state = {
            "r_prime_pool": r_prime_pool,
            "sampler_state": sampler_state
        }

        return walker_stats, new_state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        """Reduce per-walker statistics to step-level means.

        Since RDMs are large complex matrices/tensors, we skip variance computation.
        """
        return mean_reduce(walker_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        """Average over steps and normalize the RDMs.

        Args:
            batched_stats: Dictionary containing step-averaged values over the entire run.
                           Each value has shape (n_steps, ...).
            state: Unused here.

        Returns:
            Dictionary containing final, normalized 1-RDM and 2-RDM matrices,
            along with optional diagnostics like traces.
        """
        # 1. Average the step-level statistics over all MCMC steps (axis 0)
        mean_stats = {
            k: jnp.nanmean(v, axis=0) for k, v in batched_stats.items()
        }

        # 2. Extract the normalization factors
        Ni_sq = mean_stats["Ni_sq"]

        norm_1rdm = jnp.sqrt(Ni_sq[:, None] * Ni_sq[None, :])

        norm_2rdm = jnp.sqrt(
            Ni_sq[:, None, None, None] *
            Ni_sq[None, :, None, None] *
            Ni_sq[None, None, :, None] *
            Ni_sq[None, None, None, :]
        )

        # 3. Apply normalization
        one_rdm_up = mean_stats["unnorm_one_rdm_up"] / norm_1rdm
        one_rdm_down = mean_stats["unnorm_one_rdm_down"] / norm_1rdm

        two_rdm_uu = mean_stats["unnorm_two_rdm_up_up"] / norm_2rdm
        two_rdm_dd = mean_stats["unnorm_two_rdm_down_down"] / norm_2rdm
        two_rdm_ud = mean_stats["unnorm_two_rdm_up_down"] / norm_2rdm

        trace_up = jnp.trace(one_rdm_up)
        trace_down = jnp.trace(one_rdm_down)

        return {
            "one_rdm_up": one_rdm_up,
            "one_rdm_down": one_rdm_down,
            "two_rdm_up_up": two_rdm_uu,
            "two_rdm_down_down": two_rdm_dd,
            "two_rdm_up_down": two_rdm_ud,
            "one_rdm_up:trace": trace_up,
            "one_rdm_down:trace": trace_down,
            "aux_pmove_mean": mean_stats.get("aux_pmove", 0.0),
        }