r"""Reduced density matrix (RDM) estimator for periodic solid state systems.

Computes the 1-RDM and 2-RDM using meta-Lowdin localized orbitals from PySCF
and MCMC importance sampling of the auxiliary coordinate $r'$.
"""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp
from pyscf import lo

# from ferminet.utils.system import pyscf_mol_to_internal_representation
# from ferminet.train import init_electrons

from jaqmc.array_types import Params, PRNGKey
from jaqmc.app.solid.data import SolidData
from jaqmc.data import BatchedData
from jaqmc.estimator.base import Estimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
# from jaqmc.wavefunction.base import NumericWavefunctionEvaluate
from jaqmc.wavefunction.base import WavefunctionEvaluate # To get phases
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

    # f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    phase_logpsi: WavefunctionEvaluate = runtime_dep()
    scf: PeriodicSCF = runtime_dep() 
    data_field: str = runtime_dep(default="electrons")
    
    # one-body sampling MCMC settings 
    # These can be set in the config yaml file, e.g., sc_h_chain_rdm.yml
    n_sweeps: int = 1 #same as in rdm_base_config.py
    aux_steps: int = 10 
    # aux_initial_width: float = 0.1
    
    # Batching parameters needed for the auxiliary pool
    batch_size: int = 2048          # Default JaQMC run config
    ratio_naux_nbatch: float = 2.0  # From original config

    def init(self, data: SolidData, rngs: PRNGKey) -> dict[str, Any]:
        """ 
        Initialize the auxiliary electron coordinates for 1-RDM and 2-RDM sampling.
    
        """
        
        # From Estimator.init(), this function is called only once per device, and results shared across all walkers.
        # So the burn-in MCMC is done once here only
        
        #cell has basis, spin, charge, ecp, 
        cell = self.scf._cell # pyscf_cell: pyscf.pbc.gto.Cell, equivalent to self._mol in Ferminet rdm.py
        # lattice_vectors() is a built-in function in the pyscf.pbc.gto.Cell class
        # lattice_vecotors() returns a 3x3 array of the lattice vectors of the unit cell in Cartesian coordinates
        self._lattice_vectors = jnp.array(cell.lattice_vectors())
        self._mo_coeff = jnp.array(lo.orth_ao(cell, 'meta-lowdin')) #same as in Ferminet rdm.py
        # In the H-chain paper it's using next-nearest-neighbour cutoff. I think here it's different by using PBCAtomicOrbitalEvaluator's estimate_rcut
        self._ao_evaluator = PBCAtomicOrbitalEvaluator.from_pyscf(cell) #Later use with _mo_coeff to get localized meta-Lowdin orbitals
        self._kpts = jnp.asarray(self.scf.get_orbital_kpoints())
        
        # nelec = getattr(data, self.data_field).shape[0]
        # Store spin boundaries
        self.n_up = self.scf.nelectrons[0]
        self.n_down = self.scf.nelectrons[1]
        # nelec = self.n_up + self.n_down
        
        key_init, key_burn = jax.random.split(rngs)
        
        # Using self.n_sweeps directly (or data.r.shape[0] * self.n_sweeps if tracking per-walker)
        # Note currently this is not the same as FermiNet's _init_electrons. 
        # Here we just initialize uniformly along axis of 1D chain, then Gaussian in the other two axes. 
        r_prime_pool = self._init_electrons(key=key_init, num_samples=int(self.batch_size * self.ratio_naux_nbatch))
        
        self._aux_sampler = MCMCSampler(
            # initial_width and adapt_frequency from rdm_base_config.py
            # doesn't input f_sum here yet, the sampling_proposal is just telling how to propose new r' positions 
            steps=self.aux_steps, # aux_steps = 10, same as used in make_mcmc_obdm_step in Ferminet rdm.py
            initial_width=0.5,          
            adapt_frequency=10,
            sampling_proposal=make_pbc_gaussian_proposal(self._lattice_vectors) #By default electrons will wonder off the unit cell, but we only want to sample within the cell    
        )
        
        # Get the initial state for the adaptive MCMC
        sampler_state = self._aux_sampler.init(r_prime_pool, rngs)
        
        
        def burn_in_step(i, val):
            rp, state, key = val
            key, subkey = jax.random.split(key)
            rp, _, state = self._aux_sampler.step(
                batch_log_prob=self._log_fsum,
                data=rp,
                state=state, #contains data like the current width of the proposal distribution, acceptance rate, etc
                rngs=subkey
            )
            return rp, state, key
        
        num_burn_in_steps = 400  #follows rdm_base_config.py
        
        r_prime_pool, sampler_state, _ = jax.lax.fori_loop(
            0, 
            num_burn_in_steps, 
            burn_in_step, 
            (r_prime_pool, sampler_state, key_burn)
        )
        
        #r_prime.shape = (int(self.batch_size * self.ratio_naux_nbatch), 3)
        return {"r_prime_pool": r_prime_pool, "sampler_state": sampler_state}
    
    def _init_electrons(self, key: PRNGKey, num_samples: int) -> jnp.ndarray:
        """Initialize auxiliary electron coordinates for a 1D periodic chain."""
        key_x, key_yz = jax.random.split(key)
        
        # Extract the length of the unit cell along the x-axis
        # Assuming lattice_vectors is a 3x3 array where the first row is (Lx, 0, 0)
        Lx = self._lattice_vectors[0, 0] 
        
        # Uniform distribution for the x-coordinate: [0, Lx)
        x_init = jax.random.uniform(key_x, (num_samples, 1), minval=0.0, maxval=Lx)
        
        # Gaussian distribution for y and z coordinates (centered at 0)
        # Atomic radii for Sc-H are ~1-3 Bohrs; initialize with stddev of 1.0 Bohr
        yz_init = jax.random.normal(key_yz, (num_samples, 2)) * 1.0
        
        # Concatenate along the spatial axis to form the (num_samples, 3) array
        r_prime_pool = jnp.concatenate([x_init, yz_init], axis=1)
        
        # Force the initial coordinates to sit strictly within the primary cell (maybe should just cut off?)
        r_prime_pool = wrap_positions(r_prime_pool, self._lattice_vectors)
        
        return r_prime_pool

    #should still work even with PBC AOs?
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
        params: Params, #neural network parameters, not used in this estimator
        batched_data: BatchedData[SolidData], # batched_data contains the fields: electrons, atoms, primitive_atoms, charges (electrons have shape (n_batch, n_elec, 3) )
        prev_walker_stats: Mapping[str, Any],
        state: Any, #state is initialized in init() and updated in evaluate_batch_walkers. Contain RDM values and the auxiliary electron coordinates for MCMC sampling
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        del prev_walker_stats  # Not used here, but included for compatibility with the Estimator interface.
        r_prime_pool = state["r_prime_pool"]
        sampler_state = state["sampler_state"]

        # =======================================================
        # 1. MC Step to update r_prime_pool for this batch
        # =======================================================
        # rngs, subkey_mcmc = jax.random.split(rngs)
        r_prime_pool, aux_stats, sampler_state = self._aux_sampler.step(
            batch_log_prob=self._log_fsum,
            data=r_prime_pool,
            state=sampler_state,
            rngs=rngs
        )
        

        # =======================================================
        # 2. Pair r_prime's in r_prime_pool with each walker in the batch 
        # =======================================================
        rngs, subkey_1rdm, subkey_2rdm = jax.random.split(rngs, 3)
        sweep_keys_1rdm = jax.random.split(subkey_1rdm, self.n_sweeps)
        sweep_keys_2rdm = jax.random.split(subkey_2rdm, self.n_sweeps)
        
        # --- 1-RDM Sampling ---
        def draw_single_sweep_1rdm(key):
            return jax.random.choice(
                key, 
                r_prime_pool, 
                shape=(batched_data.batch_size,), 
                replace=False, 
                axis=0
            )
            
        r_prime_sweeps_1rdm = jax.vmap(draw_single_sweep_1rdm)(sweep_keys_1rdm)
        # Swap axes so the main batch dimension is first.
        r_prime_per_walker_1rdm = jnp.swapaxes(r_prime_sweeps_1rdm, 0, 1)  # Shape: (batch_size, n_sweeps, 3)
        
        # --- 2-RDM Sampling ---
        def draw_single_sweep_2rdm(key):
            # Same as in Ferminet rdm.py
            # Each sweep you shuffle, then pair the r_primes up, so get unique pairs
            rp = jax.random.shuffle(key, r_prime_pool)
            rp = rp[:2 * batched_data.batch_size, :]
            return rp.reshape((2, batched_data.batch_size, 3))
            
        r_prime_sweeps_2rdm = jax.vmap(draw_single_sweep_2rdm)(sweep_keys_2rdm)
        r_prime_per_walker_2rdm = jnp.transpose(r_prime_sweeps_2rdm, (2, 0, 1, 3))  # Shape: (batch_size, n_sweeps, 2, 3)
        
        # =======================================================
        # 3. PURE SINGLE WALKER MATH 
        # =======================================================
        def single_walker_math(walker_data, walker_rp_1rdm, walker_rp_2rdm):
            electrons = walker_data[self.data_field] #shape: (nelec, 3), electron positions from R
            nelec = electrons.shape[0]
            phase, log_mag = self.phase_logpsi(params, walker_data) #phase_logpsi defined in jaqmc/src/jaqmc/app/solid/workflow.py, using function defined in src/jaqmc/app/solid/wavefunction.py
            varphi_r = self._evaluate_mo(electrons)  #phi_i(r), shape: (nelec, norb)         
            
            # ---------------------------------------------------
            # 1-RDM 
            # ---------------------------------------------------
            varphi_rp_1rdm = self._evaluate_mo(walker_rp_1rdm)       
            fsum_rp_1rdm = self._fsum(walker_rp_1rdm)                
            varphi_rp_1rdm_over_f = varphi_rp_1rdm / fsum_rp_1rdm[:, None]
            
            # Calculate normalization factor (Ni_sq) based on the 1-RDM samples
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
            
            # ---------------------------------------------------
            # 2-RDM 
            # ---------------------------------------------------
            rp1_2rdm = walker_rp_2rdm[:, 0, :]  # Shape: (n_sweeps, 3)
            rp2_2rdm = walker_rp_2rdm[:, 1, :]  # Shape: (n_sweeps, 3)
            
            varphi_rp1_2rdm = self._evaluate_mo(rp1_2rdm)
            varphi_rp2_2rdm = self._evaluate_mo(rp2_2rdm)
            fsum_rp1_2rdm = self._fsum(rp1_2rdm)
            fsum_rp2_2rdm = self._fsum(rp2_2rdm)
            
            varphi_rp1_2rdm_over_f = varphi_rp1_2rdm / fsum_rp1_2rdm[:, None]
            varphi_rp2_2rdm_over_f = varphi_rp2_2rdm / fsum_rp2_2rdm[:, None]

            def displace_two(a, b, r1, r2):
                displaced = electrons.at[a].set(r1).at[b].set(r2)
                return self.phase_logpsi(params, walker_data.merge({self.data_field: displaced}))
                
            # Vectorize over sweeps (dim 2 & 3) and particles (dim 0 & 1)
            vmap_sweeps_2rdm = jax.vmap(displace_two, in_axes=(None, None, 0, 0))
            vmap_b_2rdm = jax.vmap(vmap_sweeps_2rdm, in_axes=(None, 0, None, None))
            vmap_a_2rdm = jax.vmap(vmap_b_2rdm, in_axes=(0, None, None, None))
            
            phase_prime_uu, log_mag_prime_uu = vmap_a_2rdm(idx_up, idx_up, rp1_2rdm, rp2_2rdm)
            phase_prime_dd, log_mag_prime_dd = vmap_a_2rdm(idx_down, idx_down, rp1_2rdm, rp2_2rdm)
            phase_prime_ud, log_mag_prime_ud = vmap_a_2rdm(idx_up, idx_down, rp1_2rdm, rp2_2rdm)
            
            wf_ratio_uu = (phase_prime_uu / phase) * jnp.exp(log_mag_prime_uu - log_mag)
            wf_ratio_dd = (phase_prime_dd / phase) * jnp.exp(log_mag_prime_dd - log_mag)
            wf_ratio_ud = (phase_prime_ud / phase) * jnp.exp(log_mag_prime_ud - log_mag)
            
            # Mask out self-interaction (a = b) for same-spin pairs[cite: 1]
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
            
            # ------------------------------------------ Return Stats ---------------------------------------
            return {
                "unnorm_one_rdm_up": one_rdm_up,
                "unnorm_one_rdm_down": one_rdm_down,
                "unnorm_two_rdm_up_up": unnorm_two_rdm_uu,
                "unnorm_two_rdm_down_down": unnorm_two_rdm_dd,
                "unnorm_two_rdm_up_down": unnorm_two_rdm_ud,
                "Ni_sq": Ni_sq,
            }
            
        # =======================================================
        # 4. VMAP THE MATH ACROSS ALL WALKERS
        # =======================================================
        walker_stats = jax.vmap(single_walker_math, in_axes=(batched_data.vmap_axis, 0, 0))(
            batched_data.data, 
            r_prime_per_walker_1rdm,
            r_prime_per_walker_2rdm
        )
        
        # Add the global auxiliary acceptance rate to the batched stats so we can track it
        walker_stats["aux_pmove"] = jnp.broadcast_to(aux_stats["pmove"], (batched_data.batch_size,))
        new_state = {
            "r_prime_pool": r_prime_pool, 
            "sampler_state": sampler_state
        }
        
        return walker_stats, new_state