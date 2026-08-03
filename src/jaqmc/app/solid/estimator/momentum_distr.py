r"""Reduced density matrix (RDM) estimator for periodic solid state systems.

Computes the 1-RDM and 2-RDM using meta-Lowdin localized orbitals from PySCF
and MCMC importance sampling of the auxiliary coordinate $r'$.
"""

from collections.abc import Mapping
from typing import Optional, Sequence
from typing import Any
import logging

import jax
from jax import numpy as jnp

from pyscf import lo
from pyscf.pbc.tools.pbc import super_cell

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
class MomentumDistribution(Estimator):
    r"""Momentum distribution n(k)
    """

    # These get values from make_estimators() in workflow.py
    # f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    phase_logpsi: WavefunctionEvaluate = runtime_dep()
    scf: PeriodicSCF = runtime_dep() 
    data_field: str = runtime_dep(default="electrons")
    supercell_matrix: Optional[Sequence[Sequence[int]]] = runtime_dep(default=None)
    
    # one-body sampling MCMC settings 
    # These can be set in the config yaml file, e.g., sc_h_chain_rdm.yml
    n_sweeps: int = 1 #same as in rdm_base_config.py
    aux_steps: int = 10 
    # aux_initial_width: float = 0.1
    
    # Batching parameters needed for the auxiliary pool
    batch_size: int = 2048          # Default JaQMC run config
    ratio_naux_nbatch: float = 2.0  # From original config
    
    phi_i: Optional[str] = None
    phi_j: Optional[str] = None
    
    N: int = runtime_dep(default=10) # Number of atoms/sites
    a: float = runtime_dep(default=2.0) # Interatomic distance
    
    # supercell_matrix: Optional[Sequence[Sequence[int]]] = None

    def init(self, data: SolidData, rngs: PRNGKey) -> dict[str, Any]:
        """ 
        Initialize the auxiliary electron coordinates for 1-RDM and 2-RDM sampling.
    
        """
        
        # From Estimator.init(), this function is called only once per device, and results shared across all walkers.
        # So the burn-in MCMC is done once here only
        
        #cell has basis, spin, charge, ecp, 
        prim_cell = self.scf._cell # pyscf_cell: pyscf.pbc.gto.Cell, equivalent to self._mol in Ferminet rdm.py
        
        # 1. Build the supercell if the matrix was provided in the YAML
        # Looks like in data_init for SolidData, R is sampled around the supercell, so here we also need to work in supercell 
        if self.supercell_matrix is not None:
            # Extract the [3, 1, 1] diagonal array for the ncopy argument
            ncopy = [int(self.supercell_matrix[i][i]) for i in range(3)]
            logging.info(f"Building PySCF supercell using ncopy: {ncopy}")
            
            # Make supercell from prim_cell
            supcell = super_cell(prim_cell, ncopy, wrap_around=False)
        else:
            logging.info("No supercell specificed, running on unit cell")
            supcell = prim_cell
        
        self._lattice_vectors = jnp.array(supcell.lattice_vectors())
            
        # lattice_vectors() is a built-in function in the pyscf.pbc.gto.Cell class
        # lattice_vecotors() returns a 3x3 array of the lattice vectors of the unit cell in Cartesian coordinates
        # self._lattice_vectors = jnp.array(cell.lattice_vectors())
        
        
        
        logging.info("Calculating Meta-Lowdin MO coefficients")
        self._mo_coeff = jnp.array(lo.orth_ao(supcell, 'meta-lowdin')) #same as in Ferminet rdm.py
        
        # In the H-chain paper it's using next-nearest-neighbour cutoff. I think here it's different by using PBCAtomicOrbitalEvaluator's estimate_rcut
        self._ao_evaluator = PBCAtomicOrbitalEvaluator.from_pyscf(supcell) #Later use with _mo_coeff to get localized meta-Lowdin orbitals
        # self._kpts = jnp.asarray(self.scf.get_orbital_kpoints())
        # This is used to evaluate MOs later after meta-Lowdin orthogonalization. I think pySCF meta-Lowdin orthogonalizes using the same orbitals on each lattice point (k = 0), so when you evaluate you need to set k = 0 when evaluating nearby orbitals too? 
        self._kpts = jnp.zeros((1, 3))
        
        
        self.n_up = supcell.nelec[0] 
        self.n_down = supcell.nelec[1]
        
        key_init, key_burn = jax.random.split(rngs)
        
        # Using self.n_sweeps directly (or data.r.shape[0] * self.n_sweeps if tracking per-walker)
        # Note currently this is not the same as FermiNet's _init_electrons. 
        # Here we just initialize uniformly along axis of 1D chain, then Gaussian in the other two axes. 
        
        # If self.batch_size is 2048, ratio is 2.0, and GPUs are 4:
        # 2048 * 2.0 * 4 = 16384 total global points.
        # After sharding, each GPU gets 4096 local points (plenty for your 1024 walker batch)
        # ---------------------------------------------------------
        global_naux = int(self.batch_size * self.ratio_naux_nbatch * jax.device_count())
        r_prime_pool = self._init_electrons(key=key_init, num_samples=global_naux)
        # r_prime_pool = self._init_electrons(key=key_init, num_samples=int(self.batch_size * self.ratio_naux_nbatch))
        
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

        return {
            "r_prime_pool": r_prime_pool, 
            "sampler_state": self._pad_sampler_state(sampler_state, r_prime_pool.shape[0]), 
            "burn_in_counter": jnp.zeros_like(r_prime_pool[:, 0], dtype=jnp.int32)
        } #last one is a flag so that r_prime_pool is burned in only once
    
    
    # Maybe can use initialize_electrons_gaussian as in src/jaqmc/app/solid/data.py? I think the Ferminet rdm.py used something like that, but because it generates for multi-electron config we'll also have to flatten and cut. After burn-in should be same?
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
    
    def _pad_sampler_state(self, state_tree, size):
        """Prepends the batch dimension to all sampler statistics (scalars and arrays)."""
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (size,) + jnp.shape(x)),
            state_tree
        )

    def _unpad_sampler_state(self, state_tree):
        """Strips the dummy batch dimension back out for the sampler."""
        return jax.tree_util.tree_map(
            lambda x: x[0],
            state_tree
        )

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
        sampler_state = self._unpad_sampler_state(state["sampler_state"]) # Unpad it!
        burn_in_counter = state["burn_in_counter"]
        
        rngs, burn_rng, step_rng, sweep_rng = jax.random.split(rngs, 4)
        
        num_burn_in_steps = 400
        def burn_in(carry):
            r_prime_pool, sampler_state, rng = carry

            def body(_, val):
                r_prime_pool, sampler_state, rng = val
                rng, subkey = jax.random.split(rng)

                r_prime_pool, _, sampler_state = self._aux_sampler.step(
                    batch_log_prob=self._log_fsum,
                    data=r_prime_pool,
                    state=sampler_state,
                    rngs=subkey,
                )

                return r_prime_pool, sampler_state, rng

            return jax.lax.fori_loop(
                0,
                num_burn_in_steps,
                body,
                (r_prime_pool, sampler_state, rng),
            )
            
        def do_burn(_):
            rp_out, state_out, _ = burn_in(
                (r_prime_pool, sampler_state, burn_rng)
            )
            return rp_out, state_out, jnp.ones_like(burn_in_counter)

        def skip(_):
            return r_prime_pool, sampler_state, burn_in_counter

        logging.info(f"Starting burn-in of {num_burn_in_steps} steps of r_prime_pool")
        r_prime_pool, sampler_state, burn_in_counter = jax.lax.cond(
            burn_in_counter[0] == 0,
            do_burn,
            skip,
            operand=None,
        )
        logging.info("Completed burn-in of r_prime_pool")

        # =======================================================
        # 1. MC Step to update r_prime_pool for this batch
        # =======================================================
        # rngs, subkey_mcmc = jax.random.split(rngs)
        logging.info("Starting MC Step to update r_prime_pool")
        r_prime_pool, aux_stats, sampler_state = self._aux_sampler.step(
            batch_log_prob=self._log_fsum,
            data=r_prime_pool,
            state=sampler_state,
            rngs=step_rng
        )
        logging.info("Completed MC Step to update r_prime_pool")
        

        # =======================================================
        # 2. Pair r_prime's in r_prime_pool with each walker in the batch 
        # =======================================================
        rngs, subkey_1rdm = jax.random.split(sweep_rng, 2)
        sweep_keys_1rdm = jax.random.split(subkey_1rdm, self.n_sweeps)
        
        # --- 1-RDM Sampling ---
        def draw_single_sweep_1rdm(key):
            return jax.random.choice(
                key, 
                r_prime_pool, 
                shape=(batched_data.batch_size,), 
                replace=False, 
                axis=0
            )
            
        # logging.info(f"Sampling {self.n_sweeps} r_prime's per walker for 1RDM")
        r_prime_sweeps_1rdm = jax.vmap(draw_single_sweep_1rdm)(sweep_keys_1rdm)
        # Swap axes so the main batch dimension is first.
        r_prime_per_walker_1rdm = jnp.swapaxes(r_prime_sweeps_1rdm, 0, 1)  # Shape: (batch_size, n_sweeps, 3)
        
        
        # =======================================================
        # 3. PURE SINGLE WALKER MATH 
        # =======================================================
        def single_walker_math(walker_data, walker_rp_1rdm):
            electrons = walker_data[self.data_field] #shape: (nelec, 3), electron positions from R
            nelec = electrons.shape[0]
            
            phase, log_mag = self.phase_logpsi(params, walker_data) #phase_logpsi defined in jaqmc/src/jaqmc/app/solid/workflow.py, using function defined in src/jaqmc/app/solid/wavefunction.py
            varphi_r = self._evaluate_mo(electrons)  #phi_i(r), shape: (nelec, norb)         
            
            # ---------------------------------------------------
            # 1-RDM 
            # ---------------------------------------------------
            # Here we assume restricted orbitals? where phi(r) is the same for both up and down spins?
            # So only need to worry about indices in Phi(R/R'/R'')
            logging.info("Starting calculation of n(k)")
            
            fsum_rp_1rdm = self._fsum(walker_rp_1rdm) #f(r')
            
            
            def displace_first(rp):
                # Displace by r' relative to the original position of electron 'a'
                new_pos = electrons[0] + rp
                displaced = electrons.at[0].set(new_pos) 
                return self.phase_logpsi(params, walker_data.merge({self.data_field: displaced}))
                
            # Vmap ONLY over the sweeps (r' pool). 
            # in_axes=(0,) tells JAX to map over the first axis of the single argument 'rp'.
            vmap_displace = jax.vmap(displace_first, in_axes=(0,))
            
            #Phi(r1+r', r2, ....), Evaluate using strictly the first electron
            phase_prime, log_mag_prime = vmap_displace(walker_rp_1rdm) 
            
            # Wavefunction ratio Phi*(r1+r', r2...)/Phi*(r1, r2...)
            # (Conjugation applied later via jnp.conj)
            conj_ratio = jnp.conj((phase_prime / phase) * jnp.exp(log_mag_prime - log_mag))
            
            # k = 2 * pi * n / (N * a) 
            n_vals = jnp.arange(-self.N // 2, self.N // 2 + 1)
            k_vals = 2.0 * jnp.pi * n_vals / (self.N * self.a)
            
            # Extract x-coordinates of auxiliary pool 
            rx_prime = walker_rp_1rdm[:, 0] # shape (n_sweeps,)
            exp_ikr = jnp.exp(1j * jnp.outer(k_vals, rx_prime)) # shape (n_k, n_sweeps)
            
            
            
            # Perform Fourier projection expectation against importance sampled f(r') pool.
            # Scale by `nelec` (which is 2 * n_up for a spin-balanced system) 
            # to account for the sum over all indistinguishable particles across both spins.
            nk_total = nelec * jnp.einsum("A,kA,A->k", conj_ratio, exp_ikr, 1.0 / fsum_rp_1rdm) / self.n_sweeps
            
            # ------------------------------------------ Return Stats ---------------------------------------
            logging.info("Completed Momentum Distribution calculations")
            return {
                "n_k_total": nk_total,
            }
            
        # =======================================================
        # 4. VMAP THE MATH ACROSS ALL WALKERS
        # =======================================================
        logging.info("Starting batch calculations of n(k)")
        walker_stats = jax.vmap(single_walker_math, in_axes=(batched_data.vmap_axis, 0))(
            batched_data.data, 
            r_prime_per_walker_1rdm
        )
        logging.info("Completed batch calculations of n(k)")
        
        new_state = {
            "r_prime_pool": r_prime_pool, 
            "sampler_state": self._pad_sampler_state(sampler_state, r_prime_pool.shape[0]), # Repad it!
            "burn_in_counter": burn_in_counter
        }
        logging.info("Completed evaluate_batch_walkers()")
        
        return walker_stats, new_state
    
    
    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
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
        
        
        logging.info("Finalizing momentum distribution n(k)")
        
        return {
            "n_k_total": mean_stats["n_k_total"],
        }
