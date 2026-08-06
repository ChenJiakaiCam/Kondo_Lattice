r"""
"""

from collections.abc import Mapping
from typing import Optional, Sequence
from typing import Any
import logging

import jax
from jax import numpy as jnp

from pyscf.pbc.tools.pbc import super_cell

from jaqmc.array_types import Params, PRNGKey
from jaqmc.app.solid.data import SolidData
from jaqmc.data import BatchedData
from jaqmc.estimator.base import Estimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import WavefunctionEvaluate # To get phases
from jaqmc.utils.atomic.scf import PeriodicSCF
from jaqmc.geometry.pbc import wrap_positions

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
    
    
    N: int = 10 # Number of atoms/sites
    a: float = 2.0 # Interatomic distance
    
    def init(self, data: SolidData, rngs: PRNGKey) -> dict[str, Any]:
        """ 
        Initialize the auxiliary electron coordinates for MCMC sampling of the momentum distribution n(k).
    
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
        self.n_up = supcell.nelec[0] 
        self.n_down = supcell.nelec[1]
        
        return {}
    

    def evaluate_batch_walkers(
        self,
        params: Params, #neural network parameters, not used in this estimator
        batched_data: BatchedData[SolidData], # batched_data contains the fields: electrons, atoms, primitive_atoms, charges (electrons have shape (n_batch, n_elec, 3) )
        prev_walker_stats: Mapping[str, Any],
        state: Any, #state is initialized in init() and updated in evaluate_batch_walkers. Contain RDM values and the auxiliary electron coordinates for MCMC sampling
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        del prev_walker_stats  # Not used here, but included for compatibility with the Estimator interface.
        
        # Creates r' strictly at m * a along the x-axis
        m_vals = jnp.arange(self.N)
        r_prime_discrete = jnp.zeros((self.N, 3))
        r_prime_discrete = r_prime_discrete.at[:, 0].set(m_vals * self.a)
        
        
        def single_walker_math(walker_data):
            electrons = walker_data[self.data_field] #shape: (nelec, 3), electron positions from R
            nelec = electrons.shape[0]
            
            phase, log_mag = self.phase_logpsi(params, walker_data) #phase_logpsi defined in jaqmc/src/jaqmc/app/solid/workflow.py, using function defined in src/jaqmc/app/solid/wavefunction.py
            
            def displace_first(rp):
                # Displace by r' relative to the original position of electron 'a'
                new_pos = electrons[0] + rp
                new_pos = wrap_positions(new_pos, self._lattice_vectors)
                displaced = electrons.at[0].set(new_pos) 
                return self.phase_logpsi(params, walker_data.merge({self.data_field: displaced}))
                
            # in_axes=(0,) tells JAX to map over the first axis of the single argument 'rp'.
            vmap_displace = jax.vmap(displace_first, in_axes=(0,))
            #Phi(r1+r', r2, ....), Evaluate using strictly the first electron
            phase_prime, log_mag_prime = vmap_displace(r_prime_discrete) 
            
            # Wavefunction ratio Phi*(r1+r', r2...)/Phi*(r1, r2...)
            conj_ratio = jnp.conj((phase_prime / phase) * jnp.exp(log_mag_prime - log_mag))
            
            # k = 2 * pi * n / (N * a) 
            Lx = self._lattice_vectors[0, 0]
            n_vals = jnp.arange(-self.N // 2, self.N // 2)
            k_vals = 2.0 * jnp.pi * n_vals / Lx
            
            exp_ikr = jnp.exp(1j * jnp.outer(k_vals, r_prime_discrete[:, 0]))
            
            # Sum over all displacements r' to get n(k) for this walker
            nk_total_complex = (nelec / self.N) * jnp.einsum("A,kA->k", conj_ratio, exp_ikr)
            
            # Discard statistical imaginary noise to enforce Hermiticity
            nk_total = jnp.real(nk_total_complex)
            
            return {
                "n_k_total": nk_total,
            }
            
        # =======================================================
        # VMAP THE MATH ACROSS ALL WALKERS
        # =======================================================
        logging.info("Starting batch calculations of n(k)")
        # +++ MODIFIED: We only map over the walker data now, since r_prime_discrete is global +++
        walker_stats = jax.vmap(single_walker_math, in_axes=(batched_data.vmap_axis,))(
            batched_data.data
        )
        
        # state is returned completely unchanged as there is no MCMC history +++
        return walker_stats, state
    
    
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
