# generate_plot_arrays.py
import os

os.environ["JAX_PLATFORMS"] = "cuda"  # MUST be cuda for the Wilkes3 A100!

import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import yaml
from jax import numpy as jnp

from jaqmc.app.solid.hamiltonian import PotentialEnergy
from jaqmc.app.solid.workflow import SolidEvalWorkflow
from jaqmc.estimator import EstimatorPipeline
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager
from jaqmc.workflow.base import init_batched_data

# ==========================================
# 1. Setup Paths & Load Config
# ==========================================
base_dir = Path("/rds/user/jc2405/hpc-work/JaQMC/jaqmc")
yaml_path = base_dir / "configs/sc_h_chain.yml"
checkpoint_dir = base_dir / "runs/sc_h_chain_3.78_spin_6_6"
target_ckpt_file = checkpoint_dir / "train_ckpt_009999.npz"

with open(yaml_path, encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)

# Inject overrides
raw_cfg["workflow"] = raw_cfg.get("workflow", {})
raw_cfg["workflow"]["source_path"] = str(target_ckpt_file)
raw_cfg["workflow"]["save_path"] = str(checkpoint_dir / "eval_results")
raw_cfg["estimators"] = {"enabled": {"density": True}}

# ==========================================
# 2. Define the Workflow
# ==========================================
cfg = ConfigManager(raw_cfg)
eval_workflow = SolidEvalWorkflow(cfg)

# 1. Define a cache file path inside your run directory
klist_cache_file = checkpoint_dir / "cached_klist.npy"

# 2. Check if we already did the 15-minute math
if klist_cache_file.exists():
    print("Loading cached k-points from disk...")

    # Load the array (allow_pickle=True handles lists of arrays)
    cached_klist = np.load(klist_cache_file, allow_pickle=True)

    # Inject it directly into the wavefunction
    eval_workflow.wf.klist = cached_klist
    print("Run SCF: Skipped (Loaded from cache!)")

else:
    print("Running SCF... ")

    # Run the expensive calculation
    eval_workflow.scf.run()

    # Extract the results
    klist = eval_workflow.scf.get_orbital_kpoints()
    eval_workflow.wf.klist = klist

    # Save the array to disk for all future runs
    np.save(klist_cache_file, klist)
    print(f"Run SCF: Complete. Data cached permanently to {klist_cache_file.name}")

# 2. Setup Random Keys
seed = int(time.time())
rngs = jax.random.PRNGKey(seed)
rngs, data_rngs = jax.random.split(rngs)

# 3. Initialize blank walkers based on your config
batched_data = init_batched_data(
    eval_workflow.data_init, eval_workflow.config.batch_size, data_rngs
)

# 4. Create an empty template state
rngs, sub_rngs = jax.random.split(rngs)
state = eval_workflow.evaluation_stage.create_state(sub_rngs, batched_data=batched_data)

# 5. Create the wrapper dictionary to tell the checkpoint manager what to load
wrapper = {
    "params": state.params,
    "batched_data": state.batched_data,
    "sampler_state": state.sampler_state,
}

# 6. Restore the checkpoint state (Call it on the evaluation_stage!)
target_ckpt_file = checkpoint_dir / "train_ckpt_009999.npz"
restored = eval_workflow.evaluation_stage.restore_checkpoint(
    target_ckpt_file, wrapper, prefix=""
)

# 7. Replace the empty state with the restored values
state = replace(
    state,
    params=restored["params"],
    batched_data=restored["batched_data"],
    sampler_state=restored["sampler_state"],
)

params = state.params
batched_data = state.batched_data
wf = eval_workflow.wf

print(
    f"Successfully loaded {batched_data.batch_size} walkers from {target_ckpt_file.name}!"
)

params = state.params
batched_data = state.batched_data

print("Loaded state contents:")
print(
    f"  - params: {type(params).__name__} with {sum(p.size for p in jax.tree.leaves(params)):,} parameters"
)
print(f"  - batched_data: {batched_data.batch_size} walkers")
print(f"  - electron positions shape: {batched_data.data.electrons.shape}")


# Define your pipeline and estimators
potential_est = PotentialEnergy(supercell_lattice=eval_workflow.wf.simulation_lattice)
kinetic_est = EuclideanKinetic(
    f_log_psi=eval_workflow.wf.logpsi, data_field="electrons"
)

estimators = {
    "kinetic": kinetic_est,
    "potential": potential_est,
}

# 3. Build and initialize the pipeline
pipeline = EstimatorPipeline(estimators)
estimator_state = pipeline.init(batched_data, jax.random.PRNGKey(0))


def compute_local_energy(params, data):
    kinetic_stats, _ = kinetic_est.evaluate_single_walker(
        params, data, {}, None, jax.random.PRNGKey(0)
    )
    potential_stats, _ = potential_est.evaluate_single_walker(
        params, data, {}, None, jax.random.PRNGKey(0)
    )
    return kinetic_stats["energy:kinetic"] + potential_stats["energy:potential"]


compute_local_energies = parallel_jax.jit_sharded(
    lambda p, bd: jax.vmap(
        lambda d: compute_local_energy(p, d),
        in_axes=(bd.vmap_axis,),
    )(bd.data),
    in_specs=(jax.sharding.PartitionSpec(), batched_data.partition_spec),
    out_specs=parallel_jax.DATA_PARTITION,
)

print("Compiling JAX graph and computing local energies... (See you in ~26 mins!)")
complex_local_energies = compute_local_energies(params, batched_data)
local_energies = jnp.real(complex_local_energies)
x_coords = batched_data.data.electrons[:, 0, 0]

arrays_cache_file = checkpoint_dir / "eval_results" / "epoch_9999_plot_arrays.npz"

# Add this line to guarantee the folder exists before saving!
arrays_cache_file.parent.mkdir(parents=True, exist_ok=True)

np.savez(arrays_cache_file, local_energies=local_energies, x_coords=x_coords)
print(f"Success! Saved raw plotting arrays to {arrays_cache_file.name}")
