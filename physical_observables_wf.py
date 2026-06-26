import os
os.environ["JAX_PLATFORMS"] = "cuda"

import time
import jax
import yaml
from pathlib import Path
from dataclasses import replace

from jaqmc.app.solid.workflow import SolidEvalWorkflow
from jaqmc.utils.config import ConfigManager
from jaqmc.workflow.base import init_batched_data
from jaqmc.estimator import EstimatorPipeline
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager
from jax import numpy as jnp
import jax.numpy as jnp
import dataclasses

from jaqmc.app.solid.hamiltonian import PotentialEnergy
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator import EstimatorPipeline
import jax

# ==========================================
# 1. Setup Paths & Load Config
# ==========================================
base_dir = Path("/rds/user/jc2405/hpc-work/JaQMC/jaqmc")
yaml_path = base_dir / "configs/sc_h_chain.yml"
checkpoint_dir = base_dir / "runs/sc_h_chain_3.78_spin_6_6"
target_ckpt_file = checkpoint_dir / "train_ckpt_009999.npz"

with open(yaml_path, "r") as f:
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


import numpy as np
from pathlib import Path

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

print(f"Successfully loaded {batched_data.batch_size} walkers from {target_ckpt_file.name}!") 


params = state.params
batched_data = state.batched_data

print("Loaded state contents:")
print(f"  - params: {type(params).__name__} with {sum(p.size for p in jax.tree.leaves(params)):,} parameters")
print(f"  - batched_data: {batched_data.batch_size} walkers")
print(f"  - electron positions shape: {batched_data.data.electrons.shape}")




P = jax.sharding.PartitionSpec

evaluate_wf = parallel_jax.jit_sharded(
    lambda p, bd: jax.vmap(wf.apply, in_axes=(None, bd.vmap_axis))(p, bd.data),
    in_specs=(P(), batched_data.partition_spec),
    out_specs=parallel_jax.DATA_PARTITION,
)
wf_output = evaluate_wf(params, batched_data)

print("Wavefunction output keys:", list(wf_output.keys()))
print(f"\nlog(ψ) shape: {wf_output['logpsi'].shape}")
print(f"log(ψ) statistics:")
print(f"  mean: {jnp.mean(wf_output['logpsi']):.4f}")
print(f"  std:  {jnp.std(wf_output['logpsi']):.4f}")





# Extract the electron coordinates array from the state
electrons = state.batched_data.data.electrons

# 1. Output the absolute global range (across all walkers, electrons, and x/y/z axes)
global_min = jnp.min(electrons)
global_max = jnp.max(electrons)
print(f"Global coordinate range: [{global_min:.4f}, {global_max:.4f}]")

# 2. Output the range specific to each spatial axis (X, Y, Z)
# By reducing across axis 0 (walkers) and axis 1 (electrons), we keep axis 2 (spatial dims)
axis_min = jnp.min(electrons, axis=(0, 1))
axis_max = jnp.max(electrons, axis=(0, 1))

print(f"X-axis range: [{axis_min[0]:.4f}, {axis_max[0]:.4f}]")
print(f"Y-axis range: [{axis_min[1]:.4f}, {axis_max[1]:.4f}]")
print(f"Z-axis range: [{axis_min[2]:.4f}, {axis_max[2]:.4f}]")




# Extract a single walker by indexing dim 0 of each batched field
single_data = dataclasses.replace(
    batched_data.data,
    **{k: batched_data.data[k][0] for k in batched_data.fields_with_batch},
)
single_output = wf.apply(params, single_data)


print("Single walker evaluation:")
# print(f"  log(ψ) = {single_output['logpsi']:.6f}")
complex_logpsi = single_output['logpsi']
print("Single walker evaluation:")
print(f"  log(ψ) = {complex_logpsi.real:.6f} + {complex_logpsi.imag:.6f}j")



# 1. Extract the computed supercell lattice directly from the workflow's wavefunction
supercell_lattice = eval_workflow.wf.simulation_lattice

# 2. Initialize the Solid estimators
kinetic_est = EuclideanKinetic(f_log_psi=wf.logpsi, data_field="electrons")
potential_est = PotentialEnergy(supercell_lattice=supercell_lattice)

estimators = {
    "kinetic": kinetic_est,
    "potential": potential_est,
}

# 3. Build and initialize the pipeline
pipeline = EstimatorPipeline(estimators)
estimator_state = pipeline.init(batched_data, jax.random.PRNGKey(0))

print("Pipeline initialized successfully!")


P = jax.sharding.PartitionSpec

evaluate = parallel_jax.jit_sharded(
    pipeline.evaluate,
    in_specs=(
        P(),                          # params: replicated
        batched_data.partition_spec,  # batched_data: batch dim sharded
        P(),                          # estimator_state: no array leaves
        P(),                          # rngs: replicated
    ),
    out_specs=(
        P(),                          # step_stats: reduced scalars
        P(),                          # estimator_state: no array leaves
    ),
)

mean_stats, estimator_state = evaluate(
    params, batched_data, estimator_state, jax.random.PRNGKey(1)
)

# finalize_stats() expects a leading step dimension — add one for single-step use
batched_mean_stats = jax.tree.map(lambda x: x[None], mean_stats)
final_stats = pipeline.finalize_stats(batched_mean_stats, estimator_state)

print(f"Computed observables (from {batched_data.batch_size} walkers):")
print(f"  Kinetic energy:   {final_stats['energy:kinetic']:.6f} Ha")
print(f"  Potential energy: {final_stats['energy:potential']:.6f} Ha")
total_energy = final_stats['energy:kinetic'] + final_stats['energy:potential']
print(f"  Total energy:     {total_energy:.6f} Ha")

print("Energy variance (from estimator pipeline):")
print(f"  Kinetic var:   {final_stats['energy:kinetic_var']:.6f}")
print(f"  Potential var: {final_stats['energy:potential_var']:.6f}")


import json
import jax.numpy as jnp

# 1. Convert JAX arrays to standard Python floats so they can be saved
# We explicitly call jnp.real() to discard the imaginary numerical noise
stats_to_save = {key: float(jnp.real(value)) for key, value in final_stats.items()}

# Add your manually calculated total energy (also taking the real part)
stats_to_save['energy:total'] = float(jnp.real(total_energy))

# 2. Define the path where the file will be saved
eval_results_dir = checkpoint_dir / "eval_results"
eval_results_dir.mkdir(parents=True, exist_ok=True) 

stats_file = eval_results_dir / f"epoch_9999_observables.json"

# 3. Write the dictionary to a JSON file
with open(stats_file, "w") as f:
    json.dump(stats_to_save, f, indent=4)

print(f"\nSuccessfully saved observables to: {stats_file}")

