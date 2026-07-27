import sys

# For rdm.py
content = open("src/jaqmc/app/solid/estimator/rdm.py").read()

if "get_supercell_kpts" not in content[:1000]:
    content = content.replace(
        "from jaqmc.geometry.pbc import make_pbc_gaussian_proposal, wrap_positions",
        "from jaqmc.geometry.pbc import make_pbc_gaussian_proposal, wrap_positions\nfrom jaqmc.utils.supercell import get_supercell_kpts, get_reciprocal_vectors"
    )

if "supercell_matrix: jnp.ndarray" not in content:
    content = content.replace(
        "    scf: PeriodicSCF = runtime_dep()",
        "    scf: PeriodicSCF = runtime_dep()\n    supercell_matrix: jnp.ndarray = runtime_dep()"
    )

open("src/jaqmc/app/solid/estimator/rdm.py", "w").write(content)
