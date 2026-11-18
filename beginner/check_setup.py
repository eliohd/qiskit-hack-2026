"""Run this once after installing: `python beginner/check_setup.py`.

Checks the package versions and computes HeH+ at one bond distance with
Hartree-Fock, exact diagonalisation and VQE. If VQE matches exact to within
chemical accuracy, your environment is ready.
"""

import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

warnings.filterwarnings("ignore", category=Warning, module="scipy")
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("Python", sys.version.split()[0])
for pkg in ["qiskit", "qiskit-nature", "qiskit-algorithms", "qiskit-aer", "pyscf", "numpy", "scipy"]:
    print(f"  {pkg:<18} {version(pkg)}")

from pes_tools import CHEMICAL_ACCURACY, build_problem, exact_energy, hf_energy, vqe_energy  # noqa: E402

d = 0.774  # Angstrom, near the HeH+ equilibrium
t0 = time.perf_counter()
problem = build_problem(d)
e_hf = hf_energy(problem)
e_exact = exact_energy(problem)
point = vqe_energy(problem)
dt = time.perf_counter() - t0

print(f"\nHeH+ at {d} A (sto3g): {point.num_qubits} qubits, {point.num_parameters} UCCSD parameters")
print(f"  Hartree-Fock : {e_hf: .6f} Ha")
print(f"  VQE (UCCSD)  : {point.energy: .6f} Ha")
print(f"  Exact        : {e_exact: .6f} Ha")
err = abs(point.energy - e_exact)
print(f"  |VQE - exact| = {err:.2e} Ha   ({dt:.1f} s)")

if err < CHEMICAL_ACCURACY:
    print("\nOK: environment works, VQE is within chemical accuracy.")
else:
    sys.exit("\nFAIL: VQE did not reach chemical accuracy; check package versions.")
