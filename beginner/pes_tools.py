"""Shared helpers for the beginner track: the HeH+ potential energy surface.

Keep reusable code here rather than in notebooks: .py files diff and merge
cleanly in git, notebooks don't. Each function computes ONE geometry; the
sweep over bond distances lives in the notebook so we can experiment with it.

Every energy returned is a TOTAL energy in Hartree (electronic + nuclear
repulsion), so all methods can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.units import DistanceUnit

HARTREE_TO_KCAL_MOL = 627.509
CHEMICAL_ACCURACY = 1.6e-3  # Hartree, ~1 kcal/mol


def build_problem(
    distance: float,
    basis: str = "sto3g",
    unit: DistanceUnit = DistanceUnit.ANGSTROM,
) -> ElectronicStructureProblem:
    """HeH+ with He at the origin and H on the z-axis at ``distance``.

    HeH+ has 2 electrons (He brings 2, H brings 1, charge +1 removes 1),
    so it is a closed-shell singlet: charge=1, spin=0.
    """
    driver = PySCFDriver(
        atom=f"He 0 0 0; H 0 0 {distance}",
        basis=basis,
        charge=1,
        spin=0,
        unit=unit,
    )
    return driver.run()


def hf_energy(problem: ElectronicStructureProblem, mapper: QubitMapper | None = None) -> float:
    """Hartree-Fock (mean-field) total energy, evaluated on the HF circuit."""
    mapper = mapper or JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    evs = StatevectorEstimator().run([(hf, qubit_op)]).result()[0].data.evs
    return float(evs) + problem.nuclear_repulsion_energy


def exact_energy(problem: ElectronicStructureProblem, mapper: QubitMapper | None = None) -> float:
    """Exact (full diagonalisation) total ground-state energy in this basis.

    PITFALL: the qubit Hamiltonian acts on the whole Fock space, i.e. on
    states with ANY number of electrons. Unfiltered, NumPyMinimumEigensolver
    returns the lowest eigenvalue over all of them; for HeH+ that's a state
    with extra electrons (~0.16 Ha too low at equilibrium). It happened to
    work for H2 in the tutorial. The filter restricts the search to states
    with the right electron count and spin.
    """
    mapper = mapper or JordanWignerMapper()
    numpy_solver = NumPyMinimumEigensolver()
    numpy_solver.filter_criterion = problem.get_default_filter_criterion()
    solver = GroundStateEigensolver(mapper, numpy_solver)
    return float(solver.solve(problem).total_energies[0])


@dataclass
class VQEPoint:
    energy: float              # total energy, Hartree
    optimal_point: np.ndarray  # optimised ansatz parameters (useful as a warm start)
    num_qubits: int
    num_parameters: int
    cost_function_evals: int


def uccsd_ansatz(problem: ElectronicStructureProblem, mapper: QubitMapper) -> UCCSD:
    """UCCSD on top of a Hartree-Fock reference, as in the tutorial."""
    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    return UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=hf,
    )


def vqe_energy(
    problem: ElectronicStructureProblem,
    mapper: QubitMapper | None = None,
    ansatz=None,
    optimizer=None,
    estimator=None,
    initial_point: np.ndarray | None = None,
) -> VQEPoint:
    """Run VQE at one geometry. Every ingredient can be swapped out.

    Defaults reproduce the tutorial: Jordan-Wigner, HF+UCCSD, SLSQP,
    noiseless StatevectorEstimator, starting from all-zero parameters (= HF).
    """
    mapper = mapper or JordanWignerMapper()
    ansatz = ansatz if ansatz is not None else uccsd_ansatz(problem, mapper)
    optimizer = optimizer or SLSQP()
    estimator = estimator or StatevectorEstimator()

    vqe = VQE(estimator, ansatz, optimizer)
    vqe.initial_point = (
        np.asarray(initial_point) if initial_point is not None else np.zeros(ansatz.num_parameters)
    )
    result = GroundStateEigensolver(mapper, vqe).solve(problem)
    raw = result.raw_result
    return VQEPoint(
        energy=float(result.total_energies[0]),
        optimal_point=np.asarray(raw.optimal_point),
        num_qubits=ansatz.num_qubits,
        num_parameters=ansatz.num_parameters,
        cost_function_evals=int(raw.cost_function_evals or 0),
    )
