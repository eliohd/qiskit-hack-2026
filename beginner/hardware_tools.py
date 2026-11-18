"""Helpers for the hardware-aware part: compact qubit encodings, transpilation
studies, noisy simulation, zero-noise extrapolation, and real-device runs.

Strategy (because of time and queue limits): optimise VQE parameters classically
on the noiseless simulator, then EVALUATE the energy at those fixed parameters
on noisy simulators and hardware. That isolates the effect of noise, so we can
quantify it and mitigate it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper

from pes_tools import build_problem, exact_energy, hf_energy, uccsd_ansatz, vqe_energy


# ---------------------------------------------------------------- encodings
def make_mapper(problem, kind: str = "tapered") -> QubitMapper:
    """'jw' = Jordan-Wigner (4 qubits for sto3g), 'parity' = parity with the
    two-qubit reduction (2 qubits), 'tapered' = parity + Z2 symmetry tapering."""
    if kind == "jw":
        return JordanWignerMapper()
    parity = ParityMapper(num_particles=problem.num_particles)
    if kind == "parity":
        return parity
    if kind == "tapered":
        return problem.get_tapered_mapper(parity)
    raise ValueError(kind)


@dataclass
class PreparedPoint:
    """Everything needed to evaluate one geometry on any backend."""
    distance: float
    qubit_op: SparsePauliOp
    circuit: QuantumCircuit      # ansatz with optimal parameters bound
    nuclear_repulsion: float
    e_hf: float
    e_exact: float
    e_vqe: float                  # noiseless statevector VQE


def prepare_point(distance: float, basis: str = "sto3g", mapping: str = "tapered",
                  initial_point=None) -> PreparedPoint:
    problem = build_problem(distance, basis=basis)
    mapper = make_mapper(problem, mapping)
    ansatz = uccsd_ansatz(problem, mapper)
    res = vqe_energy(problem, mapper=mapper, ansatz=ansatz, initial_point=initial_point)
    return PreparedPoint(
        distance=distance,
        qubit_op=mapper.map(problem.hamiltonian.second_q_op()),
        circuit=ansatz.assign_parameters(res.optimal_point),
        nuclear_repulsion=problem.nuclear_repulsion_energy,
        e_hf=hf_energy(problem, mapper),
        e_exact=exact_energy(problem, mapper),
        e_vqe=res.energy,
    )


# ---------------------------------------------------------------- transpilation
def two_qubit_count(circ: QuantumCircuit) -> int:
    return sum(1 for inst in circ.data if inst.operation.num_qubits == 2)


def layout_error_score(isa: QuantumCircuit, backend) -> float:
    """Rough fidelity cost of a transpiled circuit: sum of reported error rates
    of every gate and measurement it uses on the physical qubits it lands on."""
    target = backend.target
    score = 0.0
    for inst in isa.data:
        qargs = tuple(isa.find_bit(q).index for q in inst.qubits)
        name = inst.operation.name
        if name in target and qargs in target[name]:
            props = target[name][qargs]
            if props is not None and props.error is not None:
                score += props.error
    for q in active_qubits(isa):
        props = target["measure"][(q,)] if "measure" in target else None
        if props is not None and props.error is not None:
            score += props.error
    return score


def active_qubits(isa: QuantumCircuit) -> list[int]:
    used = {isa.find_bit(q).index for inst in isa.data for q in inst.qubits
            if inst.operation.name not in ("barrier", "delay")}
    return sorted(used)


def transpile_report(circ: QuantumCircuit, backend, optimization_level: int = 3,
                     initial_layout=None, seed: int = 0) -> dict:
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      initial_layout=initial_layout, seed_transpiler=seed)
    isa = pm.run(circ)
    return {
        "isa": isa,
        "depth": isa.depth(),
        "two_qubit_gates": two_qubit_count(isa),
        "physical_qubits": active_qubits(isa),
        "error_score": layout_error_score(isa, backend),
    }


def best_transpile(circ: QuantumCircuit, backend, n_seeds: int = 20, **kw) -> dict:
    """Try many transpiler seeds, keep the lowest error score (noise-aware layout)."""
    reports = [transpile_report(circ, backend, seed=s, **kw) for s in range(n_seeds)]
    return min(reports, key=lambda r: (r["error_score"], r["two_qubit_gates"]))


# ---------------------------------------------------------------- estimation
def estimate(isa_circuits: list[QuantumCircuit], qubit_op: SparsePauliOp, estimator,
             precision: float | None = None) -> np.ndarray:
    """Expectation values of qubit_op for ISA circuits (layout applied for you)."""
    pubs = [(c, qubit_op.apply_layout(c.layout)) for c in isa_circuits]
    result = estimator.run(pubs, precision=precision).result()
    return np.array([float(r.data.evs) for r in result]), np.array([float(r.data.stds) for r in result])


def noisy_estimator(backend, seed: int = 1234):
    """Aer estimator with the backend's calibrated noise model (local, free)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    sim = AerSimulator.from_backend(backend, seed_simulator=seed)
    return AerEstimator.from_backend(sim, options={"run_options": {"seed_simulator": seed}})


# ---------------------------------------------------------------- mitigation
def fold_two_qubit_gates(isa: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Local unitary folding for ZNE: replace every 2-qubit gate G by
    G (G^dag G)^((scale-1)/2). scale must be odd (1, 3, 5, ...). Works on ISA
    circuits because the inverse of a native 2q gate (cz/ecr) is native/self-inverse."""
    if scale % 2 != 1:
        raise ValueError("scale must be odd")
    folded = isa.copy_empty_like()
    for inst in isa.data:
        folded.append(inst)
        if inst.operation.num_qubits == 2 and inst.operation.name not in ("barrier",):
            for _ in range((scale - 1) // 2):
                folded.append(inst.operation.inverse(), inst.qubits, inst.clbits)
                folded.append(inst)
    folded._layout = isa.layout
    return folded


def zne_extrapolate(scales, values, order: int = 1) -> float:
    """Polynomial (order 1 = linear, 2 = quadratic) extrapolation to zero noise."""
    coeffs = np.polyfit(np.asarray(scales, float), np.asarray(values, float), order)
    return float(np.polyval(coeffs, 0.0))


def runtime_estimator(backend, twirling: bool = True, dd: bool = True,
                      resilience_level: int = 1, shots: int = 4096):
    """Qiskit Runtime estimator for REAL hardware with error suppression and mitigation.
    resilience_level 1 = TREX readout mitigation, 2 = + ZNE (Runtime's own)."""
    from qiskit_ibm_runtime import EstimatorV2
    est = EstimatorV2(mode=backend)
    est.options.default_shots = shots
    est.options.resilience_level = resilience_level
    est.options.dynamical_decoupling.enable = dd
    est.options.dynamical_decoupling.sequence_type = "XpXm"
    est.options.twirling.enable_gates = twirling
    est.options.twirling.enable_measure = twirling
    return est


def zne_std(scales, stds, order: int = 1) -> float:
    """Error bar of the ZNE value. The extrapolated energy is a weighted sum of the
    measured energies, so its std is sqrt(sum (w_i * std_i)^2). For scales (1, 3, 5)
    that is 1.21x the per-circuit std (linear) or 2.28x (quadratic)."""
    X = np.vander(np.asarray(scales, float), order + 1)
    weights = np.linalg.pinv(X)[-1]  # least-squares weights for the value at scale 0
    return float(np.sqrt(np.sum((weights * np.asarray(stds, float)) ** 2)))


# ---------------------------------------------------------------- hardware runs
def build_hardware_pubs(distances, scales, backend, n_seeds: int = 20):
    """One PUB per (distance, scale), noise-aware layout on THIS backend."""
    pubs, meta = [], []
    for d in distances:
        pp = prepare_point(d, mapping="tapered")
        best = best_transpile(pp.circuit, backend, n_seeds=n_seeds)
        for s in scales:
            c = fold_two_qubit_gates(best["isa"], s)
            pubs.append((c, pp.qubit_op.apply_layout(c.layout)))
            meta.append({"distance_A": d, "scale": s, "physical_qubits": best["physical_qubits"],
                         "exact": pp.e_exact, "vqe_noiseless": pp.e_vqe,
                         "nuclear_repulsion": pp.nuclear_repulsion})
    return pubs, meta


def hardware_table(result, meta, scales, backend_name: str, job_id: str):
    """Turn an Estimator result into one row per distance (raw + ZNE, with error bars)."""
    import pandas as pd
    rows = []
    for d in dict.fromkeys(m["distance_A"] for m in meta):
        idx = [i for i, m in enumerate(meta) if m["distance_A"] == d]
        E = np.array([float(result[i].data.evs) for i in idx]) + meta[idx[0]]["nuclear_repulsion"]
        sd = np.array([float(result[i].data.stds) for i in idx])
        m = meta[idx[0]]
        rows.append({"distance_A": d, "exact": m["exact"], "vqe_noiseless": m["vqe_noiseless"],
                     "hw_raw": E[0], "hw_raw_std": sd[0],
                     "hw_zne_linear": zne_extrapolate(scales, E, 1), "hw_zne_linear_std": zne_std(scales, sd, 1),
                     "hw_zne_quad": zne_extrapolate(scales, E, 2), "hw_zne_quad_std": zne_std(scales, sd, 2),
                     "physical_qubits": str(m["physical_qubits"]), "backend": backend_name, "job_id": job_id,
                     **{f"scale_{s}": e for s, e in zip(scales, E)},
                     **{f"std_{s}": e for s, e in zip(scales, sd)}})
    return pd.DataFrame(rows)
