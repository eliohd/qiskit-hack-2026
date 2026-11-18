# BasQ Qiskit Fall Fest 2026: Challenge 1, Beginner track

Potential energy surface (PES) of the **HeH⁺** molecular ion with VQE: sweep the He–H distance, plot the curve, find the equilibrium bond length, describe dissociation, and check whether the ansatz stays accurate everywhere.

## Layout

```
challenge_1_tutorial.ipynb   original H2 tutorial (reference, don't edit)
beginner/
  submission.ipynb           THE deliverable: sections mapped to judging criteria, owners, timeboxes
  pes_tools.py               chemistry helpers: build_problem, hf_energy, exact_energy, vqe_energy
  hardware_tools.py          qubit tapering, transpile/layout study, noisy sim, ZNE, Runtime estimator
  check_setup.py             run once to confirm your environment works
  results/                   CSVs of computed curves (committed)
  figures/                   PNG plots (committed)
requirements.txt             pinned, tested versions
```

## One-time setup (each of us)

Needs Python 3.11–3.13.

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
`pyscf` is pinned to 2.15.0 because it's the last release with prebuilt wheels for both Windows and Apple Silicon. If it still won't install on Windows, use WSL.

**Then, in both cases, inside the clone:**
```bash
nbstripout --install          # strips notebook outputs on commit (fewer merge conflicts)
nbdime config-git --enable    # notebook-aware git diff / merge
python beginner/check_setup.py
```
The check should end with `OK: environment works`.

## How we work together

Notebooks are JSON, so two people editing the same one produces painful conflicts. To keep things smooth:

1. **Shared logic goes in `beginner/pes_tools.py`.** Python files merge normally.
2. **One person edits `heh_plus_pes.ipynb` at a time.** For experiments, copy it to `beginner/scratch_<yourname>.ipynb`, then move the code that works into `pes_tools.py` or the main notebook.
3. **Save results to disk.** Outputs are stripped on commit, so write data to `results/*.csv` and plots to `figures/*.png` and commit those.
4. **Pull before you start, and commit small.**
   ```bash
   git pull --rebase
   # ...work...
   git add -A && git commit -m "Refine grid near equilibrium"
   git push
   ```
   For bigger experiments, use a branch (`git switch -c warm-start`) and merge it when it works.

## Things worth knowing

- **Use `exact_energy` from `pes_tools.py`, not the tutorial's version.** The tutorial's unfiltered `NumPyMinimumEigensolver` searches every electron count. For HeH⁺ the lowest state it finds has extra electrons, about 0.16 Ha below the real answer. `exact_energy` applies `problem.get_default_filter_criterion()` to fix this.
- All energies in `pes_tools.py` are **total** energies (electronic + nuclear repulsion), in Hartree.
- Chemical accuracy is 1.6 × 10⁻³ Ha ≈ 1 kcal/mol (`CHEMICAL_ACCURACY` in `pes_tools.py`).

## Roles (2-hour sprint)

The full timeline is at the top of `submission.ipynb`.

- **A: algorithm & physics** owns Sections 1–4: encodings, the classical cross-check, the noiseless PES, equilibrium and dissociation, and the larger basis and ansatz study. A also writes Section 8.
- **B: hardware & noise** owns Sections 5–7: the transpile and layout study, noisy simulation and ZNE, and the real-hardware run. **B submits the hardware job by about 0:45.**
- **Both** write Sections 8–9 and the abstract together, then do a final Restart & Run All.

Only edit cells in your own sections, and experiment in `beginner/scratch_<name>.ipynb`. With the nbdime merge driver enabled, edits to different cells merge cleanly.

**Before submitting:** run the notebook top to bottom and keep its outputs. Add `beginner/submission.ipynb -filter` to `.gitattributes` so the outputs survive commit, then export a copy with `jupyter nbconvert --to html beginner/submission.ipynb`.

## Credits

The tutorial notebook is by Benjamin Tirado for the BasQ Qiskit Fall Fest 2026 (text CC BY 4.0, code Apache 2.0). `beginner/` builds on its code.
