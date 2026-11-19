# DTpy sequence generation handover

The sequence-generation work is on `local-wip`. The base sequence types are already committed under `DTpy/sequence/`. The generator modules, package initializer, sequence TODO, and sequence tests are staged additions. The root `TODOs` file is untracked.

## Work so far

- `DTpy/sequence/sequence.py` defines `Sequence` and `SequencePoint`. Generated waveforms use scalar gradient values paired with timepoints in seconds. The base type also keeps the `points` field and the `t` and `G` aliases.
- `STEAM` ports the development generator and stores its waveform in `steam.sequence`. It uses the shared trapezoidal solver and exposes diffusion, echo, mixing, flat-top, and ramp timings. `Constant` similarly stores its waveform in `constant.sequence` and exposes the discrete `dt` and `grad` lists.
- `Scanner` requires explicit gradient strength and slew rate. When the caller omits the scanner, `STEAM` uses the named `DefaultScanner` preset (40 mT/m, 60 T/m/s) and defaults to `Proton` for the spin.
- `PGSE`, `M2SE`, and general discretisation are still stubs. Their original development code remains in `dev/seqgen/Sequence/`, alongside the reference material in `dev/seqgen/`.

## Where to resume

1. Fix the import regression from splitting `Constant` into its own module. `DTpy/sequence/__init__.py` still imports it from `.discrete`. Import it from `.constant` so the existing `from DTpy import Constant` re-export resolves.
2. Update `tests/test_sequence_generation.py` for the current API. Read waveform fields from `steam.sequence` and `constant.sequence`, and replace the `Scanner()` defaults assertion with a check of `DefaultScanner` or STEAM's fallback behavior. Use the standard-library suite from a virtual environment with NumPy installed, via `python -m unittest discover -s tests -q`.
3. Reconcile the two copies of `solve_motion_compensated_sequence`. The original is already present under `dev/seqgen/Sequence/helper/`. The root `TODOs` records the intent to keep it there while M2SE remains a stub. The package copy and its re-exports are still present.
4. Keep the gradient representation change separate from the import and test fixes. `DTpy/sequence/TODO.md` records the proposed move from scalar gradients to `Vector3`.

The suite passed 21 tests after the earlier subpackage move. That result predates the current `Constant` split, waveform composition, and scanner preset changes. The current snapshot has no fresh test result.
