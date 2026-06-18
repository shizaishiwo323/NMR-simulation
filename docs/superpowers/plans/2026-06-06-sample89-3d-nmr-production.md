# Sample 89 3D NMR Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Sample 89 full-resolution 3D CPU phase-domain NMR, run 150^3 pyGIMLi tetrahedral 3D NMR with mesh exports, then compare both simulations with the experiment using the same fixed inversion alpha.

**Architecture:** Reuse the existing 3D open-pore detection and voxel CPU solver for full resolution. Add a focused 150^3 tetrahedral-mesh workflow only where the current codebase lacks true 3D pyGIMLi mesh export and solve support, then reuse the existing fixed-alpha comparison/inversion utilities.

**Tech Stack:** Python, NumPy, SciPy sparse CG, tifffile, scikit-image, pyGIMLi/TetGen when available, matplotlib, pandas.

---

### Task 1: Current-State Audit

**Files:**
- Read: `advanced_tools/three_d_nmr/true_3d_voxel_nmr.py`
- Read: `advanced_tools/three_d_nmr/open_pore_workflow.py`
- Read: `advanced_tools/compare_experimental_t2_with_sim_average.py`
- Read: `advanced_tools/png_phase_nmr_decay.py`

- [ ] Confirm the full CT TIFF path, 150^3 TIFF path, experiment CSV path, voxel size, pore/solid labels, and fixed inversion alpha.
- [ ] Inspect existing output directories for partial previous runs and do not delete them.
- [ ] Measure current CPU/GPU/RAM environment and record it in generated manifests.

### Task 2: 150^3 pyGIMLi 3D Tetra Workflow

**Files:**
- Create or modify: `advanced_tools/three_d_nmr/pygimli_3d_tetra_nmr.py`
- Test: `tests/test_pygimli_3d_tetra_nmr.py`

- [ ] Add a mesh-summary-only path that extracts open pores, builds/estimates the pore surface, and exports mesh statistics without running the PDE.
- [ ] Add a small synthetic-volume test proving the workflow exports a mesh summary and decay CSV for a tiny pore domain.
- [ ] Run the 150^3 sample workflow with `voxel_size_um=1.70`, `fixed_alpha=476.4`, and save mesh file, mesh visualization, decay CSV, inversion CSV, and run manifest.

### Task 3: Full-Resolution 3D CPU Phase-Domain Run

**Files:**
- Reuse: `advanced_tools/three_d_nmr/true_3d_voxel_nmr.py`

- [ ] Run the full Sample 89 CT with no downsampling, open-pore detection, matrix-free CPU CG, `voxel_size_um=1.70`, and fixed alpha `476.4`.
- [ ] Save open/closed pore summary, operator summary, CG iteration history, decay CSV/figure, inversion CSV/figure, and manifest.
- [ ] If the full production run is too long for one turn, leave it running only if a managed process can be monitored; otherwise produce a completed probe plus exact resumable command.

### Task 4: Unified Experiment Comparison

**Files:**
- Create or modify: `advanced_tools/compare_experiment_with_3d_simulations.py`
- Test: `tests/test_compare_experiment_with_3d_simulations.py`

- [ ] Read only the first two columns of the experimental CSV.
- [ ] Normalize experimental and simulation decays consistently.
- [ ] Re-invert experiment, full CPU 3D simulation, and 150^3 pyGIMLi simulation with fixed alpha `476.4`.
- [ ] Save comparison CSVs, a combined decay/T2 figure, and a manifest listing all input paths and parameters.

### Task 5: Verification

- [ ] Run targeted tests for changed code.
- [ ] Confirm all requested output files exist and are non-empty.
- [ ] Inspect manifests to verify voxel size, pore/solid labels, fixed alpha, model dimension, and solver method.
- [ ] Report completed outputs and any unresolved scientific/compute limitations without overstating results.
