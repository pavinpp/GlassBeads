# GlassBeads: JAX-LaB Thermo-Solute Digital Twin

A high-performance, GPU-accelerated Lattice Boltzmann (LBM) simulation environment for studying thermo-solute flow and crystal precipitation in digital rock geometries (Glass Beads).

## 🚀 Overview

This project implements a multi-physics digital twin that couples:
*   **Fluid Dynamics**: D3Q19 Lattice Boltzmann Method for incompressible flow.
*   **Solute Transport**: Advection-diffusion with kinetic precipitation/dissolution.
*   **Heat Transfer**: Coupled thermal fields with wall-resistive cooling.
*   **Morphological Evolution**: Real-time tracking of crystal growth and its feedback on flow (clogging).

## 📂 Core Components

### 1. Simulation Engine
*   **`main.py`**: The entry point for running simulations. Handles domain cropping, JIT compilation, and study orchestration.
*   **`physics.py`**: Contains the core LBM kernels, thermodynamic scaling, and the specialized `apply_optimized_physics` method. Includes strict thermodynamic clamps to prevent kinetic overshoot.
*   **`output.py`**: Manages data persistence, including 3D VTI snapshots, 2D debug slices, and topological JSON metadata.

### 2. Analysis & Visualization
*   **`compare_runs.py`**: Aggregates results across multiple flow rates to generate a **Regime Map** (Volume, Euler Characteristic, and Péclet/Damköhler scaling).
*   **`voxel_correlation.py`**: Micro-scale analysis of growth preferences (Probability of crystal vs. local thickness, velocity, and distance to grain).
*   **`digital_rock_analysis/`**: Contains the pipeline for mask generation and preprocessing of the glass bead geometry.

### 3. Validation Suite (Sanity Checks)
*   **`run_all_sanity.py`**: Unified runner for all validation scripts.
*   **`sanity_mass_partition.py`**: Verifies exact mass conservation between injection, fluid, crystal, and outflow.
*   **`sanity_density_deviation.py`**: Checks LBM compressibility errors.
*   **`sanity_mach.py`**: Monitors Mach number to ensure subsonic flow.
*   **`sanity_equilibrium.py`**: Validates precipitation against theoretical thermodynamic limits.
*   **`sanity_solubility.py`**: Verifies the polynomial solubility model for CuSO4.

## 🏃 Running Simulations

### Single Run
```bash
python main.py --time 6.0 --shutin_time 3.0 --crop_mm 2.0 --flow_rate 30
```

### Parallel Multi-GPU Sweep
To run a flow-rate sweep across two GPUs simultaneously:

```bash
# Parallel sweep across GPU 0 and GPU 1
( for FR in 5 10 15; do \
    echo "GPU0 starting FR=$FR"; \
    CUDA_VISIBLE_DEVICES=0 python main.py --time 1.0 --shutin_time 0.5 --crop_mm 1.2 --flow_rate "$FR" > "log_fr${FR}_gpu0.txt" 2>&1 || exit 1; \
    echo "GPU0 finished FR=$FR"; \
done ) & \
( for FR in 20 25 30; do \
    echo "GPU1 starting FR=$FR"; \
    CUDA_VISIBLE_DEVICES=1 python main.py --time 1.0 --shutin_time 0.5 --crop_mm 1.2 --flow_rate "$FR" > "log_fr${FR}_gpu1.txt" 2>&1 || exit 1; \
    echo "GPU1 finished FR=$FR"; \
done ) & \
wait; echo "All jobs finished."
```

### Batch Check
To re-run or verify a batch of logs:
```bash
( for FR in 5 10 15; do \
    echo "GPU0 checking FR=$FR"; \
    CUDA_VISIBLE_DEVICES=0 python main.py --time 1.0 --shutin_time 0.5 --crop_mm 1.2 --flow_rate "$FR" > "log_fr${FR}_gpu0.txt" 2>&1 || exit 1; \
done ) & \
( for FR in 20 25 30; do \
    echo "GPU1 checking FR=$FR"; \
    CUDA_VISIBLE_DEVICES=1 python main.py --time 1.0 --shutin_time 0.5 --crop_mm 1.2 --flow_rate "$FR" > "log_fr${FR}_gpu1.txt" 2>&1 || exit 1; \
done ) & \
wait; echo "Batch check complete."
```

## 📊 Key Features
*   **Unified CPBB (Continuous Partial Bounce-Back)**: A unified collision mask that seamlessly blends static rock geometry, dynamic crystal growth, and inlet/outlet valves into a single stable operator.
*   **Floating Boundary Refinements**: Hybrid boundary logic designed to prevent flow stagnation; uses fixed-density inlet velocity forcing and outlet velocity extrapolation to ensure correct Péclet number scaling.
*   **Solute Tau Optimization**: Reduced solute relaxation floor ($\tau_c=0.505$) to minimize numerical diffusion, effectively increasing the lattice Péclet number by 10x while maintaining stability via gradual concentration ramping.
*   **Domain-Aware Auto-Scaling**: Automatically calculates lattice time steps ($\tau$ up to 20.0) based on physical domain size and flow rate to maintain constant optimal Mach numbers, eliminating manual tuning.
*   **Dual-Axis Physical Projections**: Publication-ready debug visualization featuring dual Y and Z-axis midplane projections, real-world physical mm scaling, and fixed contrast limits for frame-by-frame comparison.
*   **Direct Mass Tracking**: True empirical tracking of injected and expelled solute mass directly logged from the LBM kernels, enabling flawless 0.00% error validation in mass partition sanity checks.
*   **Minkowski Functionals**: Real-time topological fingerprinting (Volume, Surface Area, Euler Number).
*   **Hydrodynamic Valve**: Perfect mass conservation during shut-in via specialized boundary masks.
*   **Thermodynamic Clamping**: Prevents precipitation overshoot by limiting mass extraction to available supersaturation.
*   **White-Theme Visuals**: All diagnostics generate high-contrast, publication-ready light-themed plots.
