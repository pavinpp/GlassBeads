# Pore-Scale Simulation of Cooling-Induced Nucleation and Crystallization in Porous Media using JAX-LaB

## 🚀 Overview
Mineral precipitation and crystallization can significantly reduce permeability and cause clogging in porous media. This phenomenon is highly relevant to Carbon Capture and Storage (CCS), geothermal energy systems, filtration processes, and subsurface flow applications. This repository contains a high-performance, GPU-accelerated digital twin framework for studying these processes at the pore scale.

The system models the injection of a hot concentrated CuSO₄ solution into a glass-bead porous medium, followed by cooling-induced supersaturation, nucleation, crystal growth, and progressive pore clogging.

## ❓ Research Question
**“Where do crystals preferentially form, and why are these locations clogging-sensitive?”**

## 🔬 Experimental and Digital Foundation
This project is based on microfluidic experiments using a planar flow cell packed with soda-lime glass beads.
- **Glass bead size:** 250 to 500 μm
- **Chip size:** Approximately 7 mm x 7 mm x 2 mm
- **Chemistry:** Hot concentrated CuSO₄ solution injection followed by shut-in cooling.
- **Digital Geometry:** Micro-CT based digital geometry.
- **Digital Domain Size:** 323 x 91 x 336 voxels.

## 💻 Computational Framework
The simulations are built using **JAX-LaB**, a differentiable and GPU-accelerated Lattice Boltzmann library.
- **Method:** D3Q19 Lattice Boltzmann Method (LBM).
- **Physics:** Coupled flow, heat transport, solute transport, supersaturation, nucleation, crystal growth, and evolving solid fraction.
- **Performance:** GPU acceleration through JAX and XLA.
- **Hardware:** Optimized for multi-GPU execution (e.g., 2x NVIDIA RTX A5500).

## 🌡️ Physical Model
- **Cooling:** Temperature drops from approximately 75°C toward ambient temperature.
- **Thermodynamics:** Temperature-dependent CuSO₄ solubility determines local supersaturation.
- **Nucleation:** Primary nucleation occurs in unseeded voxels exceeding the metastable limit.
- **Growth:** Kinetic crystal growth in seeded voxels.
- **Morphological Feedback:** Continuous Partial Bounce-Back (CPBB) operator handles transport-morphology feedback, capturing the progressive narrowing of pore throats and eventual clogging.

## 🔄 Simulation Protocol
The workflow consists of two distinct phases:
1. **Injection Phase:** Hot concentrated solution is forced into the pore network to establish the initial solute and temperature distribution.
2. **Shut-in Phase:** Boundaries are closed (hydrodynamic valves), bulk advection stops, cooling continues, and crystallization develops as the primary driver of morphology change.

## ✅ Validation and Sanity Checks
The framework includes a rigorous validation suite to ensure physical correctness:
- **Density Stability:** Monitoring LBM compressibility errors.
- **Mass Partitioning:** Verifying exact solute mass conservation across fluid, crystal, and outflow.
- **Thermodynamic Potential:** Confirming precipitation does not exceed theoretical supersaturation limits.
- **Solubility Validation:** Verifying the polynomial solubility model against experimental data.

## 📈 Key Results
- **Spatial Footprint:** Increasing the flow rate expands the spatial footprint of crystallization.
- **Flow Rate Effects:** Low flow rates result in localized precipitation near the inlet, while higher rates lead to more widespread precipitation after shut-in.
- **Shut-in Trigger:** Crystal volume rises sharply after shut-in, identifying cooling during shut-in as a major precipitation trigger.
- **Preferential Locations:**
    - Crystal probability is highest in **narrow pore throats**.
    - Peaks in **moderate-velocity zones**, where solute supply and residence time are optimally balanced.
    - Crystals preferentially form **near grain surfaces**, especially at higher flow rates.
- **Clogging Sensitivity:** Narrow pores, moderate-velocity zones, and near-surface regions are the most clogging-sensitive locations.

## 🏁 Conclusions
- Crystallization is **transport-controlled**, not random.
- Preferential nucleation occurs in narrow pores, moderate-velocity regions, and near grain surfaces.
- The transport-morphology feedback successfully captures progressive pore clogging.
- This project demonstrates a digital-twin style framework for studying complex crystallization in porous media with high fidelity.

## 🚀 Future Work
- Testing a wider range of flow rates, shut-in times, and cooling profiles.
- Computing voxel-level Péclet and Damköhler numbers to map regime transitions.
- Quantifying the link between pore-scale clogging and macro-scale permeability loss/pressure response.
- Extending results toward predictive macro-scale models.

## 📜 Major Credits and Acknowledgements
This project is deeply built upon and inspired by the experimental and computational foundations developed by **Krasnoff & Kelly** and by **Pradhan, Gentine & Kelly**. The microfluidic glass-bead crystallization system, experimental motivation, and cooling-induced nucleation behavior are credited to Krasnoff and Kelly. The computational framework, differentiable LBM infrastructure, and JAX-LaB foundation are credited to Pradhan, Gentine, and Kelly.

### References
*   **Krasnoff, R., & Kelly, S. (2024).** *Microfluidic investigation of cooling-induced nucleation and crystallization behavior in porous media.*
*   **Pradhan, P., Gentine, P., & Kelly, S. (2026).** *JAX-LaB: A high-performance, differentiable Lattice Boltzmann library for modeling multiphase fluid dynamics in geosciences and engineering.* Journal of Advances in Modeling Earth Systems, 18, e2025MS005313. [https://doi.org/10.1029/2025MS005313](https://doi.org/10.1029/2025MS005313)

### Acknowledgements
*   **Professor Shaina Kelly** (Columbia University)
*   **Piyush Paritosh Pradhan**
*   **Rosalie Krasnoff**
*   **PoreStore Lab**, Columbia University

---

## 📂 Repository Usage

### Core Components
*   **`main.py`**: Entry point for simulations; handles domain cropping, JIT compilation, and orchestration.
*   **`physics.py`**: Core LBM kernels, thermodynamic scaling, and the specialized `apply_optimized_physics` method.
*   **`output.py`**: Manages data persistence (3D VTI, 2D slices, JSON metadata).
*   **`compare_runs.py`**: Aggregates results across multiple flow rates to generate regime maps.
*   **`voxel_correlation.py`**: Micro-scale analysis of growth preferences.
*   **`run_all_sanity.py`**: Unified runner for the validation suite.

### Running Simulations

#### Single Run
```bash
python main.py --time 6.0 --shutin_time 3.0 --crop_mm 2.0 --flow_rate 30
```

#### Parallel Multi-GPU Sweep
To run a flow-rate sweep across two GPUs simultaneously:
```bash
# Parallel sweep across GPU 0 and GPU 1
( for FR in 5 10 15; do \
    echo "GPU0 starting FR=$FR"; \
    CUDA_VISIBLE_DEVICES=0 python main.py --time 1.0 --shutin_time 0.5 --crop_mm 1.2 --flow_rate "$FR" > "log_fr${FR}_gpu0.txt" 2>&1 || exit 1; \
done ) & \
( for FR in 20 25 30; do \
    echo "GPU1 starting FR=$FR"; \
    CUDA_VISIBLE_DEVICES=1 python main.py --time 1.0 --shutin_time 0.5 --crop_mm 1.2 --flow_rate "$FR" > "log_fr${FR}_gpu1.txt" 2>&1 || exit 1; \
done ) & \
wait; echo "All jobs finished."
```

### Key Technical Features
*   **Unified CPBB (Continuous Partial Bounce-Back):** Seamlessly blends rock geometry, dynamic crystal growth, and inlet/outlet valves.
*   **Solute Tau Optimization:** Reduced solute relaxation floor ($\tau_c=0.505$) for high Péclet number stability.
*   **Domain-Aware Auto-Scaling:** Automatically calculates lattice time steps to maintain optimal Mach numbers.
*   **Minkowski Functionals:** Real-time topological fingerprinting (Volume, Surface Area, Euler Number).
*   **Thermodynamic Clamping:** Prevents precipitation overshoot via mass-extraction limits.
