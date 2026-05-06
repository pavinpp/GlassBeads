import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import re

def load_csv(path):
    data = []
    if not path.exists():
        return None
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({k: float(v) for k, v in row.items() if v is not None and k is not None})
            except ValueError:
                continue
    return data

def load_latest_topology(run_dir):
    json_files = list(run_dir.glob("topology_step_*.json"))
    if not json_files:
        return None
    json_files.sort(key=lambda x: int(re.search(r"step_(\d+)", x.name).group(1)))
    with open(json_files[-1], 'r') as f:
        return json.load(f)

def load_latest_zproj(run_dir):
    png_files = list(run_dir.glob("zproj_step_*.png"))
    if not png_files:
        return None
    png_files.sort(key=lambda x: int(re.search(r"step_(\d+)", x.name).group(1)))
    return png_files[-1]

def generate_regime_map():
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"

    # Constants per physics.py / ThermoScaling
    DX = 22e-6
    DT = 3.2267e-5
    CS = 1.0 / np.sqrt(3.0)
    L = 2.0e-3
    D = 5.0e-10
    K_GROW = 0.005
    k_phys = K_GROW / DT
    Da = k_phys * L**2 / D  # constant across the sweep

    run_dirs = [d for d in results_dir.iterdir()
                if d.is_dir() and re.match(r"output_debug_.*mm_fr\d+_\d+", d.name)]

    sweep_results = []
    for run_dir in run_dirs:
        match = re.search(r"_fr(\d+)_", run_dir.name)
        if not match:
            continue
        flow_rate = int(match.group(1))

        csv_data = load_csv(run_dir / "key_parameters.csv")
        topology = load_latest_topology(run_dir)
        zproj = load_latest_zproj(run_dir)

        if not csv_data or not topology:
            print(f"Warning: Skipping incomplete run in {run_dir.name}")
            continue

        sweep_results.append({
            "flow_rate": flow_rate,
            "csv": csv_data,
            "topology": topology,
            "zproj": zproj,
            "dir": run_dir,
        })

    if not sweep_results:
        print("No valid runs found in results/ directory.")
        return

    sweep_results.sort(key=lambda x: x['flow_rate'])

    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Thermo-Solute Crystallization Regime Map",
                 fontsize=16, y=0.98)

    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.4])
    ax_vol = fig.add_subplot(gs[0, 0])
    ax_mink = fig.add_subplot(gs[0, 1])
    ax_dim = fig.add_subplot(gs[1, 0])
    ax_thumb = fig.add_subplot(gs[2, :])

    table_data = []

    for res in sweep_results:
        fr = res['flow_rate']
        csv_data = res['csv']
        topo = res['topology']

        times = [d['Time_s'] for d in csv_data]
        vols = [d['total_precip_vol_voxels'] for d in csv_data]
        mean_machs = [d.get('Mean_Mach', d.get('Mach', 0.0)) for d in csv_data]
        dps = [abs(d.get('dP_Pa', 0.0)) for d in csv_data]

        # --- FIXED shut-in detection ----------------------------------------
        # Use dP_Pa collapse, not Mach magnitude. During shut-in dP drops by
        # >95%; Mach only drops ~20-25%. Skip i=0 (t=0 init step).
        max_dp = max(dps) if dps else 0.0
        shutin_idx = len(times)  # default: never reached
        if max_dp > 0.01:
            for i, dp in enumerate(dps):
                if i > 0 and dp < 0.05 * max_dp:
                    shutin_idx = i
                    break
        shutin_time = times[shutin_idx] if shutin_idx < len(times) else times[-1]
        # --------------------------------------------------------------------

        # Panel A: cumulative crystal volume vs time
        ax_vol.plot(times, vols, label=f"{fr} mL/hr", marker='o', markersize=4)

        # Mean Mach during flow phase (exclude i=0 init, exclude shut-in)
        flow_machs = mean_machs[1:shutin_idx] if shutin_idx > 1 else []
        flow_ma = float(np.mean(flow_machs)) if flow_machs else 0.0

        u_phys = flow_ma * CS * (DX / DT)
        Pe = u_phys * L / D
        Pe_over_Da = (u_phys / (k_phys * L)) if k_phys > 0 else 0.0

        # Mean shut-in C_avg as a third regime axis (proxy for residual solute)
        shutin_c_avg_vals = [d.get('C_avg', 0.0) for d in csv_data[shutin_idx:]] \
                            if shutin_idx < len(csv_data) else []
        mean_shutin_c_avg = float(np.mean(shutin_c_avg_vals)) if shutin_c_avg_vals else 0.0

        res['Pe'] = Pe
        res['Pe_over_Da'] = Pe_over_Da
        res['final_v'] = topo['volume_voxels_v0']
        res['final_chi'] = topo['euler_characteristic_v3']
        res['final_sv'] = topo['specific_surface_area_S_V']
        res['shutin_time'] = shutin_time

        table_data.append([fr, res['final_v'], res['final_chi'],
                           Pe, Pe_over_Da, mean_shutin_c_avg])

    # Single shut-in marker (all runs use the same simulation timing)
    median_shutin = float(np.median([r['shutin_time'] for r in sweep_results]))
    ax_vol.axvline(median_shutin, color='black', linestyle='--', alpha=0.4,
                   label=f'shut-in @ {median_shutin:.2f}s')

    ax_vol.set_title("Crystal Volume Evolution")
    ax_vol.set_xlabel("Time (s)")
    ax_vol.set_ylabel("Volume (voxels)")
    ax_vol.legend(fontsize=8)
    ax_vol.grid(True, alpha=0.3)

    # Panel B: Minkowski functionals vs flow rate
    frs = [r['flow_rate'] for r in sweep_results]
    vs = [r['final_v'] for r in sweep_results]
    chis = [r['final_chi'] for r in sweep_results]
    svs = [r['final_sv'] for r in sweep_results]

    ax_mink.set_title("Final Topology vs Flow Rate")
    ax_mink_chi = ax_mink.twinx()
    ax_mink_sv = ax_mink.twinx()
    ax_mink_sv.spines["right"].set_position(("axes", 1.18))

    p1, = ax_mink.plot(frs, vs, 'o-', color='blue', label='Volume (V)')
    p2, = ax_mink_chi.plot(frs, chis, 's-', color='darkmagenta', label='Euler ($\\chi$)')
    p3, = ax_mink_sv.plot(frs, svs, '^-', color='goldenrod', label='S/V')

    ax_mink.set_xlabel("Flow Rate (mL/hr)")
    ax_mink.set_ylabel("V (voxels)", color='blue')
    ax_mink_chi.set_ylabel("$\\chi$", color='darkmagenta')
    ax_mink_sv.set_ylabel("S/V", color='goldenrod')
    ax_mink.legend(handles=[p1, p2, p3],
                   loc='upper left', fontsize=8)
    ax_mink.grid(True, alpha=0.3)

    # Panel C: dimensionless numbers
    pes = [r['Pe'] for r in sweep_results]
    peda = [r['Pe_over_Da'] for r in sweep_results]
    da_const = [Da] * len(sweep_results)

    ax_dim.set_title(f"Regime Scaling (Da = {Da:.2e}, constant)")
    ax_dim.plot(frs, pes, 'o-', color='green', label='Pe')
    ax_dim.plot(frs, peda, 's-', color='orange', label='Pe / Da')
    ax_dim.axhline(1.0, color='black', linestyle=':', alpha=0.3,
                   label='Pe/Da = 1 (regime crossover)')
    ax_dim.set_yscale('log')
    ax_dim.set_xlabel("Flow Rate (mL/hr)")
    ax_dim.set_ylabel("Dimensionless Number")
    ax_dim.legend(loc='best', fontsize=8)
    ax_dim.grid(True, which='both', alpha=0.3)

    # Panel D: z-projection thumbnails
    ax_thumb.axis('off')
    n_runs = len(sweep_results)
    for i, res in enumerate(sweep_results):
        if res['zproj']:
            img = plt.imread(res['zproj'])
            sub_ax = fig.add_axes([0.08 + i * 0.84 / n_runs,
                                   0.05,
                                   0.84 / n_runs - 0.02,
                                   0.15])
            sub_ax.imshow(img)
            sub_ax.set_title(f"{res['flow_rate']} mL/hr",
                             fontsize=10)
            sub_ax.axis('off')

    plt.tight_layout(rect=[0, 0.22, 1, 0.95])
    plt.savefig(results_dir / "regime_map.png", dpi=150)
    print(f"\nFigure saved to {results_dir / 'regime_map.png'}")

    # Stdout summary
    print("\n" + "=" * 95)
    print(f"{'FR':<6} | {'V':<8} | {'chi':<6} | {'S/V':<8} | "
          f"{'Pe':<12} | {'Pe/Da':<12} | {'<C_avg>_shutin':<14}")
    print("-" * 95)
    for fr, v, chi, pe, peda, c_avg in table_data:
        sv = next(r['final_sv'] for r in sweep_results if r['flow_rate'] == fr)
        print(f"{fr:<6} | {v:<8.1f} | {chi:<6.0f} | {sv:<8.3f} | "
              f"{pe:<12.2e} | {peda:<12.2e} | {c_avg:<14.4f}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    generate_regime_map()
