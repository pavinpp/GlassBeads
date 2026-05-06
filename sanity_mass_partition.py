import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from pathlib import Path

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

def run_sanity_mass_balance():
    # Constants
    DX = 22e-6
    DT = 3.2267e-5
    L = 2.0e-3
    C_IN = 60.0
    VOXEL_MASS_MG = 6.389e-6
    DOMAIN_VOL_ML = (L * 100)**3  # 0.008 mL
    VOX_VOL_ML = (DX * 100)**3
    
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    
    run_dirs = [d for d in results_dir.iterdir()
                if d.is_dir() and re.match(r"output_debug_.*mm_fr\d+_\d+", d.name)]
    
    sweep_results = []
    for run_dir in run_dirs:
        match = re.search(r"_fr(\d+)_", run_dir.name)
        if not match: continue
        fr = int(match.group(1))
        csv_data = load_csv(run_dir / "key_parameters.csv")
        if not csv_data: continue
        sweep_results.append({"fr": fr, "csv": csv_data, "name": run_dir.name})
    
    if not sweep_results:
        print("No valid runs found.")
        return
        
    sweep_results.sort(key=lambda x: x['fr'])
    
    plt.style.use('default')
    # 5 subplots for flow rates + 1 for summary yield
    n_runs = len(sweep_results)
    fig = plt.figure(figsize=(12, 4 * (n_runs + 1)))
    gs = fig.add_gridspec(n_runs + 1, 1)
    
    summary_data = []
    
    for idx, res in enumerate(sweep_results):
        fr = res['fr']
        csv_data = res['csv']
        
        times = np.array([d['Time_s'] for d in csv_data])
        precip_vols = np.array([d['total_precip_vol_voxels'] for d in csv_data])
        c_avgs = np.array([d['C_avg'] for d in csv_data])
        porosities = np.array([d['mean_porosity_remaining'] for d in csv_data])
        
        # Estimate initial fluid volume
        valid_mask = (precip_vols > 1.0) & (porosities < 0.9999)
        if np.any(valid_mask):
            total_fluid_vox = np.median(precip_vols[valid_mask] / (1.0 - porosities[valid_mask]))
        else:
            total_fluid_vox = (DOMAIN_VOL_ML * 0.392) / VOX_VOL_ML
            
        initial_fluid_vol_ml = total_fluid_vox * VOX_VOL_ML
        
        # Extract direct simulation tracking data if available, else fallback to 0
        if 'Cum_Mass_In_mg' in csv_data[0] and 'Cum_Mass_Out_mg' in csv_data[0]:
            m_in = np.array([d['Cum_Mass_In_mg'] for d in csv_data])
            m_out_actual = np.array([d['Cum_Mass_Out_mg'] for d in csv_data])
        else:
            print(f"Warning: Missing direct mass tracking columns in {res['name']}.")
            mass_rate_mg_s = (fr / 3600) * C_IN * 10
            m_in = mass_rate_mg_s * np.minimum(times, 3.0) # Fallback
            m_out_actual = np.zeros_like(times)

        m_crystal = precip_vols * VOXEL_MASS_MG
        # Actual fluid mass calculation
        m_fluid = c_avgs * (C_IN / 100.0) * (porosities * initial_fluid_vol_ml) * 1000.0

        # Calculate Mass Balance Error (Residual)
        m_total_tracked = m_fluid + m_crystal + m_out_actual
        m_error = m_in - m_total_tracked
        m_error_pct = np.abs(np.divide(m_error, m_in, out=np.zeros_like(m_error), where=m_in > 1e-4)) * 100
        
        yield_pct = (m_crystal[-1] / m_in[-1] * 100) if m_in[-1] > 0 else 0
        final_err_pct = m_error_pct[-1]
        
        ax = fig.add_subplot(gs[idx])
        ax.plot(times, m_in, 'k--', label='Injected (Total)')
        ax.plot(times, m_fluid, 'blue', label='In Fluid')
        ax.plot(times, m_crystal, 'darkmagenta', label='In Crystal')
        ax.plot(times, m_out_actual, 'goldenrod', label='Outflow (Measured)')
        ax.plot(times, m_error, 'red', linestyle=':', label='Mass Balance Error')
        
        ax.set_title(f"FR = {fr} mL/hr — Solute partition\n(Final Error: {final_err_pct:.4f}%)")
        ax.set_ylabel("Mass (mg)")
        ax.legend(loc='upper left', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        if idx == n_runs - 1:
            ax.set_xlabel("Time (s)")
            
        # Individual plot saving
        run_dir = results_dir / res['name']
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        ax_ind.plot(times, m_in, 'k--', label='Injected (Total)')
        ax_ind.plot(times, m_fluid, 'blue', label='In Fluid')
        ax_ind.plot(times, m_crystal, 'darkmagenta', label='In Crystal')
        ax_ind.plot(times, m_out_actual, 'goldenrod', label='Outflow (Measured)')
        ax_ind.plot(times, m_error, 'red', linestyle=':', label='Mass Balance Error')
        ax_ind.set_xlabel("Time (s)")
        ax_ind.set_ylabel("Mass (mg)")
        ax_ind.set_title(f"Solute Partition - {res['name']}\n(Final Error: {final_err_pct:.4f}%)")
        ax_ind.legend()
        ax_ind.grid(True, alpha=0.3)
        plt.savefig(run_dir / "sanity_mass_partition.png", dpi=150)
        plt.close(fig_ind)

        summary_data.append({
            "fr": fr,
            "total_in": m_in[-1],
            "final_fluid": m_fluid[-1],
            "final_crystal": m_crystal[-1],
            "final_out": m_out_actual[-1],
            "yield": yield_pct,
            "max_error_pct": np.max(m_error_pct)
        })

    # Group by FR for the bar chart with error bars
    from collections import defaultdict
    grouped = defaultdict(list)
    for s in summary_data:
        grouped[s['fr']].append(s['yield'])
    
    sorted_frs = sorted(grouped.keys())
    means = [np.mean(grouped[f]) for f in sorted_frs]
    stds = [np.std(grouped[f]) for f in sorted_frs]
    
    # Add Yield Bar Chart
    ax_yield = fig.add_subplot(gs[n_runs])
    x_pos = np.arange(len(sorted_frs))
    ax_yield.bar(x_pos, means, yerr=stds, capsize=5, color='darkmagenta', alpha=0.7, label='Mean Yield')
    ax_yield.set_xticks(x_pos)
    ax_yield.set_xticklabels([str(f) for f in sorted_frs])
    ax_yield.set_title("Crystallization Yield vs Flow Rate (with std dev)")
    ax_yield.set_xlabel("Flow Rate (mL/hr)")
    ax_yield.set_ylabel("Yield (%)")
    ax_yield.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "sanity_mass_partition.png", dpi=150)
    plt.close()
    
    print(f"{'FR':<6} | {'Total Injected (mg)':<20} | {'Final in Fluid (mg)':<20} | {'Final in Crystal (mg)':<22} | {'Final Outflow (mg)':<20} | {'Yield (%)':<10}")
    print("-" * 115)
    for row in summary_data:
        print(f"{row['fr']:<6} | {row['total_in']:<20.4f} | {row['final_fluid']:<20.4f} | {row['final_crystal']:<22.4f} | {row['final_out']:<20.4f} | {row['yield']:<10.2f}")
    
    print("\n[INFO] Solute mass partition analysis complete.")

if __name__ == "__main__":
    run_sanity_mass_balance()
