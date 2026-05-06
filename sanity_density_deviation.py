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

def run_sanity_density():
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("LBM Compressibility Verification (Density Deviation)", fontsize=16)
    
    summary_data = []
    
    for res in sweep_results:
        fr = res['fr']
        csv_data = res['csv']
        times = [d['Time_s'] for d in csv_data]
        r_err = [d['R_err_pct'] for d in csv_data]
        dps = [abs(d.get('dP_Pa', 0.0)) for d in csv_data]
        
        # Shut-in detection (as in compare_runs.py)
        max_dp = max(dps) if dps else 0.0
        shutin_idx = len(times)
        if max_dp > 0.01:
            for i, dp in enumerate(dps):
                if i > 0 and dp < 0.05 * max_dp:
                    shutin_idx = i
                    break
        
        flow_err = r_err[1:shutin_idx] if shutin_idx > 1 else []
        shutin_err = r_err[shutin_idx:] if shutin_idx < len(r_err) else []
        
        mean_flow = np.mean(flow_err) if flow_err else 0.0
        mean_shutin = np.mean(shutin_err) if shutin_err else 0.0
        max_err = np.max(r_err)
        
        verdict = "PASS" if max_err < 3.0 else ("MARGINAL" if max_err < 5.0 else "FAIL")
        summary_data.append([fr, max_err, mean_flow, mean_shutin, verdict])
        
        ax1.plot(times, r_err, label=f"{fr} mL/hr")
        
        # Individual plot saving
        run_dir = results_dir / res['name']
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        ax_ind.plot(times, r_err, color='blue', label=f'R_err (Max {max_err:.2f}%)')
        ax_ind.axhline(5.0, color='red', linestyle='--', alpha=0.5, label='Stability (5%)')
        ax_ind.axhline(3.0, color='orange', linestyle='--', alpha=0.5, label='Target (3%)')
        ax_ind.set_xlabel("Time (s)")
        ax_ind.set_ylabel("Max Density Deviation (%)")
        ax_ind.set_title(f"Density Deviation - {res['name']}")
        ax_ind.legend()
        ax_ind.grid(True, alpha=0.3)
        plt.savefig(run_dir / "sanity_density_deviation.png", dpi=150)
        plt.close(fig_ind)

    ax1.axhline(5.0, color='red', linestyle='--', alpha=0.5, label='Stability (5%)')
    ax1.axhline(3.0, color='orange', linestyle='--', alpha=0.5, label='Target (3%)')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Max Density Deviation (%)")
    ax1.set_title("Density Deviation vs Time")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    frs = [r[0] for r in summary_data]
    max_errs = [r[1] for r in summary_data]
    colors = ['green' if e < 3 else ('goldenrod' if e < 5 else 'red') for e in max_errs]
    
    ax2.bar([str(f) for f in frs], max_errs, color=colors, alpha=0.7)
    ax2.axhline(5.0, color='red', linestyle='--', alpha=0.3)
    ax2.axhline(3.0, color='orange', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Flow Rate (mL/hr)")
    ax2.set_ylabel("Max Deviation (%)")
    ax2.set_title("Max Deviation by Flow Rate")
    
    plt.tight_layout()
    plt.savefig(results_dir / "sanity_density_deviation.png", dpi=150)
    plt.close()
    
    print(f"{'FR':<6} | {'Max R_err (%)':<15} | {'Flow Mean (%)':<15} | {'Shut-in Mean (%)':<18} | {'Verdict':<10}")
    print("-" * 75)
    for row in summary_data:
        print(f"{row[0]:<6} | {row[1]:<15.4f} | {row[2]:<15.4f} | {row[3]:<18.4f} | {row[4]:<10}")
    
    overall_pass = all(r[4] != "FAIL" for r in summary_data)
    print(f"\n[{'PASS' if overall_pass else 'FAIL'}] Density deviation check complete.")

if __name__ == "__main__":
    run_sanity_density()
