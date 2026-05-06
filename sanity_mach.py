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

def run_sanity_mach():
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
    fig.suptitle("Subsonic LBM Verification (Mach Number)", fontsize=16)
    
    summary_data = []
    
    for res in sweep_results:
        fr = res['fr']
        csv_data = res['csv']
        times = [d['Time_s'] for d in csv_data]
        # Use Mach or Mean_Mach
        machs = [d.get('Mean_Mach', d.get('Mach', 0.0)) for d in csv_data]
        dps = [abs(d.get('dP_Pa', 0.0)) for d in csv_data]
        
        # Shut-in detection
        max_dp = max(dps) if dps else 0.0
        shutin_idx = len(times)
        if max_dp > 0.01:
            for i, dp in enumerate(dps):
                if i > 0 and dp < 0.05 * max_dp:
                    shutin_idx = i
                    break
        
        flow_machs = machs[1:shutin_idx] if shutin_idx > 1 else []
        shutin_machs = machs[shutin_idx:] if shutin_idx < len(machs) else []
        
        mean_flow = np.mean(flow_machs) if flow_machs else 0.0
        mean_shutin = np.mean(shutin_machs) if shutin_machs else 0.0
        max_ma = np.max(machs)
        
        summary_data.append([fr, max_ma, mean_flow, mean_shutin])
        
        ax1.plot(times, machs, label=f"{fr} mL/hr")

        # Individual plot saving
        run_dir = results_dir / res['name']
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        ax_ind.plot(times, machs, color='blue', label=f'Mach (Max {max_ma:.4f})')
        ax_ind.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Incompressibility (0.1)')
        ax_ind.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='Conservative (0.05)')
        ax_ind.set_xlabel("Time (s)")
        ax_ind.set_ylabel("Mach Number (Ma)")
        ax_ind.set_title(f"Mach Number - {res['name']}")
        ax_ind.legend()
        ax_ind.grid(True, alpha=0.3)
        plt.savefig(run_dir / "sanity_mach.png", dpi=150)
        plt.close(fig_ind)

    ax1.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Incompressibility (0.1)')
    ax1.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='Conservative (0.05)')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mach Number (Ma)")
    ax1.set_title("Mach Number vs Time")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    frs = np.array([r[0] for r in summary_data])
    max_mas = np.array([r[1] for r in summary_data])
    
    ax2.scatter(frs, max_mas, color='blue', s=50, label='Data')
    
    # Linear fit
    if len(frs) > 1:
        slope, intercept = np.polyfit(frs, max_mas, 1)
        fit_line = slope * frs + intercept
        r_squared = 1 - (np.sum((max_mas - fit_line)**2) / np.sum((max_mas - np.mean(max_mas))**2))
        ax2.plot(frs, fit_line, 'k--', alpha=0.5, label=f'Fit ($R^2={r_squared:.4f}$)')
        ax2.annotate(f'Slope: {slope:.2e}', xy=(0.05, 0.9), xycoords='axes fraction', color='black')
    else:
        slope, r_squared = 0.0, 0.0

    ax2.set_xlabel("Flow Rate (mL/hr)")
    ax2.set_ylabel("Max Mach (Ma)")
    ax2.set_title("Max Mach vs Flow Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "sanity_mach.png", dpi=150)
    plt.close()
    
    print(f"{'FR':<6} | {'Max Ma':<10} | {'Flow Mean Ma':<15} | {'Shut-in Mean Ma':<18}")
    print("-" * 60)
    for row in summary_data:
        print(f"{row[0]:<6} | {row[1]:<10.4f} | {row[2]:<15.4f} | {row[3]:<18.4f}")
    
    print(f"\nSlope: {slope:.2e}, R^2: {r_squared:.4f}")
    overall_pass = all(r[1] < 0.1 for r in summary_data)
    print(f"\n[{'PASS' if overall_pass else 'FAIL'}] Mach number check complete.")

if __name__ == "__main__":
    run_sanity_mach()
