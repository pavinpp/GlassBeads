#!/usr/bin/env python3
"""
Sanity Check: Equilibrium Precipitation Validation
Compares simulation results against thermodynamic theory to verify:
1. Mass conservation during precipitation
2. Volumetric scaling correctness
3. Equilibrium approach (dilute vs rigorous limits)
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

def get_solubility(t_celsius):
    """Quadratic solubility curve for CuSO4 in water (g/100mL)."""
    return 0.0051 * (t_celsius**2) + 0.384 * t_celsius + 23.09

def compute_theoretical_limits(sc_params, t_final_c, porosity_init, total_vol_mm3):
    """
    Compute theoretical precipitation limits matching rigorous thermodynamic balance.
    """
    c_in = sc_params['c_in_phys']
    rho_solid = sc_params.get('rho_solid_phys', 2.284) # mg/mm3 or g/cm3
    
    # Solubility at final temperature
    s_eq_final = get_solubility(t_final_c)
    
    # 1. Dilute approximation: mass diff ignoring volume displacement
    dilute_dm_g_100ml = c_in - s_eq_final
    
    # Volume metrics
    v_pore_init_mm3 = total_vol_mm3 * porosity_init
    
    # Convert concentrations to mg/mm3 (which is equivalent to g/100mL / 100)
    c_in_mg_mm3 = c_in / 100.0
    s_eq_final_mg_mm3 = s_eq_final / 100.0
    
    # Mass calculations based on the domain volume
    mass_initial_mg = v_pore_init_mm3 * c_in_mg_mm3
    mass_final_solution_mg = v_pore_init_mm3 * s_eq_final_mg_mm3
    mass_precip_mg = mass_initial_mg - mass_final_solution_mg
    
    # 2. Rigorous calculation (accounting for fluid displacement)
    # The precipitated solid displaces fluid. 
    # V_pore_final = V_pore_init - V_precip
    # m_precip = m_init - (V_pore_init - m_precip/rho_solid) * s_eq_final_mg_mm3
    # Solving for m_precip:
    rigorous_mass_precip_mg = (mass_initial_mg - v_pore_init_mm3 * s_eq_final_mg_mm3) / (1.0 - s_eq_final_mg_mm3 / rho_solid)
    
    # Convert rigorous mass back to effective delta_c (g/100mL) for comparison
    rigorous_dm_g_100ml = (rigorous_mass_precip_mg / v_pore_init_mm3) * 100.0
    
    # Translate to mean solid fraction (s) space
    v_precip_dilute_mm3 = mass_precip_mg / rho_solid
    v_precip_rigorous_mm3 = rigorous_mass_precip_mg / rho_solid
    
    s_dilute = v_precip_dilute_mm3 / v_pore_init_mm3
    s_rigorous = v_precip_rigorous_mm3 / v_pore_init_mm3
    
    return {
        'dilute_dm': dilute_dm_g_100ml,
        'rigorous_dm': rigorous_dm_g_100ml,
        'dilute_s': s_dilute,
        'rigorous_s': s_rigorous,
        'theoretical_final_porosity': porosity_init - (v_precip_rigorous_mm3 / total_vol_mm3)
    }

def load_simulation_data(output_dir):
    """Load topology JSON and key parameters CSV."""
    dir_path = Path(output_dir)
    
    # Find JSON file
    json_files = list(dir_path.glob("topology_step_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No topology JSON found in {output_dir}")
    with open(json_files[-1], 'r') as f:
        topology = json.load(f)
        
    # Load CSV
    csv_path = dir_path / "key_parameters.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"key_parameters.csv not found in {output_dir}")
    df = pd.read_csv(csv_path)
    
    return topology, df

def main():
    parser = argparse.ArgumentParser(description="Sanity check: equilibrium precipitation validation")
    parser.add_argument("output_dir", nargs="?", help="Path to simulation output directory (optional)")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"

    if args.output_dir:
        run_dirs = [Path(args.output_dir)]
    else:
        run_dirs = [d for d in results_dir.iterdir()
                    if d.is_dir() and re.match(r"output_debug_.*mm_fr\d+_\d+", d.name)]
    
    if not run_dirs:
        print("No valid runs found.")
        return

    # Scaling parameters matching digital twin logs
    sc_params = {
        'c_in_phys': 60.0,
        't_amb_phys': 20.0,
        't_hot_phys': 75.0,
        'c_ref_phys': 80.575, 
        'rho_solid_phys': 2.284
    }
    
    summary_data = []

    for run_dir in run_dirs:
        try:
            topology, df = load_simulation_data(run_dir)
        except FileNotFoundError as e:
            print(f"Skipping {run_dir.name}: {e}")
            continue

        match = re.search(r"_fr(\d+)_", run_dir.name)
        fr = int(match.group(1)) if match else 0

        initial_porosity = topology.get("initial_porosity", 0.3637)
        final_porosity = topology.get("final_porosity", 0.0)
        crystal_voxels = topology.get("volume_voxels_v0", 0)
        
        # Extract final states from the dataframe
        final_step = df.iloc[-1]
        t_f = final_step['T_avg_C']
        s_sim_actual = (initial_porosity - final_porosity) / initial_porosity

        # Domain volume approximation (from user manual calc)
        total_vol_mm3 = 7.84 

        limits = compute_theoretical_limits(sc_params, t_f, initial_porosity, total_vol_mm3)

        ratio = (s_sim_actual / limits['rigorous_s']) * 100 if limits['rigorous_s'] > 0 else 0
        summary_data.append({
            "fr": fr,
            "name": run_dir.name,
            "ratio": ratio,
            "s_sim": s_sim_actual,
            "s_rigorous": limits['rigorous_s']
        })

        print(f"\n--- {run_dir.name} ---")
        print(f"  Final Temperature (T_f):       {t_f:.2f} C")
        print(f"  Mean solid fraction (s_sim):   {s_sim_actual:.4f}")
        print(f"  Theoretical Rigorous s:        {limits['rigorous_s']:.4f}")
        print(f"  Thermodynamic Completion:      {ratio:.2f}%")

        # Individual Plotting
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['Time_s'], df['crystal_occupied_pct'] / 100.0, label='Simulated Solid Fraction', color='blue')
        ax.axhline(limits['rigorous_s'], color='red', linestyle='--', label='Theoretical Equilibrium (Rigorous)')
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Solid Fraction', fontsize=11, fontweight='bold')
        ax.set_title(f'Crystal Growth Trajectory - {run_dir.name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = run_dir / "sanity_equilibrium.png"
        plt.savefig(output_path, dpi=150, facecolor='white')
        plt.close()

    if not args.output_dir and summary_data:
        summary_data.sort(key=lambda x: x['fr'])
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        frs = [str(s['fr']) for s in summary_data]
        ratios = [s['ratio'] for s in summary_data]
        
        ax.bar(frs, ratios, color='blue', alpha=0.7)
        ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='Theoretical Limit (100%)')
        ax.set_xlabel("Flow Rate (mL/hr)")
        ax.set_ylabel("Thermodynamic Completion (%)")
        ax.set_title("Equilibrium Approach Across All Runs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "sanity_equilibrium.png", dpi=150)
        plt.close()
        print(f"\nSaved global summary: {results_dir / 'sanity_equilibrium.png'}")

if __name__ == "__main__":
    main()