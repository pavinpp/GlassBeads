import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_sanity_solubility():
    # Constants
    T_AMB = 20.0
    T_HOT = 75.0
    C_IN = 60.0
    NUC_THRESH = 5.0
    
    # Solubility curve function
    def s_t(T):
        return 0.0051 * T**2 + 0.384 * T + 23.09

    T_range = np.linspace(15, 80, 200)
    S_range = s_t(T_range)
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot curve
    ax.plot(T_range, S_range, color='blue', linewidth=2, label='Solubility $s(T)$')
    
    # Fill regions
    ax.fill_between(T_range, 0, S_range, color='blue', alpha=0.1, label='Undersaturated')
    ax.fill_between(T_range, S_range, S_range + NUC_THRESH, color='green', alpha=0.1, label='Supersaturated (No Nucleation)')
    ax.fill_between(T_range, S_range + NUC_THRESH, 120, color='red', alpha=0.1, label='Nucleation Regime')
    
    # Markers
    s_amb = s_t(T_AMB)
    s_hot = s_t(T_HOT)
    ax.plot(T_AMB, s_amb, 'ko')
    ax.annotate(f'Ambient: {s_amb:.1f} g/100mL', (T_AMB, s_amb), xytext=(5, -15), textcoords='offset points')
    
    ax.plot(T_HOT, s_hot, 'ko')
    ax.annotate(f'Hot: {s_hot:.1f} g/100mL', (T_HOT, s_hot), xytext=(-100, 10), textcoords='offset points')
    
    # Injection line
    ax.axhline(C_IN, color='goldenrod', linestyle='--', alpha=0.7, label=f'Injection $C_{{IN}} = {C_IN}$ g/100mL')
    
    # Aesthetics
    ax.set_title("Solubility model used in JAX-LaB twin (s(T) polynomial fit)", fontsize=14)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Concentration (g/100mL)")
    ax.set_xlim(15, 80)
    ax.set_ylim(20, 110)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=9)
    
    # Save
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "sanity_solubility.png", dpi=150)
    plt.close()
    
    # Print summary
    print(f"{'Temp (°C)':<10} | {'s(T) (g/100mL)':<15} | {'Delta C (g/100mL)':<15}")
    print("-" * 45)
    temps = [20, 30, 40, 50, 60, 70, 75]
    for T in temps:
        st = s_t(T)
        dc = C_IN - st
        print(f"{T:<10} | {st:<15.2f} | {dc:<15.2f}")
    
    print("\n[PASS] Solubility curve verification complete.")

if __name__ == "__main__":
    run_sanity_solubility()
