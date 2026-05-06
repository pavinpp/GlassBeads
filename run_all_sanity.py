#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    print(f"\n>>> Running {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    scripts = [
        "sanity_density_deviation.py",
        "sanity_mach.py",
        "sanity_mass_partition.py",
        "sanity_equilibrium.py",
        "sanity_solubility.py"
    ]
    
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
            
    print(f"\n========================================")
    print(f"Sanity Check Summary: {success_count}/{len(scripts)} passed")
    print(f"Global results in: results/")
    print(f"Individual results in: results/output_debug_*/")
    print(f"========================================\n")

if __name__ == "__main__":
    main()
