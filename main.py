import os

# OPTIMIZATION: Prevent JAX from greedy pre-allocation (fixes OOM on small domains)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Disables annoying TF/JAX logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import numpy as np
import jax
import jax.numpy as jnp
import argparse
import time
import csv
from datetime import datetime
from pathlib import Path

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).resolve().parent))

from src.lattice import LatticeD3Q19
from physics import ThermoScaling, ThermoGravityFlowSim
from output import visualize_thermo_planes, save_vti_snapshot, save_z_projection, compute_extended_diagnostics, compute_minkowski_invariants

def crop_center(data, target_shape_voxels):
    z, y, x = data.shape
    tz, ty, tx = target_shape_voxels
    
    if z < tz or y < ty or x < tx:
        print(f"    WARNING: Source mask shape {data.shape} smaller than target {target_shape_voxels}")
        print(f"    Effective crop will be ({min(z,tz)}, {min(y,ty)}, {min(x,tx)}) — NOT a cube!")
        print(f"    Max safe crop_mm for this source: {min(data.shape) * 22 / 1000:.2f} mm")
        
    start_z = max(0, (z - tz) // 2); start_y = max(0, (y - ty) // 2); start_x = max(0, (x - tx) // 2)
    end_z = min(z, start_z + tz); end_y = min(y, start_y + ty); end_x = min(x, start_x + tx)
    print(f"Cropping indices: Z[{start_z}:{end_z}], Y[{start_y}:{end_y}], X[{start_x}:{end_x}]")
    return data[start_z:end_z, start_y:end_y, start_x:end_x]

def run_optimized_study():
    parser = argparse.ArgumentParser(description="JAX-LaB Digital Twin: Thermo-Solute Flow")
    parser.add_argument("--time", type=float, default=4.0, help="Total physical time in seconds")
    parser.add_argument("--shutin_time", type=float, default=2.0, help="Duration of the shut-in phase (seconds) at the end of the run")
    parser.add_argument("--flow_rate", type=float, default=15.0, help="Injection flow rate in mL/hr")
    parser.add_argument("--crop_mm", type=float, default=1.2, help="Size of the central crop cube in mm")
    args = parser.parse_args()

    # 1. Scale Initialization & Domain Config
    dx_microns = 22.0
    sc = ThermoScaling(
        dx_microns=dx_microns,
        q_ml_hr=args.flow_rate,
        diameter_mm=args.crop_mm,
        delta_c_nuc_phys=2.0,
        delta_c_grow_phys=0.5,
        base_k_nuc=0.001,
        base_k_grow=0.25 #old is 0.025, increased to speed up precipitation for testing
    )

    # 2. Dynamic Shut-In Calculation
    inject_time = args.time - args.shutin_time
    if inject_time < 0:
        raise ValueError("Error: --shutin_time cannot be greater than total --time")
    
    total_steps = sc.print_summary(args.time, flow_time_phys=inject_time)
    shutin_step = int(inject_time / sc.dt)
    
    print(f"[INIT] Flow Phase: 0 to {shutin_step} steps ({inject_time:.2f}s)")
    print(f"[INIT] Shut-in Phase: {shutin_step} to {total_steps} steps ({args.shutin_time:.2f}s)\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "results" / f"output_debug_{args.crop_mm}mm_fr{int(args.flow_rate):02d}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Dynamic Crop Calculation
    mask_path = project_root / "digital_rock_analysis" / "output" / "geometry_mask.npy"
    if not mask_path.exists():
        print(f"\n[ERROR] Geometry mask not found at: {mask_path}")
        print("Please run the digital rock pipeline first to generate the mask.")
        return
    mask = np.load(mask_path)
    target_voxels = [int(round((args.crop_mm * 1000) / dx_microns)) for _ in range(3)]
    cropped_mask = crop_center(mask, target_voxels)
    z, y, x = cropped_mask.shape; new_z = z - (z % 2); new_y = y - (y % 2); new_x = x - (x % 2)
    if (new_z, new_y, new_x) != (z, y, x): cropped_mask = cropped_mask[:new_z, :new_y, :new_x]
    mask_lab = np.transpose(cropped_mask.astype(np.uint8), (2, 1, 0))
    nx, ny, nz = mask_lab.shape

    housing_mask = np.zeros_like(mask_lab)
    housing_mask[:, [0, -1], :] = 1; housing_mask[:, :, [0, -1]] = 1
    mask_lab[:, [0, -1], :] = 1; mask_lab[:, :, [0, -1]] = 1
    rad_vox = (sc.diameter / 2) / sc.dx; mid_y, mid_z = ny // 2, nz // 2
    y_g, z_g = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
    circle = ((y_g - mid_y)**2 + (z_g - mid_z)**2) <= rad_vox**2
    mask_lab[0, ~circle] = 1; mask_lab[-1, ~circle] = 1
    housing_mask[0, ~circle] = 1; housing_mask[-1, ~circle] = 1
    inlet_idx = np.column_stack(np.where((mask_lab[0] == 0) & circle))
    inlet_idx = np.insert(inlet_idx, 0, 0, axis=1)
    outlet_idx = np.column_stack(np.where((mask_lab[-1] == 0) & circle))
    outlet_idx = np.insert(outlet_idx, 0, nx-1, axis=1)

    total_voxels = mask_lab.size
    fluid_voxels_initial = int(np.sum(mask_lab == 0))
    initial_porosity = fluid_voxels_initial / total_voxels
    
    estimated_wall_time_s = (total_voxels * total_steps) / 120_000_000 + 60
    estimated_mins = estimated_wall_time_s / 60.0
    
    print(f"[INIT] Domain shape: {mask_lab.shape} | Total voxels: {total_voxels}")
    print(f"[INIT] Initial fluid voxels: {fluid_voxels_initial}")
    print(f"[INIT] Initial porosity (geometric): {initial_porosity*100:.2f}%")
    print(f"======================================================================")
    print(f" ESTIMATED COMPUTE TIME: ~{estimated_mins:.1f} minutes")
    print(f"======================================================================\n")

    sim = ThermoGravityFlowSim(geometry_mask=mask_lab, housing_mask=housing_mask, inlet_idx=inlet_idx, outlet_idx=outlet_idx, scaling=sc,
                               lattice=LatticeD3Q19("f32/f32"), nx=nx, ny=ny, nz=nz, omega=1.0/sc.tau, precision="f32/f32", print_info_rate=0)

    print(f"JIT Compiling Debug Run → {output_dir.name}\nInlet porosity: {sim.inlet_porosity:.4f}")
    t0 = time.time(); f_current = sim.assign_fields_sharded()
    g_current = sim.initialize_passive_fields(init_val=0.0)
    h_current = sim.initialize_passive_fields(init_val=sim.scaling.t_amb)
    s_current = sim.distributed_array_init(f_current.shape[:-1] + (1,), sim.precisionPolicy.output_dtype, init_val=0.0)
    t_current = 0; ma_n = 0.0; stable = True; cum_mass_in = cum_mass_out = cum_vol_in = cum_vol_out = jnp.float32(0.0)
    # Define snapshot steps: 3 between (0, shutin) and 3 between (shutin, total)
    # plus the fixed points: 0, shutin, and total_steps.
    raw_steps = np.unique(np.concatenate([
        np.linspace(0, shutin_step, 5),
        np.linspace(shutin_step, total_steps, 5)
    ]).astype(int)).tolist()
    
    snapshot_steps = [0]
    for s in raw_steps[1:]:
        if s - snapshot_steps[-1] >= 2500 or s == total_steps:
            if s == total_steps and s - snapshot_steps[-1] < 2500 and len(snapshot_steps) > 1:
                snapshot_steps[-1] = s
            else:
                snapshot_steps.append(s)
                
    n_snaps_total = len(snapshot_steps)

    csv_path = output_dir / "key_parameters.csv"
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Step", "Time_s", "Mach", "Mean_Mach", "R_err_pct", "dP_Pa", "T_avg_C", "T_min_C", "T_max_C", "C_avg", "S_max", "total_precip_vol_voxels", "mean_porosity_remaining", "crystal_occupied_pct", "Cum_Mass_In_mg", "Cum_Mass_Out_mg"])
        
        for i, t_next in enumerate(snapshot_steps):
            if not stable: break
            
            # Sub-stepping loop to handle transitions and snapshot intervals
            while t_current < t_next:
                # Determine the next target step (either the snapshot step or the shut-in transition)
                target = t_next
                is_shutin = (t_current >= shutin_step)
                
                if t_current < shutin_step < target:
                    target = shutin_step
                
                t_current, f_current, g_current, h_current, s_current, ma_n, stable, cum_mass_in, cum_mass_out, cum_vol_in, cum_vol_out = sim.run_peak_performance(
                    t_current, target, f_current, g_current, h_current, s_current, cum_mass_in, cum_mass_out, cum_vol_in, cum_vol_out, shutin_step, is_shutin)
                
                if t_current == shutin_step and shutin_step < t_next:
                    print(f"\n>>> TRANSITION TO SHUT-IN at step {t_current}")

            # Now t_current == t_next, take the snapshot
            print(f"--- Saving Snapshot {i+1}/{n_snaps_total} at step {t_current} (t={t_current*sc.dt:.3f}s) ---")
            
            # Cast data back to host for post-processing
            f_host = sim.precisionPolicy.cast_to_compute(f_current)
            rho_n, u_n = sim.update_macroscopic(f_host)
            c_n = jnp.sum(sim.precisionPolicy.cast_to_compute(g_current), axis=-1, keepdims=True)
            t_n = jnp.sum(sim.precisionPolicy.cast_to_compute(h_current), axis=-1, keepdims=True)
            
            u_arr = np.array(u_n)
            c_arr = np.array(c_n)
            t_arr = np.array(t_n)
            s_arr = np.array(s_current)
            rho_arr = np.array(rho_n)
            
            t_phys = sc.t_amb_phys + t_arr * (sc.t_hot_phys - sc.t_amb_phys)
            c_phys = c_arr * sc.c_ref_phys
            s_t = sc.get_solubility(t_phys)
            
            visualize_thermo_planes(u_arr, c_arr, t_arr, c_phys - s_t, s_arr, mask_lab, output_dir, f"_step_{int(t_current)}_shutin_{1 if t_current >= shutin_step else 0}")
            
            save_vti_snapshot(output_dir, int(t_current), u_arr, c_arr, t_arr, s_arr, rho_arr, mask_lab, dx_microns=22.0, dt_seconds=sc.dt)
            save_z_projection(output_dir, int(t_current), s_arr, mask_lab)
            
            diags = compute_extended_diagnostics(sim, u_arr, c_arr, t_arr, s_arr, rho_arr, mask_lab, sc)
            csv_writer.writerow([
                int(t_current), float(t_current * sc.dt), diags["Mach"], diags["Mean_Mach"],
                diags["R_err_pct"], diags["dP_Pa"], diags["T_avg_C"],
                diags["T_min_C"], diags["T_max_C"],
                diags["C_avg"], diags["S_max"], diags["total_precip_vol_voxels"],
                diags["mean_porosity_remaining"], diags["crystal_occupied_pct"],
                float(cum_mass_in * sc.mass_per_voxel_mg),
                float(cum_mass_out * sc.mass_per_voxel_mg)
            ])
            csv_file.flush()

    # Final Morphological Validation (Minkowski Functionals)
    print("\n>>> COMPUTING FINAL MORPHOLOGICAL FINGERPRINT (MINKOWSKI FUNCTIONALS)...")
    compute_minkowski_invariants(s_arr, mask_lab, output_dir, int(t_current), initial_porosity=initial_porosity, fluid_voxels_initial=fluid_voxels_initial)
    
    print(f"\nComplete — wall time {time.time() - t0:.1f}s | output: {output_dir.name}")

if __name__ == "__main__":
    run_optimized_study()