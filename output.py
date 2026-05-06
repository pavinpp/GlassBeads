import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
import json
import porespy as ps
from skimage import measure

def visualize_thermo_planes(u_final, c_final, t_final, dc_final, s_final, geometry_mask, output_dir, suffix="", dx_microns=22.0):
    plt.style.use('default')
    nx, ny, nz, _ = u_final.shape
    mid_z = nz // 2
    geom_slice_z = geometry_mask[:, :, mid_z].T
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 25))
    fig.suptitle(f"Thermo-Solute Debug {suffix}", fontsize=16)
    
    extent_z = [0, nx * dx_microns / 1000, 0, ny * dx_microns / 1000] # x (mm), y (mm)
    
    titles_z = ["Velocity magnitude", "Solute c (norm)", "Temperature (norm)", "Supersaturation Δc (g/100mL)", "Solid fraction φ_s (accumulated)"]
    cbar_labels = ["Velocity (Lattice Units)", "Solute (Normalized)", "Temperature (Normalized)", "Δc (g/100mL)", "Solid Fraction φ_s"]
    
    mag_z = np.sqrt(u_final[:, :, mid_z, 0]**2 + u_final[:, :, mid_z, 1]**2)
    
    vmax_u = 0.05
    vmax_dc = 10.0
    
    im_u_z = axes[0].imshow(mag_z.T, origin='lower', cmap='turbo', extent=extent_z, vmin=0, vmax=vmax_u)
    im_c_z = axes[1].imshow(c_final[:, :, mid_z, 0].T, origin='lower', cmap='viridis', extent=extent_z, vmin=0, vmax=1.0)
    im_t_z = axes[2].imshow(t_final[:, :, mid_z, 0].T, origin='lower', cmap='inferno', extent=extent_z, vmin=0, vmax=1.0)
    im_dc_z = axes[3].imshow(dc_final[:, :, mid_z, 0].T, origin='lower', cmap='plasma', extent=extent_z, vmin=-1.0, vmax=vmax_dc)
    im_s_z = axes[4].imshow(np.clip(s_final[:, :, mid_z, 0].T, 0.0, 1.0), origin='lower', cmap='cividis', extent=extent_z, vmin=0, vmax=1.0)
    
    ims_z = [im_u_z, im_c_z, im_t_z, im_dc_z, im_s_z]
    
    for i, ax in enumerate(axes):
        ax.imshow(np.where(geom_slice_z == 1, 1.0, np.nan), origin='lower', cmap='gray', alpha=0.3, extent=extent_z)
        ax.set_title(titles_z[i], fontsize=12)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        cbar_z = fig.colorbar(ims_z[i], ax=ax, fraction=0.046, pad=0.04)
        cbar_z.set_label(cbar_labels[i], fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / f"debug_summary{suffix}.png")
    plt.close()

def save_vti_snapshot(output_dir, step, u, c, t, s, rho, mask, dx_microns=22.0, dt_seconds=1.0):
    """
    Write a single VTI file at output_dir / f"snapshot_step_{step:08d}.vti"
    containing datasets: 'u' (nx,ny,nz,3), 'c' (nx,ny,nz,1), 't' (nx,ny,nz,1),
    's' (nx,ny,nz,1), 'rho' (nx,ny,nz,1), 'mask' (nx,ny,nz uint8).
    All arrays cast to float32 except mask.
    Add attributes on the grid: step, dx_microns, dt_seconds.
    """
    filename = output_dir / f"snapshot_step_{step:08d}.vti"
    nx, ny, nz, _ = u.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx_microns, dx_microns, dx_microns))
    grid.field_data["step"] = [int(step)]
    grid.field_data["dx_microns"] = [float(dx_microns)]
    grid.field_data["dt_seconds"] = [float(dt_seconds)]
    
    grid.point_data["u"] = np.asarray(u, dtype=np.float32).reshape(-1, 3, order="F")
    grid.point_data["c"] = np.asarray(c[..., 0], dtype=np.float32).flatten(order="F")
    grid.point_data["t"] = np.asarray(t[..., 0], dtype=np.float32).flatten(order="F")
    grid.point_data["s"] = np.asarray(s[..., 0], dtype=np.float32).flatten(order="F")
    grid.point_data["rho"] = np.asarray(rho[..., 0], dtype=np.float32).flatten(order="F")
    grid.point_data["mask"] = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    
    grid.save(filename)

def save_z_projection(output_dir, step, s_field, mask, dx_microns=22.0):
    """
    Project the solid fraction along the z-axis. Save as PNG.
    Filename: f"zproj_step_{step:08d}.png". Use cmap='cividis', overlay mask outline at
    alpha=0.3 in gray. Includes physical scales and consistent color limits.
    """
    plt.style.use('default')
    nx, ny, nz, _ = s_field.shape
    
    extent_z = [0, nx * dx_microns / 1000, 0, ny * dx_microns / 1000] # x (mm), y (mm)

    # Project maximum solid fraction along Z
    s_proj_z = np.max(np.clip(s_field[..., 0], 0.0, 1.0), axis=2)
    mask_proj_z = np.max(mask, axis=2) 
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Maximum Solid Fraction $\\varphi_s$ Projection @ step {step}", fontsize=16)

    im_z = ax.imshow(s_proj_z.T, origin='lower', cmap='cividis', extent=extent_z, vmin=0, vmax=1.0)
    ax.imshow(np.where(mask_proj_z.T == 1, 1.0, np.nan), origin='lower', cmap='gray', alpha=0.3, extent=extent_z)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    cbar_z = fig.colorbar(im_z, ax=ax, fraction=0.046, pad=0.04)
    cbar_z.set_label("Solid Fraction $\\varphi_s$", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f"zproj_step_{step:08d}.png")
    plt.close()

def compute_extended_diagnostics(sim, u, c, t, s, rho, mask, scaling):
    """
    Returns a dict with keys matching the CSV columns above (excluding
    Step and Time_s). All values are Python floats. fluid_mask = (mask == 0).
    """
    fluid_mask = (mask == 0)
    
    # 1. dP_Pa
    rho_in_plane = rho[1, ..., 0]
    rho_out_plane = rho[-2, ..., 0]
    inlet_mask = np.array(sim.inlet_plane_fluid_mask)
    outlet_mask = np.array(sim.outlet_plane_fluid_mask)
    
    in_count = float(max(1.0, np.sum(inlet_mask)))
    out_count = float(max(1.0, np.sum(outlet_mask)))
    
    rho_in = float(np.sum(np.where(inlet_mask, rho_in_plane, 0.0)) / in_count)
    rho_out = float(np.sum(np.where(outlet_mask, rho_out_plane, 0.0)) / out_count)
    dp_pa = float((rho_in - rho_out) * (1.0/3.0) * 1000.0 * (scaling.dx / scaling.dt)**2)
    
    # 2. R_err_pct
    r_err_pct = float(np.max(np.where(fluid_mask, np.abs(rho[..., 0] - 1.0), 0.0)) * 100)
    
    # 3. Mach
    valid_ma = np.where(fluid_mask, np.sqrt(np.sum(u**2, axis=-1)), 0.0)
    ma_max = float(np.max(valid_ma) / scaling.cs)
    
    # Bulk-averaged Mach computed only over fluid voxels (mask == 0, excluding crystals)
    active_fluid_mask = fluid_mask & (s[..., 0] < 0.1)
    if np.any(active_fluid_mask):
        ma_mean = float(np.mean(valid_ma[active_fluid_mask]) / scaling.cs)
    else:
        ma_mean = 0.0
    
    # 4. T_avg_C, T_min_C, T_max_C  (fluid-only)
    t_phys = scaling.t_amb_phys + t[..., 0] * (scaling.t_hot_phys - scaling.t_amb_phys)
    t_in_fluid = t_phys[fluid_mask]
    t_avg_c = float(np.mean(t_in_fluid))
    t_min_c = float(np.min(t_in_fluid))
    t_max_c = float(np.max(t_in_fluid))
    
    # 5. C_avg
    # FIX: Sum the mass across the ENTIRE grid (including wall bounce-back populations) 
    # but divide by fluid volume so the external mass partition script balances perfectly.
    c_avg = float(np.sum(c[..., 0]) / np.sum(fluid_mask))
    
    # 6. S_max
    s_clipped = np.clip(s[..., 0], 0.0, 1.0)
    s_max = float(np.max(s_clipped[fluid_mask]))
    
    # 7. total_precip_vol_voxels
    total_precip_vol_voxels = float(np.sum(s_clipped[fluid_mask]))
    
    # 8. mean_porosity_remaining
    total_fluid_voxel_count = float(np.sum(fluid_mask))
    mean_porosity_remaining = float(1.0 - (total_precip_vol_voxels / total_fluid_voxel_count)) if total_fluid_voxel_count > 0 else 0.0
    
    # Crystal occupancy as fraction of initial pore space
    total_fluid_voxels = float(np.sum(fluid_mask))
    crystal_occupied_pct = (total_precip_vol_voxels / total_fluid_voxels * 100) if total_fluid_voxels > 0 else 0.0

    return {
        "Mach": ma_max,
        "Mean_Mach": ma_mean,
        "R_err_pct": r_err_pct,
        "dP_Pa": dp_pa,
        "T_avg_C": t_avg_c,
        "T_min_C": t_min_c,
        "T_max_C": t_max_c,
        "C_avg": c_avg,
        "S_max": s_max,
        "total_precip_vol_voxels": total_precip_vol_voxels,
        "mean_porosity_remaining": mean_porosity_remaining,
        "crystal_occupied_pct": crystal_occupied_pct
    }

def compute_minkowski_invariants(s_final, geometry_mask, output_dir, step, threshold=0.5, initial_porosity=None, fluid_voxels_initial=None):
    """
    Computes the 3 basic Minkowski invariants (V, S, chi) for the precipitated crystal phase.
    Note: Integral mean curvature is excluded to maintain compatibility with NumPy 2.x.
    """
    try:
        # 1. Isolate the crystal phase (boolean mask)
        # Ensure we only measure crystals in the pore space, not the rock
        print(f"  [Debug] s_final shape: {s_final.shape}, geometry_mask shape: {geometry_mask.shape}")
        crystal_mask = (s_final[..., 0] >= threshold) & (geometry_mask == 0)

        # Check if there are any crystals to analyze
        n_vox = np.count_nonzero(crystal_mask)
        print(f"  [Debug] Crystal voxels: {n_vox}")

        if n_vox == 0:
            print(f"  --> No crystals detected (max s = {np.max(s_final):.4f}). Skipping Minkowski functionals.")
            return

        # 2. Compute stable metrics via PoreSpy and Scikit-Image
        # V: Total Volume (zeroth functional)
        volume = float(n_vox)

        # S: Surface Area (first functional)
        # Uses marching cubes for sub-voxel surface area estimation
        print(f"  [Debug] Computing surface area via PoreSpy...")
        mesh = ps.tools.mesh_region(crystal_mask)
        surface_area = float(ps.metrics.mesh_surface_area(mesh=mesh))

        # Euler Characteristic (third functional)
        # chi = Vertices - Edges + Faces (measures topological holes/connectivity)
        print(f"  [Debug] Computing Euler characteristic via scikit-image...")
        euler_chi = float(measure.euler_number(crystal_mask, connectivity=3))

        # Save to JSON
        metrics = {
            "step": int(step),
            "volume_voxels_v0": volume,
            "surface_area_voxels_v1": surface_area,
            "euler_characteristic_v3": euler_chi,
            "specific_surface_area_S_V": surface_area / max(1.0, volume)
        }

        if initial_porosity is not None and fluid_voxels_initial is not None:
            crystal_fraction_of_pore = volume / fluid_voxels_initial
            final_porosity = initial_porosity * (1.0 - crystal_fraction_of_pore)
            print(f"  --> Porosity: initial = {initial_porosity*100:.2f}%, "
                  f"final = {final_porosity*100:.2f}%, "
                  f"reduction = {(initial_porosity - final_porosity)*100:.3f}% "
                  f"({crystal_fraction_of_pore*100:.3f}% of pore space occupied by crystal)")
            
            # Also save to JSON
            metrics["initial_porosity"] = initial_porosity
            metrics["final_porosity"] = final_porosity
            metrics["pore_occupied_by_crystal_pct"] = crystal_fraction_of_pore * 100

        json_path = output_dir / f"topology_step_{step}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"  --> Topological fingerprint saved: V={volume:.1f}, S={surface_area:.1f}, chi={euler_chi:.1f}")

    except Exception as e:
        print(f"  --> Warning: Failed to compute Minkowski functionals: {e}")
        import traceback
        traceback.print_exc()