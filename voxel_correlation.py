import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import porespy as ps
import scipy.ndimage
from pathlib import Path
import re

# Constants
DX = 22e-6

def conditional_probability(scalar_field, crystal_mask, fluid_mask, n_bins=20, log_x=False):
    fluid_vals = scalar_field[fluid_mask]
    crystal_vals = scalar_field[crystal_mask]
    if fluid_vals.size == 0 or crystal_vals.size == 0:
        return None, None, None
        
    if log_x:
        positive = fluid_vals[fluid_vals > 0]
        if positive.size == 0:
            return None, None, None
        bins = np.logspace(np.log10(positive.min()), np.log10(fluid_vals.max()), n_bins + 1)
    else:
        bins = np.linspace(fluid_vals.min(), fluid_vals.max(), n_bins + 1)
        
    fluid_hist, _ = np.histogram(fluid_vals, bins=bins)
    crystal_hist, _ = np.histogram(crystal_vals, bins=bins)
    
    p_crystal = np.divide(crystal_hist.astype(float), fluid_hist,
                          out=np.zeros(fluid_hist.shape, dtype=float),
                          where=fluid_hist > 0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, p_crystal, fluid_hist

def run_analysis():
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    
    run_dirs = [d for d in results_dir.iterdir()
                if d.is_dir() and re.match(r"output_debug_.*mm_fr\d+_\d+", d.name)]
    
    sweep_results = []
    
    for run_dir in run_dirs:
        match = re.search(r"_fr(\d+)_", run_dir.name)
        if not match: continue
        fr = int(match.group(1))
        
        # Find latest VTI
        vti_files = list(run_dir.glob("snapshot_step_*.vti"))
        if not vti_files:
            continue
        vti_files.sort(key=lambda x: int(re.search(r"step_(\d+)", x.name).group(1)))
        latest_vti = vti_files[-1]
        
        sweep_results.append({
            "fr": fr,
            "vti_path": latest_vti,
            "name": run_dir.name
        })
        
    sweep_results.sort(key=lambda x: x['fr'])
    
    # Store aggregated data for sweep figure
    sweep_data = []
    
    plt.style.use('default')
    
    for res in sweep_results:
        fr = res['fr']
        vti_path = res['vti_path']
        
        print(f"\nProcessing FR = {fr} mL/hr from {vti_path.name}...")
        try:
            grid = pv.read(vti_path)
            nx, ny, nz = grid.dimensions
            
            u_flat = grid.point_data["u"]
            s_flat = grid.point_data["s"]
            mask_flat = grid.point_data["mask"]
            
            # Recover 3D shapes. VTI was saved with order='F'
            u = u_flat.reshape((nx, ny, nz, 3), order='F')
            s = s_flat.reshape((nx, ny, nz), order='F')
            mask = mask_flat.reshape((nx, ny, nz), order='F')
            
            fluid_mask = (mask == 0)
            crystal_mask = (s >= 0.5) & fluid_mask
            
            n_fluid = fluid_mask.sum()
            n_crystal = crystal_mask.sum()
            crystal_frac = (n_crystal / n_fluid * 100) if n_fluid > 0 else 0
            
            if n_crystal < 50:
                print(f"  [Warning] Crystal mask sum = {n_crystal} (< 50). Statistics may be unreliable.")
                
            velocity_mag = np.linalg.norm(u, axis=-1)
            
            print(f"  Computing local thickness for FR {fr} (this might take a moment)...")
            local_thickness = ps.filters.local_thickness(fluid_mask)
            
            print(f"  Computing distance to grain for FR {fr}...")
            dist_to_grain = scipy.ndimage.distance_transform_edt(fluid_mask)
            
            mean_lt_fluid = local_thickness[fluid_mask].mean() if n_fluid > 0 else 0
            mean_lt_cryst = local_thickness[crystal_mask].mean() if n_crystal > 0 else 0
            
            mean_v_fluid = velocity_mag[fluid_mask].mean() if n_fluid > 0 else 0
            mean_v_cryst = velocity_mag[crystal_mask].mean() if n_crystal > 0 else 0
            
            mean_d_fluid = dist_to_grain[fluid_mask].mean() if n_fluid > 0 else 0
            mean_d_cryst = dist_to_grain[crystal_mask].mean() if n_crystal > 0 else 0
            
            print(f"FR = {fr} mL/hr")
            print(f"  Total fluid voxels:        {n_fluid}")
            print(f"  Total crystal voxels:      {n_crystal}")
            print(f"  Crystal fraction:          {crystal_frac:.4f}%")
            print(f"  Mean local thickness in crystals: {mean_lt_cryst:.2f} vs {mean_lt_fluid:.2f} (fluid) voxels")
            print(f"  Mean velocity in crystals:        {mean_v_cryst:.4e} vs {mean_v_fluid:.4e} (fluid) lattice u")
            print(f"  Mean dist-to-grain in crystals:   {mean_d_cryst:.2f} vs {mean_d_fluid:.2f} (fluid) voxels")
            
            sweep_data.append({
                "fr": fr,
                "n_fluid": n_fluid,
                "n_crystal": n_crystal,
                "crystal_frac": crystal_frac,
                "lt_ratio": (mean_lt_cryst / mean_lt_fluid) if mean_lt_fluid > 0 else 0,
                "v_ratio": (mean_v_cryst / mean_v_fluid) if mean_v_fluid > 0 else 0,
                "d_ratio": (mean_d_cryst / mean_d_fluid) if mean_d_fluid > 0 else 0
            })
            
            # Conditional Probabilities
            lt_bins, lt_prob, lt_hist = conditional_probability(local_thickness, crystal_mask, fluid_mask, n_bins=20, log_x=False)
            v_bins, v_prob, v_hist = conditional_probability(velocity_mag, crystal_mask, fluid_mask, n_bins=20, log_x=True)
            d_bins, d_prob, d_hist = conditional_probability(dist_to_grain, crystal_mask, fluid_mask, n_bins=20, log_x=False)
            
            sweep_data[-1].update({
                "lt_bins": lt_bins, "lt_prob": lt_prob,
                "v_bins": v_bins, "v_prob": v_prob,
                "d_bins": d_bins, "d_prob": d_prob
            })
            
            # 2D Histogram
            valid_2d = fluid_mask & (velocity_mag > 0)
            lt_vals = local_thickness[valid_2d]
            v_vals = np.log10(velocity_mag[valid_2d])
            
            crystal_valid = crystal_mask & valid_2d
            lt_vals_cryst = local_thickness[crystal_valid]
            v_vals_cryst = np.log10(velocity_mag[crystal_valid])
            
            lt_bins_2d = np.linspace(lt_vals.min(), lt_vals.max(), 20)
            v_bins_2d = np.linspace(v_vals.min(), v_vals.max(), 20)
            
            fluid_hist_2d, _, _ = np.histogram2d(lt_vals, v_vals, bins=[lt_bins_2d, v_bins_2d])
            crystal_hist_2d, _, _ = np.histogram2d(lt_vals_cryst, v_vals_cryst, bins=[lt_bins_2d, v_bins_2d])
            
            p_crystal_2d = np.divide(crystal_hist_2d.astype(float), fluid_hist_2d,
                                     out=np.zeros_like(fluid_hist_2d, dtype=float),
                                     where=fluid_hist_2d >= 5)
            p_crystal_2d[fluid_hist_2d < 5] = np.nan
            
            # Plotting per-run
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"FR = {fr} mL/hr — Voxel-level precipitation correlations", fontsize=16)
            
            def plot_dual(ax, bin_centers, prob, hist, xlabel, title, log_x=False):
                if bin_centers is None: return
                ax2 = ax.twinx()
                ax2.fill_between(bin_centers, hist, step="mid", color='gray', alpha=0.2)
                ax2.set_ylabel("Fluid voxel count", color='gray')
                
                ax.plot(bin_centers, prob, 'o-', color='blue')
                ax.set_xlabel(xlabel)
                ax.set_ylabel("P(crystal)", color='blue')
                if log_x:
                    ax.set_xscale('log')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
            plot_dual(axs[0, 0], lt_bins, lt_prob, lt_hist, "Local thickness (voxels)", "P(crystal | local_thickness)", log_x=False)
            plot_dual(axs[0, 1], v_bins, v_prob, v_hist, "Velocity magnitude (lattice u)", "P(crystal | velocity_mag)", log_x=True)
            plot_dual(axs[1, 0], d_bins, d_prob, d_hist, "Distance to grain (voxels)", "P(crystal | dist_to_grain)", log_x=False)
            
            # Panel D: 2D Heatmap
            ax2d = axs[1, 1]
            im = ax2d.imshow(p_crystal_2d.T, origin='lower', cmap='magma',
                             extent=[lt_bins_2d.min(), lt_bins_2d.max(), v_bins_2d.min(), v_bins_2d.max()],
                             aspect='auto')
            ax2d.set_xlabel("Local thickness (voxels)")
            ax2d.set_ylabel("Velocity magnitude (lattice u, log10)")
            ax2d.set_title("P(crystal | thickness, velocity)")
            cbar = fig.colorbar(im, ax=ax2d)
            cbar.set_label("P(crystal)")
            
            plt.tight_layout()
            plt.savefig(results_dir / f"voxel_corr_fr{fr:02d}.png", dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"  [Error] Failed to process FR = {fr}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Sweep figure
    if sweep_data:
        print("\nGenerating sweep summary figure...")
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Voxel-level precipitation correlations across flow-rate sweep", fontsize=16)
        
        min_fr = min(d['fr'] for d in sweep_data)
        max_fr = max(d['fr'] for d in sweep_data)
        
        import matplotlib.cm as cm
        cmap = cm.viridis
        
        for d in sweep_data:
            fr = d['fr']
            norm_val = (fr - min_fr) / max(1, (max_fr - min_fr))
            c = cmap(norm_val)
            
            if d.get('lt_bins') is not None:
                axs[0].plot(d['lt_bins'], d['lt_prob'], '-', color=c, label=f"{fr} mL/hr")
            if d.get('v_bins') is not None:
                axs[1].plot(d['v_bins'], d['v_prob'], '-', color=c, label=f"{fr} mL/hr")
            if d.get('d_bins') is not None:
                axs[2].plot(d['d_bins'], d['d_prob'], '-', color=c, label=f"{fr} mL/hr")
                
        axs[0].set_xlabel("Local thickness (voxels)")
        axs[0].set_ylabel("P(crystal)")
        axs[0].set_title("Thickness dependence")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=8)
        
        axs[1].set_xlabel("Velocity magnitude (lattice u)")
        axs[1].set_ylabel("P(crystal)")
        axs[1].set_title("Velocity dependence")
        axs[1].set_xscale('log')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=8)

        axs[2].set_xlabel("Distance to grain (voxels)")
        axs[2].set_ylabel("P(crystal)")
        axs[2].set_title("Distance dependence")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(results_dir / "voxel_corr_sweep.png", dpi=150)
        plt.close()
        
        print("\nSweep-comparison summary:")
        print(f"{'FR':<6} | {'Crystal frac (%)':<18} | {'<thickness>_xtl/<thickness>_fluid':<35} | {'<|u|>_xtl/<|u|>_fluid':<25} | {'<dist>_xtl/<dist>_fluid':<25}")
        print("-" * 115)
        for d in sweep_data:
            print(f"{d['fr']:<6} | {d['crystal_frac']:<18.4f} | {d['lt_ratio']:<35.4f} | {d['v_ratio']:<25.4f} | {d['d_ratio']:<25.4f}")
            
    print("\n[INFO] Voxel correlation analysis complete.")

if __name__ == "__main__":
    run_analysis()