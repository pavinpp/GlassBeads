import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_and_visualize():
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent
    mask_path = project_root / "digital_rock_analysis" / "output" / "geometry_mask_cropped.npy"
    output_dir = project_root / "digital_rock_analysis" / "output"
    
    if not mask_path.exists():
        print(f"Error: {mask_path} not found.")
        return
    
    # Load mask
    mask = np.load(mask_path).astype(np.uint8)
    
    # Transpose to match JAX-LaB (nx, ny, nz)
    mask_lab = np.transpose(mask, (2, 1, 0))
    nx, ny, nz = mask_lab.shape
    
    # Circular mask parameters (exactly as in flow simulation)
    dx_mm = 22.5e-3 # 22.5 um in mm
    diameter_mm = 1.2
    radius_voxels = (diameter_mm / 2.0) / dx_mm
    
    mid_y, mid_z = ny // 2, nz // 2
    y_grid, z_grid = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
    circle_mask = ((y_grid - mid_y)**2 + (z_grid - mid_z)**2) <= radius_voxels**2
    
    # Count fluid voxels
    inlet_plane = mask_lab[0, :, :]
    outlet_plane = mask_lab[-1, :, :]
    
    inlet_fluid = (inlet_plane == 0) & circle_mask
    outlet_fluid = (outlet_plane == 0) & circle_mask
    
    # Prepare Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0d0d0d')
    fig.suptitle(f"Circular Boundary Analysis (1.2mm Diameter)", color='white', fontsize=16)
    
    # Visualization Settings
    cmap = 'gray_r'
    
    # 1. Inlet Visualization
    # Everything outside the circle is solid (Black). Inside, fluid is White.
    inlet_viz = np.ones_like(inlet_plane) # Start all solid
    inlet_viz[inlet_fluid] = 0 # Fluid paths are white
    
    axes[0].imshow(inlet_viz.T, cmap=cmap, origin='lower')
    axes[0].set_title(f"Inlet (X=0)\nFlow Area: {np.sum(inlet_fluid) * dx_mm**2:.4f} mm^2", color='white')
    axes[0].axis('off')
    
    # 2. Outlet Visualization
    outlet_viz = np.ones_like(outlet_plane) # Start all solid
    outlet_viz[outlet_fluid] = 0 # Fluid paths are white
    
    axes[1].imshow(outlet_viz.T, cmap=cmap, origin='lower')
    axes[1].set_title(f"Outlet (X={nx-1})\nFlow Area: {np.sum(outlet_fluid) * dx_mm**2:.4f} mm^2", color='white')
    axes[1].axis('off')
    
    # Add legend
    fig.text(0.5, 0.05, "Annotation:  ⬜ White = Open Flow Path  |  ⬛ Black = Blocked by Bead/Wall", 
             color='white', ha='center', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='#2a2a2a', edgecolor='#4fc3f7', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    output_path = output_dir / "boundary_blockage.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    print(f"\nBoundary visualization saved to: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    calculate_and_visualize()
