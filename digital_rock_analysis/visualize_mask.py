import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_mask():
    input_path = Path("output/geometry_mask_cropped.npy")
    if not input_path.exists():
        print(f"Error: {input_path} not found. Please run crop_mask.py first.")
        return

    print(f"Loading {input_path}...")
    mask = np.load(input_path)
    nz, ny, nx = mask.shape
    print(f"Mask shape: {mask.shape}")

    # Determine middle indices
    mid_z = nz // 2
    mid_y = ny // 2
    mid_x = nx // 2

    # Create figure with 3 panes
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='#0d0d0d')
    fig.suptitle(f"Cropped Geometry Mask: {nz} x {ny} x {nx}", color='white', fontsize=14)

    # Visualization settings: 0 (Fluid) -> White, 1 (Solid) -> Black using gray_r
    cmap = 'gray_r'
    
    # XY Plane (Top view)
    axes[0].imshow(mask[mid_z, :, :], cmap=cmap, origin='lower')
    axes[0].set_title(f"XY Plane (Z={mid_z})", color='white')
    axes[0].set_xlabel("X", color='white')
    axes[0].set_ylabel("Y", color='white')

    # XZ Plane (Front view)
    axes[1].imshow(mask[:, mid_y, :], cmap=cmap, origin='lower')
    axes[1].set_title(f"XZ Plane (Y={mid_y})", color='white')
    axes[1].set_xlabel("X", color='white')
    axes[1].set_ylabel("Z", color='white')

    # YZ Plane (Side view)
    axes[2].imshow(mask[:, :, mid_x], cmap=cmap, origin='lower')
    axes[2].set_title(f"YZ Plane (X={mid_x})", color='white')
    axes[2].set_xlabel("Y", color='white')
    axes[2].set_ylabel("Z", color='white')

    # Add Legend/Annotation
    fig.text(0.5, 0.05, "Annotation:  ⬜ White = Pore (Fluid)  |  ⬛ Black = Solid (Beads)", 
             color='white', ha='center', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='#2a2a2a', edgecolor='#4fc3f7', boxstyle='round,pad=0.5'))

    for ax in axes:
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    plt.subplots_adjust(bottom=0.15)
    
    # Save preview
    output_img = Path("output/cropped_mask_preview.png")
    plt.savefig(output_img, facecolor=fig.get_facecolor())
    print(f"Preview saved to {output_img}")
    
    plt.show()

if __name__ == "__main__":
    visualize_mask()
