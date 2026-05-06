import numpy as np
import os
from pathlib import Path

def crop_center(data, target_shape_voxels):
    """Perform a centered crop on a 3D array."""
    z, y, x = data.shape
    tz, ty, tx = target_shape_voxels
    
    if z < tz or y < ty or x < tx:
        print(f"⚠️  WARNING: Source mask shape {data.shape} smaller than target {target_shape_voxels}")
        print(f"    Effective crop will be ({min(z,tz)}, {min(y,ty)}, {min(x,tx)}) — NOT a cube!")
        print(f"    Max safe crop_mm for this source: {min(data.shape) * 22 / 1000:.2f} mm")
        
    start_z = max(0, (z - tz) // 2)
    start_y = max(0, (y - ty) // 2)
    start_x = max(0, (x - tx) // 2)
    
    # Ensure we don't exceed original bounds
    end_z = min(z, start_z + tz)
    end_y = min(y, start_y + ty)
    end_x = min(x, start_x + tx)
    
    print(f"Cropping indices: Z[{start_z}:{end_z}], Y[{start_y}:{end_y}], X[{start_x}:{end_x}]")
    return data[start_z:end_z, start_y:end_y, start_x:end_x]

def main():
    input_path = Path("output/geometry_mask.npy")
    if not input_path.exists():
        print(f"Error: {input_path} not found. Please run the pipeline first.")
        return

    print(f"Loading {input_path}...")
    mask = np.load(input_path)
    current_shape = mask.shape
    print(f"Current shape (Z, Y, X): {current_shape}")

    # 1. Ask for Voxel Size
    try:
        res = float(input("Enter voxel resolution (microns/voxel) [default: 22.5]: ") or 22.5)
    except ValueError:
        res = 22.5
    
    # 2. Ask for Target Size
    print("\nEnter target size in mm (e.g., 7 7 2):")
    try:
        size_input = input("Target Dimensions (A B C in mm): ").split()
        if len(size_input) == 3:
            target_mm = [float(s) for s in size_input]
        else:
            target_mm = [7.0, 7.0, 2.0]
    except ValueError:
        target_mm = [7.0, 7.0, 2.0]

    # 3. Ask for Orientation
    print("\nWhich axis matches which dimension?")
    print(f"0: Axis 0 (Size {current_shape[0]})")
    print(f"1: Axis 1 (Size {current_shape[1]})")
    print(f"2: Axis 2 (Size {current_shape[2]})")
    
    axis_mapping = []
    for i, dim in enumerate(target_mm):
        axis = int(input(f"Assign {dim}mm to Axis (0, 1, or 2): "))
        axis_mapping.append((axis, dim))
    
    # Sort mapping to match array order (Z, Y, X)
    axis_mapping.sort()
    target_shape_mm = [m[1] for m in axis_mapping]
    
    # Convert mm to voxels
    target_voxels = [int(round((mm * 1000) / res)) for mm in target_shape_mm]
    print(f"\nTarget shape in voxels: {target_voxels}")

    # Perform crop
    cropped_mask = crop_center(mask, target_voxels)
    
    # Ensure even dimensions for multi-GPU sharding (X, Y, Z must all be even)
    # This prevents JAX-LaB from auto-padding and causing shape mismatches
    z, y, x = cropped_mask.shape
    new_z = z - (z % 2)
    new_y = y - (y % 2)
    new_x = x - (x % 2)
    if (new_z, new_y, new_x) != (z, y, x):
        print(f"Trimming to even dimensions for multi-GPU: ({z}, {y}, {x}) -> ({new_z}, {new_y}, {new_x})")
        cropped_mask = cropped_mask[:new_z, :new_y, :new_x]
    
    # --- Flow Configuration ---
    print("\n--- Flow Configuration ---")
    print("Which plane is the INLET? (Fluid enters through this face)")
    print("0: YZ Plane (Flow along X-axis)")
    print("1: XZ Plane (Flow along Y-axis)")
    print("2: XY Plane (Flow along Z-axis)")
    
    try:
        plane_choice = int(input("Inlet Plane (0, 1, or 2) [default: 0]: ") or 0)
    except ValueError:
        plane_choice = 0

    # Map Plane Choice to Array Axis
    # Array is (Z, Y, X) -> indices (0, 1, 2)
    # YZ Plane -> Flow is X (Axis 2)
    # XZ Plane -> Flow is Y (Axis 1)
    # XY Plane -> Flow is Z (Axis 0)
    mapping = {
        0: {"axis": 2, "name": "YZ", "desc": "Flow along X"},
        1: {"axis": 1, "name": "XZ", "desc": "Flow along Y"},
        2: {"axis": 0, "name": "XY", "desc": "Flow along Z"}
    }
    
    flow_axis = mapping[plane_choice]["axis"]
    plane_name = mapping[plane_choice]["name"]

    print(f"\nFlow Direction along {mapping[plane_choice]['desc']}:")
    print(" 1: Positive (Left -> Right / Front -> Back / Top -> Bottom)")
    print("-1: Negative (Right -> Left / Back -> Front / Bottom -> Top)")
    try:
        flow_dir = int(input("Direction (1 or -1) [default: 1]: ") or 1)
    except ValueError:
        flow_dir = 1

    # Define Inlet/Outlet indices based on direction
    axis_size = cropped_mask.shape[flow_axis]
    if flow_dir == 1:
        inlet_idx = 0
        outlet_idx = axis_size - 1
    else:
        inlet_idx = axis_size - 1
        outlet_idx = 0

    # Save result
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    mask_path = output_dir / "geometry_mask_cropped.npy"
    np.save(mask_path, cropped_mask)
    
    # Save Metadata/Config TXT
    config_path = output_dir / "simulation_config.txt"
    with open(config_path, "w") as f:
        f.write("# JAX-LaB Simulation Configuration\n")
        f.write(f"voxel_size_microns: {res}\n")
        f.write(f"shape_zyx: {cropped_mask.shape}\n")
        f.write(f"pore_value: 0\n")
        f.write(f"solid_value: 1\n")
        f.write(f"flow_axis: {flow_axis}\n")
        f.write(f"flow_direction: {flow_dir}\n")
        f.write(f"inlet_plane: {plane_name}\n")
        f.write(f"inlet_plane_idx: {inlet_idx}\n")
        f.write(f"outlet_plane_idx: {outlet_idx}\n")
        f.write(f"porosity: {1.0 - (np.sum(cropped_mask) / cropped_mask.size):.4f}\n")

    print(f"\nSuccess! Cropped mask saved to {mask_path}")
    print(f"Configuration saved to {config_path}")
    print(f"Inlet Plane: {plane_name} (Axis {flow_axis})")
    print(f"Final shape: {cropped_mask.shape}")
    print(f"Final porosity: {1.0 - (np.sum(cropped_mask) / cropped_mask.size):.4f}")

if __name__ == "__main__":
    main()
