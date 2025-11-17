import open3d as o3d
import numpy as np
import copy

def rotate_point_cloud(pcd, axis='y', degrees=180):
    """
    Rotate point cloud around specified axis.
    
    Parameters:
    - pcd: point cloud to rotate
    - axis: 'x', 'y', or 'z'
    - degrees: rotation angle in degrees
    
    Returns:
    - rotated point cloud
    """
    pcd_rotated = copy.deepcopy(pcd)
    
    # Convert degrees to radians
    angle = np.radians(degrees)
    
    # Create rotation matrix
    if axis == 'x':
        R = pcd_rotated.get_rotation_matrix_from_axis_angle([angle, 0, 0])
    elif axis == 'y':
        R = pcd_rotated.get_rotation_matrix_from_axis_angle([0, angle, 0])
    elif axis == 'z':
        R = pcd_rotated.get_rotation_matrix_from_axis_angle([0, 0, angle])
    else:
        print(f"Unknown axis: {axis}")
        return pcd_rotated
    
    # Rotate around center
    center = pcd_rotated.get_center()
    pcd_rotated.rotate(R, center=center)
    
    return pcd_rotated

def translate_point_cloud(pcd, x=0, y=0, z=0):
    """Translate point cloud."""
    pcd_translated = copy.deepcopy(pcd)
    pcd_translated.translate([x, y, z])
    return pcd_translated

def visualize_point_clouds(point_clouds, window_name="Point Clouds"):
    """Visualize multiple point clouds with different colors."""
    colors = [
        [1, 0, 0],      # Red
        [0, 1, 0],      # Green
        [0, 0, 1],      # Blue
        [1, 1, 0],      # Yellow
        [1, 0, 1],      # Magenta
        [0, 1, 1],      # Cyan
    ]
    
    vis_clouds = []
    for i, pcd in enumerate(point_clouds):
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.paint_uniform_color(colors[i % len(colors)])
        vis_clouds.append(pcd_temp)
    
    o3d.visualization.draw_geometries(vis_clouds,
                                     window_name=window_name,
                                     width=1024, height=768)

def combine_left_right_views(left_file, right_file, 
                             left_rotation_y=0, right_rotation_y=180,
                             auto_align=True, visualize=True):
    """
    Combine left and right view PLY files.
    
    Parameters:
    - left_file: path to left view PLY
    - right_file: path to right view PLY
    - left_rotation_y: rotation for left view around Y axis (degrees)
    - right_rotation_y: rotation for right view around Y axis (degrees)
    - auto_align: automatically move point clouds together after rotation
    - visualize: show intermediate steps
    
    Returns:
    - combined point cloud
    """
    print("Loading point clouds...")
    left_pcd = o3d.io.read_point_cloud(left_file)
    right_pcd = o3d.io.read_point_cloud(right_file)
    
    print(f"Left view: {len(left_pcd.points)} points")
    print(f"Right view: {len(right_pcd.points)} points")
    
    # Visualize original
    if visualize:
        print("\nOriginal point clouds (Red=Left, Green=Right)")
        visualize_point_clouds([left_pcd, right_pcd], "Original Views")
    
    # Rotate left view
    if left_rotation_y != 0:
        print(f"\nRotating left view {left_rotation_y}° around Y axis...")
        left_pcd = rotate_point_cloud(left_pcd, axis='y', degrees=left_rotation_y)
    
    # Rotate right view
    if right_rotation_y != 0:
        print(f"Rotating right view {right_rotation_y}° around Y axis...")
        right_pcd = rotate_point_cloud(right_pcd, axis='y', degrees=right_rotation_y)
    
    # Visualize after rotation (before alignment)
    if visualize:
        print("\nAfter rotation, before alignment (Red=Left, Green=Right)")
        visualize_point_clouds([left_pcd, right_pcd], "After Rotation")
    
    # Auto-align: move point clouds close together
    if auto_align:
        print("\nAligning point clouds...")
        
        # Get centers
        left_center = left_pcd.get_center()
        right_center = right_pcd.get_center()
        
        print(f"Left center: {left_center}")
        print(f"Right center: {right_center}")
        
        # Calculate offset to bring them together
        # Move both to origin, then offset slightly
        offset_distance = -0.1  # Small offset to keep them slightly apart
        
        # Move left to origin minus offset
        left_pcd.translate(-left_center)
        left_pcd.translate([-offset_distance/2, 0, 0.12])
        
        # Move right to origin plus offset
        right_pcd.translate(-right_center)
        right_pcd.translate([offset_distance/2, 0, -0.12])
        
        # Visualize after alignment
        if visualize:
            print("\nAfter alignment (Red=Left, Green=Right)")
            visualize_point_clouds([left_pcd, right_pcd], "After Alignment")
    
    # Combine
    print("\nCombining point clouds...")
    combined = left_pcd + right_pcd
    
    # Clean up
    print("Removing duplicate points...")
    combined = combined.voxel_down_sample(voxel_size=0.002)
    
    print(f"Combined: {len(combined.points)} points")
    
    return combined

def interactive_rotation(left_file, right_file):
    """
    Interactively adjust rotation angles and alignment.
    Allows you to try different rotations until alignment looks good.
    """
    print("Loading point clouds...")
    left_pcd = o3d.io.read_point_cloud(left_file)
    right_pcd = o3d.io.read_point_cloud(right_file)
    
    while True:
        print("\n" + "="*60)
        print("Current rotation settings:")
        
        # Get rotation angles from user
        try:
            left_y = float(input("Left view Y rotation (degrees) [0]: ") or "0")
            right_y = float(input("Right view Y rotation (degrees) [180]: ") or "180")
            
            # Additional rotations if needed
            print("\nOptional additional rotations (press Enter to skip):")
            left_x = float(input("Left view X rotation (degrees) [0]: ") or "0")
            right_x = float(input("Right view X rotation (degrees) [0]: ") or "0")
            
            # Alignment option
            auto_align = input("\nAuto-align point clouds? (y/n) [y]: ").lower() or "y"
            auto_align = auto_align == 'y'
            
        except ValueError:
            print("Invalid input, using defaults")
            left_y, right_y = 0, 180
            left_x, right_x = 0, 0
            auto_align = True
        
        # Apply rotations
        left_rotated = copy.deepcopy(left_pcd)
        right_rotated = copy.deepcopy(right_pcd)
        
        if left_y != 0:
            left_rotated = rotate_point_cloud(left_rotated, 'y', left_y)
        if left_x != 0:
            left_rotated = rotate_point_cloud(left_rotated, 'x', left_x)
        if right_y != 0:
            right_rotated = rotate_point_cloud(right_rotated, 'y', right_y)
        if right_x != 0:
            right_rotated = rotate_point_cloud(right_rotated, 'x', right_x)
        
        # Auto-align if requested
        if auto_align:
            left_center = left_rotated.get_center()
            right_center = right_rotated.get_center()
            
            offset_distance = 0.05
            left_rotated.translate(-left_center)
            left_rotated.translate([-offset_distance/2, 0, 0])
            right_rotated.translate(-right_center)
            right_rotated.translate([offset_distance/2, 0, 0])
        
        # Visualize
        print("\nVisualizing... (Red=Left, Green=Right)")
        visualize_point_clouds([left_rotated, right_rotated], 
                              f"L:Y{left_y}°X{left_x}° R:Y{right_y}°X{right_x}°")
        
        # Ask if satisfied
        response = input("\nSatisfied with alignment? (y/n/q to quit): ").lower()
        if response == 'y':
            combined = left_rotated + right_rotated
            combined = combined.voxel_down_sample(voxel_size=0.002)
            return combined, (left_y, left_x, right_y, right_x)
        elif response == 'q':
            return None, None

# Example usage
if __name__ == "__main__":
    left_file = "/Users/jordan/Documents/3DME/3DME/MogeProcessed/left1.ply"
    right_file = "/Users/jordan/Documents/3DME/3DME/MogeProcessed/right1.ply"
    
    # METHOD 1: Simple - assume right view needs 180° rotation
    print("="*60)
    print("METHOD 1: Simple Combination with Auto-Alignment")
    print("="*60)
    combined = combine_left_right_views(
        left_file, 
        right_file,
        left_rotation_y=0,    # No rotation for left
        right_rotation_y=180, # 180° for right to face same direction
        auto_align=True,      # Automatically move them together
        visualize=True
    )
    
    if combined is not None:
        # Visualize final result
        print("\nFinal combined point cloud:")
        o3d.visualization.draw_geometries([combined],
                                         window_name="Combined",
                                         width=1024, height=768)
        
        # Save
        output_file = "combined_person.ply"
        o3d.io.write_point_cloud(output_file, combined)
        print(f"\n✓ Saved to {output_file}")
    
    # METHOD 2: Interactive adjustment (uncomment to use)
    # print("\n" + "="*60)
    # print("METHOD 2: Interactive Rotation Adjustment")
    # print("="*60)
    # combined, rotations = interactive_rotation(left_file, right_file)
    # 
    # if combined is not None:
    #     print(f"\nFinal rotations used: {rotations}")
    #     o3d.io.write_point_cloud("combined_person.ply", combined)
    #     print("✓ Saved to combined_person.ply")
