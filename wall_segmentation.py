import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os

def apply_wallpaper(wall_path, wallpaper_path, output_path, wall_width_ft=10, wall_height_ft=8, 
                   wallpaper_width_ft=2.0, wallpaper_height_ft=2.0):
    """
    Apply wallpaper pattern to wall image using SAM segmentation with proper dimensional scaling
    
    Args:
        wall_path: Path to the wall image
        wallpaper_path: Path to the wallpaper pattern image  
        output_path: Path where the result will be saved
        wall_width_ft: Real-world width of the wall in feet (user input)
        wall_height_ft: Real-world height of the wall in feet (user input)
        wallpaper_width_ft: Real-world width of wallpaper tile (fixed at 1.0 ft)
        wallpaper_height_ft: Real-world height of wallpaper tile (fixed at 1.0 ft)
    """
    try:
        # Check if model file exists
        checkpoint_path = "models/sam_vit_b.pth"
        if not os.path.exists(checkpoint_path):
            # Try alternative paths
            alt_paths = [
                "sam_vit_b.pth",
                "../models/sam_vit_b.pth",
                "./models/sam_vit_b_01ec64.pth"  # Full filename
            ]
            
            checkpoint_found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    checkpoint_found = True
                    break
            
            if not checkpoint_found:
                print(f"Warning: SAM model not found. Using basic image blending instead.")
                return apply_wallpaper_basic(wall_path, wallpaper_path, output_path, 
                                           wall_width_ft, wall_height_ft, 
                                           wallpaper_width_ft, wallpaper_height_ft)
        
        model_type = "vit_b"
        
        # Load SAM model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        predictor = SamPredictor(sam)
        
        # Load and preprocess wall image
        image = cv2.imread(wall_path)
        if image is None:
            raise ValueError(f"Cannot read wall image from {wall_path}")
        
        # Resize if too large for performance
        MAX_RES = 1024
        original_shape = image.shape
        scale_factor = 1.0
        if max(image.shape) > MAX_RES:
            scale_factor = MAX_RES / max(image.shape)
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        
        height, width, _ = image.shape
        
        # Smart wall detection: find large uniform areas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (31, 31), 0)
        
        # Find the brightest region (often wall areas)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
        
        # Use multiple points for better segmentation
        input_points = np.array([
            [max_loc[0], max_loc[1]],  # Brightest point
            [width//2, height//2],     # Center point
            [width//4, height//4],     # Quarter points
            [3*width//4, 3*height//4]
        ])
        input_labels = np.array([1, 1, 1, 1])  # All positive points
        
        # Generate mask
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Choose the best mask based on score
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        # Create properly tiled wallpaper based on real dimensions
        tiled_wallpaper = create_properly_tiled_wallpaper(
            wallpaper_path, width, height, 
            wall_width_ft, wall_height_ft,
            wallpaper_width_ft, wallpaper_height_ft
        )
        
        # Apply wallpaper to masked areas
        result = image.copy()
        
        # Create a smooth transition using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.GaussianBlur(mask_smooth.astype(np.float32), (5, 5), 0)
        
        # Apply wallpaper with smooth blending
        for c in range(3):  # For each color channel
            result[:, :, c] = (
                image[:, :, c] * (1 - mask_smooth) + 
                tiled_wallpaper[:, :, c] * mask_smooth
            ).astype(np.uint8)
        
        # Resize back to original if it was scaled down
        if max(original_shape) > MAX_RES:
            result = cv2.resize(result, (original_shape[1], original_shape[0]))
        
        # Save result
        success = cv2.imwrite(output_path, result)
        if not success:
            raise ValueError(f"Failed to save result image to {output_path}")
        
        print(f"Wallpaper applied successfully! Result saved to {output_path}")
        print(f"Wall dimensions: {wall_width_ft}ft x {wall_height_ft}ft")
        print(f"Wallpaper tile size: {wallpaper_width_ft}ft x {wallpaper_height_ft}ft")
        
    except Exception as e:
        print(f"Error in SAM-based wallpaper application: {str(e)}")
        print("Falling back to basic image blending...")
        return apply_wallpaper_basic(wall_path, wallpaper_path, output_path,
                                   wall_width_ft, wall_height_ft,
                                   wallpaper_width_ft, wallpaper_height_ft)

def create_properly_tiled_wallpaper(wallpaper_path, target_width_px, target_height_px,
                                  wall_width_ft, wall_height_ft, 
                                  wallpaper_width_ft, wallpaper_height_ft):
    """
    Create properly scaled and tiled wallpaper based on real-world dimensions
    
    Args:
        wallpaper_path: Path to the 1ft x 1ft wallpaper tile
        target_width_px: Target width in pixels (image resolution)
        target_height_px: Target height in pixels (image resolution)
        wall_width_ft: Real wall width in feet
        wall_height_ft: Real wall height in feet
        wallpaper_width_ft: Wallpaper tile width in feet (1.0)
        wallpaper_height_ft: Wallpaper tile height in feet (1.0)
    """
    try:
        # Load the wallpaper tile
        wallpaper = cv2.imread(wallpaper_path)
        if wallpaper is None:
            raise ValueError(f"Cannot read wallpaper from {wallpaper_path}")
        
        # Calculate how many tiles we need based on real dimensions
        tiles_x = int(wall_width_ft / wallpaper_width_ft)  # e.g., 10ft / 1ft = 10 tiles
        tiles_y = int(wall_height_ft / wallpaper_height_ft)  # e.g., 8ft / 1ft = 8 tiles
        
        # Handle fractional tiles by adding one more if needed
        if wall_width_ft % wallpaper_width_ft > 0:
            tiles_x += 1
        if wall_height_ft % wallpaper_height_ft > 0:
            tiles_y += 1
        
        print(f"Creating tiled wallpaper: {tiles_x} x {tiles_y} tiles")
        
        # Calculate the size each tile should be in pixels
        tile_width_px = target_width_px // tiles_x
        tile_height_px = target_height_px // tiles_y
        
        # Resize the wallpaper tile to the calculated pixel size
        wallpaper_resized = cv2.resize(wallpaper, (tile_width_px, tile_height_px))
        
        # Create the tiled pattern
        # First, create a row of tiles
        row = wallpaper_resized
        for i in range(1, tiles_x):
            row = np.hstack([row, wallpaper_resized])
        
        # Then stack rows to create the full pattern
        tiled_wallpaper = row
        for i in range(1, tiles_y):
            tiled_wallpaper = np.vstack([tiled_wallpaper, row])
        
        # Crop or pad to exact target dimensions if needed
        current_height, current_width = tiled_wallpaper.shape[:2]
        
        if current_width > target_width_px or current_height > target_height_px:
            # Crop if larger
            tiled_wallpaper = tiled_wallpaper[:target_height_px, :target_width_px]
        elif current_width < target_width_px or current_height < target_height_px:
            # Pad if smaller
            pad_width = max(0, target_width_px - current_width)
            pad_height = max(0, target_height_px - current_height)
            
            # Create padding by repeating edge pixels
            if pad_width > 0:
                right_pad = np.repeat(tiled_wallpaper[:, -1:], pad_width, axis=1)
                tiled_wallpaper = np.hstack([tiled_wallpaper, right_pad])
            
            if pad_height > 0:
                bottom_pad = np.repeat(tiled_wallpaper[-1:, :], pad_height, axis=0)
                tiled_wallpaper = np.vstack([tiled_wallpaper, bottom_pad])
        
        return tiled_wallpaper
        
    except Exception as e:
        print(f"Error creating properly tiled wallpaper: {str(e)}")
        # Fallback to simple resize
        wallpaper = cv2.imread(wallpaper_path)
        return cv2.resize(wallpaper, (target_width_px, target_height_px))

def apply_wallpaper_basic(wall_path, wallpaper_path, output_path,
                         wall_width_ft=10, wall_height_ft=8, 
                         wallpaper_width_ft=2.0, wallpaper_height_ft=2.0):
    """
    Fallback method: Basic wallpaper application with proper tiling
    """
    try:
        # Load images
        wall_img = cv2.imread(wall_path)
        wallpaper_img = cv2.imread(wallpaper_path)
        
        if wall_img is None:
            raise ValueError(f"Cannot read wall image from {wall_path}")
        if wallpaper_img is None:
            raise ValueError(f"Cannot read wallpaper image from {wallpaper_path}")
        
        height, width = wall_img.shape[:2]
        
        # Create properly tiled wallpaper
        tiled_wallpaper = create_properly_tiled_wallpaper(
            wallpaper_path, width, height,
            wall_width_ft, wall_height_ft,
            wallpaper_width_ft, wallpaper_height_ft
        )
        
        # Create a simple mask based on brightness and color uniformity
        gray = cv2.cvtColor(wall_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Create mask for uniform areas (likely walls)
        # Find areas with low gradient (uniform regions)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create mask for low-gradient areas (walls)
        _, mask = cv2.threshold(gradient_normalized, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to mask for smooth blending
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (15, 15), 0)
        
        # Blend images
        result = wall_img.copy()
        for c in range(3):
            result[:, :, c] = (
                wall_img[:, :, c] * (1 - mask_blurred) + 
                tiled_wallpaper[:, :, c] * mask_blurred
            ).astype(np.uint8)
        
        # Save result
        success = cv2.imwrite(output_path, result)
        if not success:
            raise ValueError(f"Failed to save result image to {output_path}")
        
        print(f"Basic wallpaper application completed! Result saved to {output_path}")
        print(f"Applied {wall_width_ft}ft x {wall_height_ft}ft wall with 1ft x 1ft tiles")
        
    except Exception as e:
        print(f"Error in basic wallpaper application: {str(e)}")
        # Final fallback: simple overlay
        return apply_simple_overlay(wall_path, wallpaper_path, output_path, 
                                  wall_width_ft, wall_height_ft,
                                  wallpaper_width_ft, wallpaper_height_ft)

def apply_simple_overlay(wall_path, wallpaper_path, output_path, 
                        wall_width_ft=10, wall_height_ft=8,
                        wallpaper_width_ft=2.0, wallpaper_height_ft=2.0, alpha=0.7):
    """
    Simple overlay method as final fallback with proper tiling
    """
    try:
        wall_img = cv2.imread(wall_path)
        wallpaper_img = cv2.imread(wallpaper_path)
        
        if wall_img is None or wallpaper_img is None:
            raise ValueError("Cannot read one or both input images")
        
        height, width = wall_img.shape[:2]
        
        # Create properly tiled wallpaper
        tiled_wallpaper = create_properly_tiled_wallpaper(
            wallpaper_path, width, height,
            wall_width_ft, wall_height_ft,
            wallpaper_width_ft, wallpaper_height_ft
        )
        
        # Simple alpha blending
        result = cv2.addWeighted(wall_img, 1-alpha, tiled_wallpaper, alpha, 0)
        
        success = cv2.imwrite(output_path, result)
        if not success:
            raise ValueError(f"Failed to save result image to {output_path}")
        
        print(f"Simple overlay completed! Result saved to {output_path}")
        
    except Exception as e:
        raise ValueError(f"All wallpaper application methods failed: {str(e)}")

# Legacy function for backward compatibility
def create_tiled_wallpaper(wallpaper_path, target_width, target_height, output_path):
    """
    Legacy function - creates a simple tiled wallpaper pattern
    """
    try:
        wallpaper = cv2.imread(wallpaper_path)
        if wallpaper is None:
            raise ValueError(f"Cannot read wallpaper from {wallpaper_path}")
        
        wp_height, wp_width = wallpaper.shape[:2]
        
        # Calculate how many tiles we need
        tiles_x = int(np.ceil(target_width / wp_width))
        tiles_y = int(np.ceil(target_height / wp_height))
        
        # Create tiled image
        tiled = np.tile(wallpaper, (tiles_y, tiles_x, 1))
        
        # Crop to exact target size
        tiled_cropped = tiled[:target_height, :target_width]
        
        cv2.imwrite(output_path, tiled_cropped)
        return output_path
        
    except Exception as e:
        print(f"Error creating tiled wallpaper: {str(e)}")
        return wallpaper_path  # Return original if tiling fails