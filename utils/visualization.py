import cv2
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple
import random
from config import VisualizationConfig
from .image_processing import get_image_name


def create_visualizations(img_path: str, 
                         results: Dict[str, Any], 
                         output_dir: str,
                         config: Optional[VisualizationConfig] = None) -> Dict[str, str]:
    """
    Create and save visualization images with improved error handling
    
    Args:
        img_path: Path to original image
        results: Detection results dictionary
        output_dir: Output directory for visualizations
        config: Visualization configuration
        
    Returns:
        Dictionary of saved file paths
        
    Raises:
        ValueError: If original image cannot be loaded
        OSError: If output directory cannot be created
    """
    if config is None:
        config = VisualizationConfig()
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create output directory {output_dir}: {e}")
    
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Could not load original image: {img_path}")
    
    img_name = get_image_name(img_path)
    saved_files = {}
    
    # 1. Save probability map
    probmap = results['probmap']
    probmap_vis = (probmap * 255).astype(np.uint8)
    probmap_path = os.path.join(output_dir, f"{img_name}_probmap.jpg")
    cv2.imwrite(probmap_path, probmap_vis)
    saved_files['probmap_path'] = probmap_path
    print(f"Saved probability map: {probmap_path}")
    
    # 2. Save peak map
    peak_map = results['peak_map']
    peak_vis = (peak_map * 255).astype(np.uint8)
    peak_path = os.path.join(output_dir, f"{img_name}_peaks.jpg")
    cv2.imwrite(peak_path, peak_vis)
    saved_files['peak_path'] = peak_path
    print(f"Saved peak map: {peak_path}")
    
    # 3. Create berry detection overlay
    overlay_path = None
    if len(results['peak_coords']) > 0:
        overlay = create_detection_overlay(
            original_img, 
            results['peak_coords'], 
            probmap.shape, 
            config
        )
        
        overlay_path = os.path.join(output_dir, f"{img_name}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)
        saved_files['overlay_path'] = overlay_path
        print(f"Saved berry overlay: {overlay_path}")
    
    results_path = os.path.join(output_dir, f"{img_name}_results.txt")
    save_detection_results(img_path, results, results_path)
    saved_files['results_path'] = results_path
    print(f"Saved results text: {results_path}")
    
    return saved_files


def create_detection_overlay(original_img: np.ndarray, 
                           peak_coords: np.ndarray, 
                           probmap_shape: Tuple[int, int],
                           config: VisualizationConfig) -> np.ndarray:
    """
    Create detection overlay with improved visualization
    
    Args:
        original_img: Original image
        peak_coords: Peak coordinates
        probmap_shape: Shape of probability map
        config: Visualization configuration
        
    Returns:
        Image with detection overlay
    """
    overlay = original_img.copy()
    
    scale_y = original_img.shape[0] / probmap_shape[0]
    scale_x = original_img.shape[1] / probmap_shape[1]
    
    random.seed(42)
    
    for i, (y, x) in enumerate(peak_coords):
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        
        color = (
            random.randint(50, 255), 
            random.randint(50, 255), 
            random.randint(50, 255)
        )
        
        cv2.circle(overlay, (orig_x, orig_y), config.circle_radius, color, -1)
        cv2.circle(overlay, (orig_x, orig_y), config.circle_border, 
                  config.border_color, config.border_thickness)
        
        cv2.putText(overlay, str(i + 1), (orig_x - 5, orig_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.border_color, 1)
    
    return overlay


def save_detection_results(img_path: str, 
                          results: Dict[str, Any], 
                          results_path: str) -> None:
    """
    Save detailed detection results to text file
    
    Args:
        img_path: Path to original image
        results: Detection results
        results_path: Output file path
    """
    with open(results_path, 'w') as f:
        f.write(f"DeepGrape Detection Results\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Image: {img_path}\n")
        f.write(f"Total detected berries: {len(results['peak_coords'])}\n")
        f.write(f"Probability map shape: {results['probmap'].shape}\n")
        f.write(f"\nDetection Statistics:\n")
        f.write(f"  Raw peaks detected: {results.get('raw_peak_count', 'N/A')}\n")
        f.write(f"  After edge filtering: {results.get('edge_filtered_count', 'N/A')}\n")
        f.write(f"  Final count: {results.get('final_count', len(results['peak_coords']))}\n")
        f.write(f"\nPeak Coordinates (y, x):\n")
        
        for i, (y, x) in enumerate(results['peak_coords']):
            f.write(f"  Berry {i+1:3d}: (y={y:4.0f}, x={x:4.0f})\n")
