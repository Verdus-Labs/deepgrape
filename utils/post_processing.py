import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from config import InferenceConfig


def non_maximum_suppression_efficient(peak_coords: np.ndarray, min_distance: int) -> np.ndarray:
    """
    Efficient Non-Maximum Suppression using vectorized operations
    
    Args:
        peak_coords: Array of peak coordinates (N, 2)
        min_distance: Minimum distance between peaks
        
    Returns:
        Filtered peak coordinates
    """
    if len(peak_coords) == 0:
        return peak_coords
    
    coords = peak_coords.astype(float)
    
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    too_close = (distances < min_distance) & (distances > 0)
    
    to_remove = set()
    
    for i in range(len(coords)):
        if i in to_remove:
            continue
        close_indices = np.where(too_close[i])[0]
        to_remove.update(close_indices)
    
    keep_indices = [i for i in range(len(coords)) if i not in to_remove]
    return peak_coords[keep_indices]


def filter_edge_detections(peak_coords: np.ndarray, 
                          image_shape: Tuple[int, int], 
                          edge_margin: int) -> np.ndarray:
    """
    Filter out detections near image edges
    
    Args:
        peak_coords: Array of peak coordinates (N, 2) in (y, x) format
        image_shape: Shape of the image (height, width)
        edge_margin: Margin from edges in pixels
        
    Returns:
        Filtered peak coordinates
    """
    if len(peak_coords) == 0:
        return peak_coords
        
    h, w = image_shape
    
    valid_mask = (
        (peak_coords[:, 0] >= edge_margin) &  # y >= margin
        (peak_coords[:, 0] < h - edge_margin) &  # y < h - margin
        (peak_coords[:, 1] >= edge_margin) &  # x >= margin
        (peak_coords[:, 1] < w - edge_margin)    # x < w - margin
    )
    
    return peak_coords[valid_mask]


def detect_peaks(output: torch.Tensor, config: InferenceConfig) -> Dict[str, Any]:
    """
    Detect peaks in probability map with configurable parameters
    
    Args:
        output: Model output tensor (1, 1, H, W)
        config: Inference configuration
        
    Returns:
        Dictionary containing detection results
    """
    # Smooth output (avg pool)
    avg_pooled = F.avg_pool2d(
        output[0], 
        kernel_size=config.avg_pool_kernel, 
        stride=1, 
        padding=config.avg_pool_kernel // 2
    )
    
    max_pooled = F.max_pool2d(
        avg_pooled, 
        kernel_size=config.max_pool_kernel, 
        stride=1, 
        padding=config.max_pool_kernel // 2
    )
    peak_map = torch.where(avg_pooled == max_pooled, avg_pooled, torch.zeros_like(avg_pooled))
    
    peak_binary = (peak_map >= config.val_score).to(torch.uint8)
    peak_coords_raw = torch.nonzero(peak_binary[0]).cpu().numpy()  # (y, x) coordinates
    
    h, w = peak_binary[0].shape
    filtered_coords = filter_edge_detections(peak_coords_raw, (h, w), config.edge_margin)
    
    final_coords = non_maximum_suppression_efficient(filtered_coords, config.min_distance)
    
    return {
        'probmap': output[0][0].cpu().numpy(),
        'peak_map': peak_map[0].cpu().numpy(),
        'peak_coords': final_coords,
        'raw_peak_count': len(peak_coords_raw),
        'edge_filtered_count': len(filtered_coords),
        'final_count': len(final_coords)
    }
