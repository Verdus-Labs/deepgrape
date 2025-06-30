import cv2
import numpy as np
import torch
from typing import Tuple, Optional
from config import InferenceConfig


def letterbox(im: np.ndarray, 
              new_shape: Tuple[int, int] = (640, 640), 
              color: Tuple[int, int, int] = (114, 114, 114), 
              auto: bool = True, 
              scaleFill: bool = False, 
              scaleup: bool = True, 
              stride: int = 32) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints
    
    Args:
        im: Input image
        new_shape: Target shape (height, width)
        color: Padding color
        auto: Minimum rectangle padding
        scaleFill: Stretch to fill
        scaleup: Allow scaling up
        stride: Stride multiple constraint
        
    Returns:
        Tuple of (processed_image, ratio, padding)
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, ratio, (dw, dh)


def preprocess_image(img_path: str, config: Optional[InferenceConfig] = None) -> torch.Tensor:
    """
    Preprocess image for model input
    
    Args:
        img_path: Path to input image
        config: Inference configuration
        
    Returns:
        Preprocessed image tensor
        
    Raises:
        ValueError: If image cannot be loaded
        FileNotFoundError: If image file doesn't exist
    """
    if config is None:
        config = InferenceConfig()
        
    if not cv2.os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
        
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    H = int((img.shape[0] + config.stride - 1) / config.stride) * config.stride
    W = int((img.shape[1] + config.stride - 1) / config.stride) * config.stride
    img, ratio, pad = letterbox(img, (H, W), auto=False, scaleup=True, stride=config.stride)
    
    # Normalize and convert to tensor
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def get_image_name(img_path: str) -> str:
    """Extract clean image name from path"""
    import os
    base_name = os.path.basename(img_path)
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        base_name = base_name.replace(ext, '').replace(ext.upper(), '')
    return base_name
