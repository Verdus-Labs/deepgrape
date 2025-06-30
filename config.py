import os
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class InferenceConfig:
    """Configuration class for inference parameters"""
    val_score: float = 0.7
    mask_threshold: float = 9e-3
    edge_margin: int = 50
    min_distance: int = 25
    avg_pool_kernel: int = 3
    max_pool_kernel: int = 7
    default_shape: Tuple[int, int] = (640, 640)
    stride: int = 32
    
    
@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    default_model: str = 'Resnet50'
    checkpoint_path: str = '/app/best_50_0.2.pth.tar'
    available_models: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = ['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101']


@dataclass
class VisualizationConfig:
    """Configuration class for visualization parameters"""
    circle_radius: int = 10
    circle_border: int = 12
    border_color: Tuple[int, int, int] = (255, 255, 255)
    border_thickness: int = 2
    supported_extensions: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.png', '.jpg', '.jpeg']
