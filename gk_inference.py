import modal
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys
sys.path.append('/app')
from config import InferenceConfig, ModelConfig, VisualizationConfig
from utils.image_processing import preprocess_image, get_image_name
from utils.post_processing import detect_peaks
from utils.visualization import create_visualizations
from utils.logging_config import setup_logging, get_logger

# Modal app setup
app = modal.App("grape-inference-single")

# Volume for storing berry data visualizations
berry_volume = modal.Volume.from_name("berry-data", create_if_missing=True)

# Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch", 
        "torchvision", 
        "opencv-python-headless", 
        "numpy"
    ])
    .apt_install(["libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"])
    # Add model files
    .add_local_dir("app", "/app", copy=True)
    .add_local_dir("utils", "/app/utils", copy=True)
    .add_local_file("config.py", "/app/config.py", copy=True)
    # Add the checkpoint file
    .add_local_file("weight/best_50_0.2.pth.tar", "/app/best_50_0.2.pth.tar", copy=True)
    # Add just the specific image we want to process
    .add_local_file("input_images/1.jpg", "/input_images/1.jpg", copy=True)
)


def inference_single_image(model, img_path, config=None, device='cuda'):
    """Run inference on a single image with configurable parameters"""
    if config is None:
        config = InferenceConfig()
    
    logger = get_logger()
    
    model.eval()
    with torch.no_grad():
        try:
            val_data = preprocess_image(img_path, config).to(device)
        except Exception as e:
            logger.error(f"Failed to preprocess image {img_path}: {e}")
            raise

        output, _ = model(val_data)
        
        logger.info(f"Raw output - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
        
        results = detect_peaks(output, config)
        results['original_tensor'] = val_data
        
        logger.info(f"Detected {len(results['peak_coords'])} berries after all filtering")
        
        return results


@app.function(
    image=image,
    volumes={"/berry_data": berry_volume},
    gpu="A10G",
    timeout=600
)
def test_different_thresholds():
    """Test different threshold values to find optimal berry detection"""
    import sys
    sys.path.insert(0, '/app')
    
    from model_init import init_model
    from train import load_checkpoint
    
    logger = setup_logging()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = init_model('Resnet50').to(device)
    model, epoch = load_checkpoint(model, '/app/best_50_0.2.pth.tar')
    logger.info(f"Model loaded (epoch {epoch})")
    
    img_path = "/input_images/1.jpg"
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    logger.info("Testing different thresholds:")
    logger.info("=" * 50)
    
    for threshold in thresholds:
        config = InferenceConfig(val_score=threshold)
        results = inference_single_image(model, img_path, config=config, device=device)
        num_berries = len(results['peak_coords'])
        logger.info(f"Threshold {threshold}: {num_berries} berries detected")
    
    logger.info("=" * 50)
    return {"message": "Threshold testing complete"}

@app.function(
    image=image,
    volumes={"/berry_data": berry_volume},
    gpu="A10G",
    timeout=600
)
def run_single_image_inference():
    """Run inference on 1.jpg and save visualizations to berry-data volume"""
    import sys
    sys.path.insert(0, '/app')
    
    from model_init import init_model
    from train import load_checkpoint
    
    logger = setup_logging()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = init_model('Resnet50').to(device)
    model, epoch = load_checkpoint(model, '/app/best_50_0.2.pth.tar')
    logger.info(f"Model loaded (epoch {epoch})")
    
    # Run inference on 1.jpg
    img_path = "/input_images/1.jpg"
    logger.info(f"Processing: {img_path}")
    
    config = InferenceConfig(val_score=0.7)
    vis_config = VisualizationConfig()
    results = inference_single_image(model, img_path, config=config, device=device)
    
    # Print results
    num_berries = len(results['peak_coords'])
    logger.info(f"Detected {num_berries} berries")
    logger.info(f"Peak coordinates: {results['peak_coords'][:10]}...")  # Show first 10
    logger.info(f"Probability map shape: {results['probmap'].shape}")
    logger.info(f"Peak map shape: {results['peak_map'].shape}")
    
    # Create and save visualizations to berry-data volume
    output_dir = "/berry_data/visualizations"
    logger.info(f"Saving visualizations to: {output_dir}")
    
    vis_files = create_visualizations(img_path, results, output_dir, vis_config)
    
    # Commit changes to volume
    berry_volume.commit()
    logger.info("Visualizations saved to berry-data volume!")
    
    return {
        'num_berries': num_berries,
        'peak_coords': results['peak_coords'].tolist(),
        'probmap_shape': results['probmap'].shape,
        'peak_map_shape': results['peak_map'].shape,
        'visualization_files': vis_files
    }

@app.local_entrypoint()
def main():
    """Local entrypoint to run the inference"""
    logger = setup_logging()
    logger.info("Starting grape inference on Modal for 1.jpg...")
    
    # Run the inference
    result = run_single_image_inference.remote()
    
    logger.info(f"Inference completed!")
    logger.info(f"Found {result['num_berries']} berries in 1.jpg")
    return result
