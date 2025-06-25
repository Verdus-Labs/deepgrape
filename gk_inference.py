import modal
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

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
    # Add the checkpoint file
    .add_local_file("weight/best_50_0.2.pth.tar", "/app/best_50_0.2.pth.tar", copy=True)
    # Add just the specific image we want to process
    .add_local_file("input_images/1.jpg", "/input_images/1.jpg", copy=True)
)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
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
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess_single_image(img_path):
    """Preprocess image for model input"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Align image to 32-pixel boundaries (required by model)
    H = int((img.shape[0] + 32 - 1) / 32) * 32
    W = int((img.shape[1] + 32 - 1) / 32) * 32
    img, ratio, pad = letterbox(img, (H, W), auto=False, scaleup=True)
    
    # Normalize and convert to tensor
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # Add batch dimension
    
    return img_tensor

def inference_single_image(model, img_path, val_score=0.7, mask_threshold=9e-3, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Preprocess image
        val_data = preprocess_single_image(img_path).to(device)

        # Forward pass
        output, _ = model(val_data)  # (1, 1, H, W)
        
        # Debug: Check output statistics
        print(f"Raw output - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")

        # Smooth output (avg pool)
        avg_pooled = F.avg_pool2d(output[0], kernel_size=3, stride=1, padding=1)
        
        # Detect local maxima (peak map) - use larger kernel for more selective peaks
        max_pooled = F.max_pool2d(avg_pooled, kernel_size=7, stride=1, padding=3)  # Increased from 3 to 7
        peak_map = torch.where(avg_pooled == max_pooled, avg_pooled, torch.zeros_like(avg_pooled))
        
        print(f"Peak map - min: {peak_map.min():.4f}, max: {peak_map.max():.4f}")
        print(f"Peaks above {val_score}: {(peak_map >= val_score).sum().item()}")

        # Binary map of detected peaks with higher threshold
        peak_binary = (peak_map >= val_score).to(torch.uint8)  # (1, H, W)
        peak_coords_raw = torch.nonzero(peak_binary[0]).cpu().numpy()  # list of (y,x)
        
        # Filter out edge detections (likely false positives)
        h, w = peak_binary[0].shape
        edge_margin = 50  # pixels from edge
        
        filtered_coords = []
        for y, x in peak_coords_raw:
            if (edge_margin <= y < h - edge_margin and 
                edge_margin <= x < w - edge_margin):
                filtered_coords.append([y, x])
        
        peak_coords = np.array(filtered_coords) if filtered_coords else np.array([]).reshape(0, 2)
        
        print(f"Peaks after edge filtering: {len(peak_coords)} (removed {len(peak_coords_raw) - len(peak_coords)} edge detections)")
        
        # Additional filtering: remove peaks that are too close to each other
        if len(peak_coords) > 0:
            final_coords = []
            min_distance = 25  # minimum pixels between berries
            
            for y, x in peak_coords:
                too_close = False
                for fy, fx in final_coords:
                    distance = np.sqrt((y - fy)**2 + (x - fx)**2)
                    if distance < min_distance:
                        too_close = True
                        break
                if not too_close:
                    final_coords.append([y, x])
            
            peak_coords = np.array(final_coords)
            print(f"Final peaks after distance filtering: {len(peak_coords)}")

        return {
            'probmap': output[0][0].cpu().numpy(),  # raw probmap
            'peak_map': peak_map[0].cpu().numpy(),  # local maxima values
            'peak_coords': peak_coords,             # (y,x) coordinates of peaks
            'original_tensor': val_data,            # for visualization
            'raw_peak_count': len(peak_coords_raw), # before filtering
            'edge_filtered_count': len(filtered_coords), # after edge filtering
        }

def create_visualizations(img_path, results, output_dir):
    """Create and save visualization images"""
    import random
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Warning: Could not load original image {img_path}")
        return
    
    img_name = os.path.basename(img_path).replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    
    # 1. Save probability map
    probmap = results['probmap']
    probmap_vis = (probmap * 255).astype(np.uint8)
    probmap_path = f"{output_dir}/{img_name}_probmap.jpg"
    cv2.imwrite(probmap_path, probmap_vis)
    print(f"Saved probability map: {probmap_path}")
    
    # 2. Save peak map
    peak_map = results['peak_map']
    peak_vis = (peak_map * 255).astype(np.uint8)
    peak_path = f"{output_dir}/{img_name}_peaks.jpg"
    cv2.imwrite(peak_path, peak_vis)
    print(f"Saved peak map: {peak_path}")
    
    # 3. Create berry detection overlay
    if len(results['peak_coords']) > 0:
        overlay = original_img.copy()
        for y, x in results['peak_coords']:
            # Scale coordinates if needed
            scale_y = original_img.shape[0] / probmap.shape[0]
            scale_x = original_img.shape[1] / probmap.shape[1]
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            
            # Draw colored circles for detected berries
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(overlay, (orig_x, orig_y), 10, color, -1)
            cv2.circle(overlay, (orig_x, orig_y), 12, (255, 255, 255), 2)
        
        overlay_path = f"{output_dir}/{img_name}_overlay.jpg"
        cv2.imwrite(overlay_path, overlay)
        print(f"Saved berry overlay: {overlay_path}")
    
    # 4. Save text results
    results_path = f"{output_dir}/{img_name}_results.txt"
    with open(results_path, 'w') as f:
        f.write(f"Image: {img_path}\n")
        f.write(f"Detected berries: {len(results['peak_coords'])}\n")
        f.write(f"Probability map shape: {probmap.shape}\n")
        f.write(f"Peak coordinates:\n")
        for i, (y, x) in enumerate(results['peak_coords']):
            f.write(f"  Berry {i+1}: (y={y}, x={x})\n")
    
    print(f"Saved results text: {results_path}")
    
    return {
        'probmap_path': probmap_path,
        'peak_path': peak_path,
        'overlay_path': overlay_path if len(results['peak_coords']) > 0 else None,
        'results_path': results_path
    }

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
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = init_model('Resnet50').to(device)
    model, epoch = load_checkpoint(model, '/app/best_50_0.2.pth.tar')
    print(f"Model loaded (epoch {epoch})")
    
    img_path = "/input_images/1.jpg"
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Testing different thresholds:")
    print("=" * 50)
    
    for threshold in thresholds:
        results = inference_single_image(model, img_path, val_score=threshold, device=device)
        num_berries = len(results['peak_coords'])
        print(f"Threshold {threshold}: {num_berries} berries detected")
    
    print("=" * 50)
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
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = init_model('Resnet50').to(device)
    model, epoch = load_checkpoint(model, '/app/best_50_0.2.pth.tar')
    print(f"Model loaded (epoch {epoch})")
    
    # Run inference on 1.jpg
    img_path = "/input_images/1.jpg"
    print(f"Processing: {img_path}")
    
    results = inference_single_image(model, img_path, val_score=0.7, device=device)
    
    # Print results
    num_berries = len(results['peak_coords'])
    print(f"Detected {num_berries} berries")
    print(f"Peak coordinates: {results['peak_coords'][:10]}...")  # Show first 10
    print(f"Probability map shape: {results['probmap'].shape}")
    print(f"Peak map shape: {results['peak_map'].shape}")
    
    # Create and save visualizations to berry-data volume
    output_dir = "/berry_data/visualizations"
    print(f"Saving visualizations to: {output_dir}")
    
    vis_files = create_visualizations(img_path, results, output_dir)
    
    # Commit changes to volume
    berry_volume.commit()
    print("Visualizations saved to berry-data volume!")
    
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
    print("Starting grape inference on Modal for 1.jpg...")
    
    # Run the inference
    result = run_single_image_inference.remote()
    
    print(f"Inference completed!")
    print(f"Found {result['num_berries']} berries in 1.jpg")
    return result