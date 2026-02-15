import torch
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pretrained_microscopy_models as pmm
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from tqdm import tqdm 

### CONFIGURATION ###

# SET THE PARAMETERS OF THE TRAINED MODEL TO USE
MODEL_ARCHITECTURE = "Unet"
ENCODER_NAME = "resnet50"
PRETRAINING_STRATEGY = "imagenet"

# specify model path and input and output file paths
MODELS_DIR = './models'
INPUT_IMAGE_DIR = './data/val/images/'
OUTPUT_DIR = './results'


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
CLASS_COLORS = {
    0: [0, 0, 0],         # Background 
    1: [0, 255, 255],     # Monolayer  
    2: [255, 0, 0]        # Multilayer 
}
VISUALIZATION_SPACING = 0.03
# =====================================================================================



def predict_large_image(model, image_path, num_classes, tile_size=512, overlap=100):
    """Predicts a mask and a confidence map for a single large image."""
    model.to(DEVICE); model.eval()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width, _ = image.shape
    transform = A.Compose([A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0), ToTensorV2()])
    step = tile_size - overlap
    pad_height = (step - (original_height - overlap) % step) % step
    pad_width = (step - (original_width - overlap) % step) % step
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    padded_height, padded_width, _ = padded_image.shape
    logits_canvas = torch.zeros((num_classes, padded_height, padded_width), dtype=torch.float32, device=DEVICE)
    counts_canvas = torch.zeros((padded_height, padded_width), dtype=torch.float32, device=DEVICE)
    for y in range(0, padded_height - overlap, step):
        for x in range(0, padded_width - overlap, step):
            tile = padded_image[y:y+tile_size, x:x+tile_size]
            with torch.no_grad():
                transformed_tile = transform(image=tile)["image"].unsqueeze(0).to(DEVICE)
                prediction_logits = model(transformed_tile)
                logits_canvas[:, y:y+tile_size, x:x+tile_size] += prediction_logits.squeeze(0)
                counts_canvas[y:y+tile_size, x:x+tile_size] += 1
    averaged_logits = logits_canvas / (counts_canvas + 1e-8)
    averaged_probabilities = F.softmax(averaged_logits, dim=0)
    confidence_map, final_prediction = torch.max(averaged_probabilities, dim=0)
    final_prediction = final_prediction.cpu().numpy().astype(np.uint8)
    confidence_map = confidence_map.cpu().numpy()
    return final_prediction[:original_height, :original_width], confidence_map[:original_height, :original_width]

def generate_statistics_report(predicted_mask, confidence_map):
    """Calculates coverage and confidence stats and returns a formatted string report."""
    total_pixels = predicted_mask.size; monolayer_pixels = np.count_nonzero(predicted_mask == 1); multilayer_pixels = np.count_nonzero(predicted_mask == 2)
    avg_mono_conf = np.mean(confidence_map[predicted_mask == 1]) if monolayer_pixels > 0 else 0
    avg_multi_conf = np.mean(confidence_map[predicted_mask == 2]) if multilayer_pixels > 0 else 0
    monolayer_percent = (monolayer_pixels / total_pixels) * 100; multilayer_percent = (multilayer_pixels / total_pixels) * 100
    if multilayer_pixels > 0: ratio_text = f"{monolayer_pixels / multilayer_pixels:.2f} : 1"
    else: ratio_text = "N/A"
    total_flake_pixels = monolayer_pixels + multilayer_pixels; composition_line = ""
    if total_flake_pixels > 0:
        mono_flake_percent = (monolayer_pixels / total_flake_pixels) * 100; multi_flake_percent = (multilayer_pixels / total_flake_pixels) * 100
        composition_line = f"\n  - Composition: {mono_flake_percent:.1f}% Mono / {multi_flake_percent:.1f}% Multi"
    report = (f"-- Quantitative Analysis --\n\nImage Coverage:\n"
              f"  Monolayer : {monolayer_percent:>6.2f}%\n  Multilayer: {multilayer_percent:>6.2f}%\n\n"
              f"Material Analysis:\n  Mono:Multi Ratio: {ratio_text}{composition_line}\n\n"
              f"Model Confidence:\n  Avg. Mono Conf: {avg_mono_conf:.1%}\n  Avg. Multi Conf: {avg_multi_conf:.1%}\n"
              f"--------------------------")
    return report

def save_and_visualize_results(original_image, predicted_mask, confidence_map, report_text, output_path_prefix, class_colors, spacing):
    """Saves the raw mask and creates a combined visualization with images and stats."""
    def mask_to_rgb(mask, color_map):
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8); 
        for class_idx, color in color_map.items(): rgb_mask[mask == class_idx] = color
        return rgb_mask
    output_mask_path = f"{output_path_prefix}_predicted_mask_raw.png"; cv2.imwrite(output_mask_path, predicted_mask)
    color_visual_mask = mask_to_rgb(predicted_mask, class_colors)
    fig = plt.figure(figsize=(28, 8)); gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.8], wspace=spacing)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1]); ax3 = fig.add_subplot(gs[2]); ax_text = fig.add_subplot(gs[3])
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)); ax1.set_title("Original Image"); ax1.axis('off')
    ax2.imshow(color_visual_mask); ax2.set_title("Predicted Mask"); ax2.axis('off')
    conf_plot = ax3.imshow(confidence_map, cmap='magma', vmin=0, vmax=1); ax3.set_title("Prediction Confidence"); ax3.axis('off')
    fig.colorbar(conf_plot, ax=ax3, fraction=0.046, pad=0.04)
    ax_text.text(0.0, 0.95, report_text, ha="left", va="top", fontsize=11, family='monospace'); ax_text.axis('off')
    output_viz_path = f"{output_path_prefix}_visualization_report.png"; plt.savefig(output_viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig) 

def get_model(architecture, encoder, pretraining, num_classes):
    """Creates an empty model structure."""
   
    print(f"Creating model structure: {architecture} with {encoder} encoder...")
    try:
        model = pmm.segmentation_training.create_segmentation_model(
            architecture=architecture, encoder=encoder, encoder_weights=pretraining, classes=num_classes
        )
    except Exception as e:
        print(f"!!! ERROR: Could not create model structure."); print(f"!!! Error details: {e}"); raise e
    return model.to(DEVICE)


def main():
    print("--- Batch Prediction and Analysis Script ---")
    
    # Automatically construct the model path and load the model 
    model_filename = f"{MODEL_ARCHITECTURE.lower()}_{ENCODER_NAME}_{PRETRAINING_STRATEGY}_flakes.pth"
    model_path = os.path.join(MODELS_DIR, model_filename) if MODELS_DIR else model_filename
    model = get_model(MODEL_ARCHITECTURE, ENCODER_NAME, PRETRAINING_STRATEGY, NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Successfully loaded trained model: {model_filename}")
    except FileNotFoundError:
        print(f"!!! FATAL ERROR: Model file not found at: {model_path}"); return
    except Exception as e:
        print(f"!!! FATAL ERROR: Failed to load model weights. Error: {e}"); return

    # Get the list of images to process
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    if not image_files:
        print(f"!!! ERROR: No images found in the specified input directory: {INPUT_IMAGE_DIR}"); return
    
    print(f"Found {len(image_files)} images to process.")

    # Loop through each image, predict, analyze, and save
    for image_filename in tqdm(image_files, desc="Processing Images"):
        try:
            image_path = os.path.join(INPUT_IMAGE_DIR, image_filename)
            
            predicted_mask, confidence_map = predict_large_image(model, image_path, num_classes=NUM_CLASSES)
            statistics_report_text = generate_statistics_report(predicted_mask, confidence_map)
            
            # Print the report for the current image to the console
            print(f"\n--- Results for: {image_filename} ---")
            print(statistics_report_text)

            original_image = cv2.imread(image_path)
            base_filename = os.path.splitext(image_filename)[0]
            
            output_path_prefix = os.path.join(OUTPUT_DIR, base_filename) if OUTPUT_DIR else base_filename
            if OUTPUT_DIR: os.makedirs(OUTPUT_DIR, exist_ok=True)

            save_and_visualize_results(
                original_image, predicted_mask, confidence_map, 
                statistics_report_text, output_path_prefix, 
                CLASS_COLORS, VISUALIZATION_SPACING
            )
        except Exception as e:
            # If one image fails, print the error and continue with the next one
            print(f"\n!!! ERROR processing file: {image_filename} !!!")
            print(f"    Reason: {e}")
            print("    Skipping to next file.")
            continue
            
    print("\n🎉 Batch processing complete. 🎉")
    
if __name__ == "__main__":
    main()
