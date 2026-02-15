
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pretrained_microscopy_models as pmm  # The library for MicroNet models

import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

### CONFIGURATION ###

MODEL_ARCHITECTURE = "Unet"      # Options: "Unet", "Segnet"
ENCODER_NAME = "resnet50"    # Options: "resnet34", "resnet50"
PRETRAINING_STRATEGY = "imagenet"  # Options: "imagenet" or "micronet"

CLASS_COLORS = {
    0: [0, 0, 0],         # Background 
    1: [0, 255, 255],     # Monolayer  
    2: [255, 0, 0]        # Multilayer
}

VISUALIZATION_SPACING = 0.05


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4      
NUM_EPOCHS = 100    
NUM_WORKERS = 0     
NUM_CLASSES = 3    
VAL_BATCH_SIZE = 4  

#  data paths 
#"
TRAIN_IMG_DIR = './data/train/images/'
TRAIN_MASK_DIR = './data/train/masks/'
VAL_IMG_DIR = './data/val/images/'
VAL_MASK_DIR = './data/val/masks/'


class FlakeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_filename = os.path.splitext(self.images[index])[0] + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR); image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None: raise FileNotFoundError(f"Mask file not found or is corrupted: {mask_path}")
        mask[mask == 20] = 1; mask[mask == 40] = 2
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask.long()

train_transform = A.Compose([
    A.Resize(height=512, width=512), A.Rotate(limit=45, p=1.0),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")
    for data, targets in loop:
        data, targets = data.to(device=DEVICE), targets.to(device=DEVICE)
        with torch.cuda.amp.autocast():
            predictions = model(data); loss = loss_fn(predictions, targets)
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model, num_classes, device="cuda"):
    num_correct, num_pixels, dice_score = 0, 0, 0
    intersection_per_class = torch.zeros(num_classes, device=device)
    union_per_class = torch.zeros(num_classes, device=device)
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(device), y.to(device); preds = torch.argmax(model(x), dim=1)
            num_correct += (preds == y).sum(); num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            for cls in range(num_classes):
                intersection_per_class[cls] += ((preds == cls) & (y == cls)).sum()
                union_per_class[cls] += ((preds == cls) | (y == cls)).sum()
    iou_per_class = intersection_per_class / (union_per_class + 1e-8)
    print(f"\n--- Validation Metrics ---"); print(f"Pixel Accuracy: {num_correct/num_pixels*100:.2f}%")
    print(f"Dice Score: {dice_score/len(loader):.4f}"); print(f"IoU for Background (Class 0): {iou_per_class[0]:.4f}")
    print(f"IoU for Monolayer (Class 1): {iou_per_class[1]:.4f}"); print(f"IoU for Multilayer (Class 2): {iou_per_class[2]:.4f}")
    print(f"Mean IoU (mIoU): {iou_per_class.mean():.4f}"); print(f"--------------------------\n")
    model.train()

def visualize_predictions(loader, model, class_colors, spacing=0.1, device="cuda"):
    model.eval()
    try: images, true_masks = next(iter(loader))
    except StopIteration: print("Validation loader is empty."); return
    
    images, true_masks = images.to(device), true_masks.to(device)
    with torch.no_grad(): predicted_masks = torch.argmax(model(images), dim=1).cpu().numpy()
    images, true_masks = images.cpu().numpy(), true_masks.cpu().numpy()
    
    num_images_to_plot = len(images)
    
    def mask_to_rgb(mask, color_map):
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in color_map.items(): rgb_mask[mask == class_idx] = color
        return rgb_mask

    fig, axes = plt.subplots(num_images_to_plot, 3, figsize=(15, 5 * num_images_to_plot))
    
   
    fig.subplots_adjust(wspace=spacing, hspace=spacing)
    
  
    if num_images_to_plot == 1:
        # If only one row, axes is a 1D array
        axes[0].set_title("Original Image", fontsize=14)
        axes[1].set_title("Ground Truth Mask", fontsize=14)
        axes[2].set_title("Predicted Mask", fontsize=14)
    else:
        # If multiple rows, axes is a 2D array
        axes[0, 0].set_title("Original Image", fontsize=14)
        axes[0, 1].set_title("Ground Truth Mask", fontsize=14)
        axes[0, 2].set_title("Predicted Mask", fontsize=14)

    for i in range(num_images_to_plot):
        image = np.transpose(images[i], (1, 2, 0)); image = np.clip(image, 0, 1)
        true_mask_viz = mask_to_rgb(true_masks[i], class_colors)
        pred_mask_viz = mask_to_rgb(predicted_masks[i], class_colors)
        
        ax1, ax2, ax3 = axes[i] if num_images_to_plot > 1 else axes
        
        ax1.imshow(image); ax1.axis('off')
        ax2.imshow(true_mask_viz); ax2.axis('off')
        ax3.imshow(pred_mask_viz); ax3.axis('off')
        
    plt.savefig(f"prediction_grid_{MODEL_ARCHITECTURE}_{ENCODER_NAME}_{PRETRAINING_STRATEGY}.png", dpi=150, bbox_inches='tight')
    plt.show(); model.train()

def get_model(architecture, encoder, pretraining, num_classes):
    print(f"Creating model: {architecture} with {encoder} encoder pretrained on: {pretraining}")
    try:
        model = pmm.segmentation_training.create_segmentation_model(
            architecture=architecture, encoder=encoder, encoder_weights=pretraining, classes=num_classes
        )
    except Exception as e:
        print(f"!!! ERROR: Could not create model with pmm library."); print(f"!!! Error details: {e}"); raise e
    return model.to(DEVICE)

def main():
    model = get_model(MODEL_ARCHITECTURE, ENCODER_NAME, PRETRAINING_STRATEGY, NUM_CLASSES)
    loss_fn = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_ds = FlakeDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_ds = FlakeDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n======== Epoch {epoch+1}/{NUM_EPOCHS} ({MODEL_ARCHITECTURE} | {ENCODER_NAME} | {PRETRAINING_STRATEGY}) ========")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        check_accuracy(val_loader, model, num_classes=NUM_CLASSES, device=DEVICE)
        
    model_save_path = f"{MODEL_ARCHITECTURE.lower()}_{ENCODER_NAME}_{PRETRAINING_STRATEGY}_flakes.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n🎉 Model training complete! Model saved to {model_save_path}")
    
    print("\n--- Visualizing some predictions on the validation set ---")
    visualize_predictions(val_loader, model, class_colors=CLASS_COLORS, device=DEVICE)
    
if __name__ == "__main__":
    main()