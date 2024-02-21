#util2.py

import torch
from torchvision.models.segmentation import fcn_resnet50
import torch.nn as nn
from torchvision import transforms as T
import copy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def collate_fn(batch):
    images, masks, classes = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks, classes


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks, _ in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item()
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_pixels
    return {'loss': avg_loss, 'accuracy': accuracy}


# Define the model
def get_model(num_classes):
    # Load an FCN model pre-trained on COCO
    model = fcn_resnet50(pretrained=True)

    # Replace the classifier with a new one for the number of classes
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return model

num_epochs = 25


def get_transform(train):
    transform_list = []  # Renamed the list to avoid conflict
    transform_list.append(T.Resize((256, 256)))
    if train:
        transform_list.append(T.RandomHorizontalFlip(0.5))
        transform_list.append(T.RandomVerticalFlip(0.5))
        transform_list.append(T.RandomRotation(20))
        transform_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    transform_list.append(T.ToTensor())
    return T.Compose(transform_list)


# Train and validate function
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, device):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
    
    model.load_state_dict(best_model_wts)
    return model


# Function to apply ensemble of models to an image and average their outputs
def ensemble_predict(models, image_tensor):
    ensemble_output = None
    for model in models:
        with torch.no_grad():
            output = model(image_tensor)['out']
        if ensemble_output is None:
            ensemble_output = output
        else:
            ensemble_output += output
    ensemble_output /= len(models)
    return ensemble_output


# Function to visualize the prediction
def visualize_prediction(image_path, prediction, alpha=0.5, title='Prediction'):
    # Read the original image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    image_np = np.array(image)
    
    # Create an RGBA mask where each class is colored differently with the specified transparency
    mask = prediction.cpu().numpy()
    mask_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    mask_rgba[mask == 1] = [255, 0, 0, int(255 * alpha)]  # Cancer in red
    mask_rgba[mask == 2] = [0, 255, 0, int(255 * alpha)]  # Mix in green
    mask_rgba[mask == 3] = [0, 0, 255, int(255 * alpha)]  # Warthin in blue

    # Overlay the mask on the image
    overlayed_image = image.copy()
    overlayed_image.paste(Image.fromarray(mask_rgba, mode='RGBA'), (0, 0), Image.fromarray(mask_rgba, mode='RGBA'))
    
    # Display the image with the overlay
    plt.figure()  # Create a new figure
    plt.imshow(overlayed_image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def dice_coefficient(pred_mask, gt_mask):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat)
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice


def resize_prediction(prediction, target_shape):
    if len(prediction.shape) == 2:  # 如果只有两个维度，增加一个通道维度
        prediction = prediction.unsqueeze(0)
    prediction = prediction.unsqueeze(0)  # 增加一个批次维度
    resized_prediction = torch.nn.functional.interpolate(prediction.float(), 
                                                         size=target_shape, 
                                                         mode='nearest').squeeze(0)
    return resized_prediction.squeeze(0)
