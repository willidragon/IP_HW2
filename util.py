#util.py

import torch
from tqdm import tqdm
import torchvision.models.detection as detection
from torchvision.ops import box_iou
from torchvision.ops import nms
from torch.nn.functional import softmax
from collections import defaultdict
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json


def collate_fn(batch):
    """
    Collate function to be used with PyTorch DataLoader. 
    Converts a batch of images and targets into a format suitable for model training.

    Args:
        batch (list): List of tuples containing images and targets.

    Returns:
        tuple: A tuple containing a stacked tensor of images and a list of targets.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    targets = [{k: torch.as_tensor(v) for k, v in t.items()} for t in targets]
    return images, targets

def get_model(num_classes):
    """
    Get a custom Faster R-CNN model for the specified number of classes.
    Loads a pre-trained Faster R-CNN model and replaces the classifier with a new one 
    for the given number of classes.

    Args:
        num_classes (int): Number of target classes.

    Returns:
        torch.nn.Module: Custom Faster R-CNN model.
    """
    # Load a model pre-trained on COCO
    model = detection.fasterrcnn_resnet50_fpn(weights=True)

    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, scheduler, data_loader, device):
    """
    Perform one training epoch on the provided data.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        device (torch.device): Device on which the model and data should be located.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(data_loader, desc="Training", unit="batch"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
    scheduler.step()
    return running_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """
    Evaluate the model on the provided data and calculate the mean Intersection over Union (IoU).

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device on which the model and data should be located.

    Returns:
        float: Mean Intersection over Union (IoU) for the evaluation.
    """
    model.eval()
    iou_sum = 0
    num_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating", unit="batch"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for target, output in zip(targets, outputs):
                iou = box_iou(target['boxes'], output['boxes'])
                iou_sum += iou.sum().item()
                num_samples += iou.numel()

    return iou_sum / num_samples



def ensemble_predict_single_image(models, image_tensor, device, iou_threshold=0.5):
    """
    Make predictions on a single image using an ensemble of models.

    Args:
        models (list): List of PyTorch models for ensemble prediction.
        image_tensor (torch.Tensor): Input image tensor.
        device (torch.device): Device on which the models and data should be located.
        iou_threshold (float, optional): IoU threshold for non-maximum suppression. Default is 0.5.

    Returns:
        dict: Final ensemble prediction containing 'boxes' and 'labels'.
    """
    # Ensure the image tensor is in the correct format and on the right device
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        all_boxes = []
        all_labels = []
        all_scores = []

        for model in models:
            model = model.to(device)
            prediction = model(image_tensor)[0]

            scores = softmax(prediction['scores'], dim=0).to(device)  # Ensure scores are on the CUDA device
            labels = prediction['labels'].to(device)  # Ensure labels are on the CUDA device

            for box, label, score in zip(prediction['boxes'], labels, scores):
                all_boxes.append(box)
                all_labels.append(label)
                all_scores.append(score)

        # Group boxes by labels
        grouped_boxes = defaultdict(list)
        grouped_scores = defaultdict(list)

        for box, label, score in zip(all_boxes, all_labels, all_scores):
            grouped_boxes[label.item()].append(box)
            grouped_scores[label.item()].append(score)

        # Average boxes and find the label with the highest average score
        final_boxes = []
        final_labels = []

        for label, boxes in grouped_boxes.items():
            avg_box = torch.mean(torch.stack(boxes), dim=0)
            avg_score = torch.mean(torch.tensor(grouped_scores[label], device=device))
            final_boxes.append(avg_box)
            final_labels.append((label, avg_score.item()))  # Ensure tuple of label and score

        # Sort labels by average score and keep the label with the highest score
        # Prepare the tensors for NMS
        final_boxes = torch.stack(final_boxes).to(device)
        final_scores = torch.tensor([score for _, score in final_labels], device=device)  # Directly use score

        # Apply NMS
        keep_indices = nms(final_boxes, final_scores, iou_threshold)
        final_prediction = {
            'boxes': final_boxes[keep_indices], 
            'labels': torch.tensor([label for label, _ in final_labels], device=device)[keep_indices]
        }

        return final_prediction


def draw_boxes(image, prediction, label_name):
    """
    Draw bounding boxes on an image based on the prediction and annotate with a label name.

    Args:
        image (torch.Tensor): Input image tensor.
        prediction (dict): Prediction containing 'boxes' and 'labels'.
        label_name (str): Label name to annotate the image with.
    """
    # Convert image tensor to numpy array
    image = image.permute(1, 2, 0).cpu().numpy()


    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    ax.set_title(label_name)  # Set the title of the plot

    # Extract boxes and labels
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()

    # Define colors and labels for your classes
    colors = {1: 'red', 2: 'blue'}
    class_labels = {1: 'right normal', 2: 'left normal'}

    # Draw boxes with labels
    for box, label in zip(boxes, labels):
        x, y, x2, y2 = box
        rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2,
                                 edgecolor=colors.get(label, 'green'), facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f"{label_name}: {class_labels.get(label, 'unknown')}",
                 color=colors.get(label, 'green'), fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.draw()
    
    plt.waitforbuttonpress() 



def get_ground_truth_boxes(json_path, original_size, transformed_size):
    
    with open(json_path, 'r') as file:
        data = json.load(file)

    boxes = []
    labels = []

    for shape in data['shapes']:
        label = shape['label'].lower()

        if label not in ['left normal', 'right normal']:
            continue

        points = shape['points']
        x_coordinates, y_coordinates = zip(*points)
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)

        # Scale the bounding box coordinates according to the image resize
        scale_x, scale_y = transformed_size[1] / original_size[0], transformed_size[0] / original_size[1]
        box = [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]
        boxes.append(box)
        labels.append(label)

    return boxes, labels


def get_prediction_boxes(image, prediction):
    image = image.permute(1, 2, 0).cpu().numpy()

    # Extract boxes and labels directly from the prediction dictionary
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    return boxes, labels


def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou