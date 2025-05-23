"""
This script contains the code for the model inference.

Video frames are passed to the model and the predicted RGB mask is returned.
"""

from torchvision import transforms
import time
import torch
from torch.quantization import quantize_dynamic
from torch.nn.utils import prune
from PIL import Image
import numpy as np

preprocess = transforms.Compose([
    transforms.Resize((704, 1280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Finalize pruning
    return model

def quantize_model(model):
    quantized_model = quantize_dynamic(
        model,  # Your original model
        {torch.nn.Linear},  # Layers to quantize
        dtype=torch.qint8
    )
    return quantized_model


def load_segmentation_model(model_path: str) -> tuple:
    """
    Load a trained segmentation model from a file.

    Args:
        model_path (str): Path to the model file (.pt).

    Returns:
        model (nn.Module): The loaded segmentation model.
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = torch.load(model_path, map_location=device)

    model = prune_model(model)
    model.to(device)

    if device.type == 'cpu':
        model = quantize_model(model)

    model.eval()

    print(f'\nModel loaded from "{model_path}" at time {time.ctime()}')

    model.eval()
    print(f'Model set to evaluation mode at time {time.ctime()}')
    return model, device


def image_to_tensor(img: Image, trained_model, device: str) -> np.array:
    """
    Converts an input image to a tensor and makes a prediction using a trained model.

    Args:
        img: The input image to be converted.
        trained_model: The trained model used for making predictions.
        device: The device to perform inference on (GPU or CPU).

    Returns:
        output_labels_np: A numpy array representing the predicted class labels.
    """

    if isinstance(img, list):
        img = img[0]
    elif isinstance(img, Image.Image):
        img = img

    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = trained_model(input_tensor)

    output_labels_np = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return output_labels_np

def generate_segmentation_labels_np(images: np.array, trained_model, device: str) -> np.array:
    """
    Converts a batch of input images to tensors and makes predictions using a trained model.

    Args:
        images (list of PIL.Image.Image): List of input images to be converted.
        trained_model (nn.Module): The trained model used for making predictions.
        device (torch.device): The device to perform inference on (GPU or CPU).

    Returns:
        output_labels_np (numpy.ndarray): A numpy array containing the predicted class labels for each image.
    """
    batch_tensors = torch.stack([preprocess(img) for img in images]).to(device)

    with torch.no_grad():
        outputs = trained_model(batch_tensors)

    output_labels_np = torch.argmax(outputs, dim=1).cpu().numpy()

    return output_labels_np