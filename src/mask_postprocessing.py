"""
This module contains functions for applying post-processing operations to segmentation masks.
"""
import cv2
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from src import model_inference, settings, ui_input_variables


def ensure_rgb(segmentation_mask: np.array) -> np.array:
    """ Ensure the segmentation mask is in RGB format. """
    if len(segmentation_mask.shape) == 3 and segmentation_mask.shape[2] == 3:
        return segmentation_mask
    return settings.COLOR_MAP[segmentation_mask.astype(np.uint8)]

def apply_erosion(segmentation_mask):
    """
    Apply erosion to the segmentation mask to reduce the size of the segmented objects.

    Args:
        segmentation_mask (numpy array): The segmentation mask to be eroded.

    Returns:
        numpy array: The eroded segmentation mask.
    """
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    eroded_mask = cv2.erode(segmentation_mask_rgb, settings.EROSION_KERNEL, iterations=settings.EROSION_ITERATIONS)
    return eroded_mask.astype(np.uint8)

def apply_dilation(segmentation_mask):
    """
    Apply dilation to the segmentation mask to increase the size of the segmented objects.

    Args:
        segmentation_mask (numpy array): The segmentation mask to be dilated.

    Returns:
        numpy array: The dilated segmentation mask.
    """
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    dilated_mask = cv2.dilate(segmentation_mask_rgb, settings.DILATION_KERNEL, iterations=settings.DILATION_ITERATIONS)
    return dilated_mask.astype(np.uint8)

def apply_gaussian_smoothing(segmentation_mask):
    """
    Applies Gaussian smoothing to a segmentation mask.

    Args:
        segmentation_mask (numpy array): The segmentation mask to be smoothed.

    Returns:
        numpy array: The smoothed segmentation mask.
    """
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    return np.array(cv2.GaussianBlur(segmentation_mask_rgb, settings.GAUSSIAN_SMOOTHING_KERNEL_SHAPE, 0)).astype(np.uint8)

def apply_median_filtering(segmentation_mask):
    """
    Applies a median filter to a segmentation mask.

    Args:
        segmentation_mask (numpy array): The segmentation mask to be filtered.

    Returns:
        numpy array: The filtered segmentation mask.
    """
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    return np.array(cv2.medianBlur(segmentation_mask_rgb, settings.MEDIAN_FILTERING_KERNEL_SIZE)).astype(np.uint8)

def apply_crf(original_image, segmentation_mask):
    """
    Applies Conditional Random Field (CRF) post-processing to a segmentation mask.

    Args:
        original_image (numpy array): The original image used as input to the CRF.
        segmentation_mask (numpy array): The segmentation mask to be processed.

    Returns:
        numpy array: The post-processed segmentation mask with CRF refinement.

    This function applies CRF post-processing to a segmentation mask using the given original image. It first creates a
    DenseCRF object with the appropriate dimensions and number of classes. It then sets the unary energy of the CRF
    using the segmentation mask and other parameters. It adds pairwise Gaussian and bilateral energy terms to the CRF.
    Finally, it infers the most likely labels for each pixel using the CRF and returns the mapped inference labels as a
    numpy array. If the mapped inference labels have a shape equal to the original image shape, it reshapes it to match
    the original image dimensions.
    """
    num_classes = settings.NUM_CLASSES

    dense_crf = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], num_classes)
    unary = unary_from_labels(segmentation_mask, num_classes, gt_prob=0.90, zero_unsure=False)

    dense_crf.setUnaryEnergy(unary)
    dense_crf.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    dense_crf.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    inference = dense_crf.inference(5)

    mapped_inference_labels = np.argmax(inference, axis=0)
    mapped_inference_labels = settings.COLOR_MAP[mapped_inference_labels]

    if mapped_inference_labels.shape[0] == original_image.shape[0] * original_image.shape[1]:
        mapped_inference_labels = mapped_inference_labels.reshape(original_image.shape[0], original_image.shape[1], 3)

    return mapped_inference_labels

def apply_mask_postprocessing(raw: np.array, mask_rgb_np: np.array) -> tuple:
    mask = mask_rgb_np
    if ui_input_variables.CRF_ON:
        mask = apply_crf(raw, mask_rgb_np)

    if ui_input_variables.EROSION_ON:
        mask = apply_erosion(mask)

    if ui_input_variables.DILATION_ON:
        mask = apply_erosion(mask)

    if ui_input_variables.GAUSSIAN_SMOOTHING_ON:
        mask = apply_gaussian_smoothing(mask)

    if ui_input_variables.MEDIAN_FILTERING_ON:
        mask = apply_median_filtering(mask)

    else:
        mask = ensure_rgb(mask)
        segmented_frame_np_gray = model_inference.image_to_tensor(Image.fromarray(mask), settings.MODEL,
                                                                  settings.DEVICE).astype(np.uint8)
        segmented_frame_img_rgb = settings.COLOR_MAP[segmented_frame_np_gray]

        segmented_frame_np_rgb = np.array(segmented_frame_img_rgb)

        raw = cv2.resize(raw, (1280, 704), interpolation=cv2.INTER_NEAREST)
        mask = segmented_frame_np_rgb

    return raw, mask
