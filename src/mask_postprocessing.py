"""
This module contains functions for applying post-processing operations to segmentation masks.
"""

import cv2
import numpy as np
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from src import settings, ui_input_variables


def ensure_rgb(segmentation_mask: np.array) -> np.array:
    """
    Ensure the segmentation mask is in RGB format using COLOR_MAP.

    Args:
        segmentation_mask (numpy array): Segmentation mask of shape (1, 704, 1280).

    Returns:
        numpy array: RGB segmentation mask of shape (704, 1280, 3).
    """
    segmentation_mask = segmentation_mask[0]  # Remove the batch dimension.
    assert segmentation_mask.max() < settings.COLOR_MAP.shape[0], "Class label exceeds COLOR_MAP range"
    assert settings.COLOR_MAP.shape[1] == 3, "COLOR_MAP must map labels to RGB"

    return settings.COLOR_MAP[segmentation_mask.astype(np.uint8)]


def apply_erosion(segmentation_mask: np.array) -> np.array:
    """
    Apply erosion to reduce the size of the segmented objects.

    Args:
        segmentation_mask (numpy array): RGB segmentation mask of shape (704, 1280, 3).

    Returns:
        numpy array: Eroded segmentation mask of shape (704, 1280, 3).
    """

    return cv2.erode(segmentation_mask, settings.EROSION_KERNEL, iterations=settings.EROSION_ITERATIONS)


def apply_dilation(segmentation_mask: np.array) -> np.array:
    """
    Apply dilation to increase the size of the segmented objects.

    Args:
        segmentation_mask (numpy array): RGB segmentation mask of shape (704, 1280, 3).

    Returns:
        numpy array: Dilated segmentation mask of shape (704, 1280, 3).
    """
    return cv2.dilate(segmentation_mask, settings.DILATION_KERNEL, iterations=settings.DILATION_ITERATIONS)


def apply_gaussian_smoothing(segmentation_mask: np.array) -> np.array:
    """
    Apply Gaussian smoothing to the segmentation mask.

    Args:
        segmentation_mask (numpy array): RGB segmentation mask of shape (704, 1280, 3).

    Returns:
        numpy array: Smoothed segmentation mask of shape (704, 1280, 3).
    """
    return cv2.GaussianBlur(segmentation_mask, settings.GAUSSIAN_SMOOTHING_KERNEL_SHAPE, 0)


def apply_median_filtering(segmentation_mask: np.array) -> np.array:
    """
    Apply median filtering to the segmentation mask.

    Args:
        segmentation_mask (numpy array): RGB segmentation mask of shape (704, 1280, 3).

    Returns:
        numpy array: Filtered segmentation mask of shape (704, 1280, 3).
    """
    return cv2.medianBlur(segmentation_mask, settings.MEDIAN_FILTERING_KERNEL_SIZE)


def apply_crf(original_image: np.array, segmentation_mask: np.array) -> np.array:
    """
    Apply Conditional Random Field (CRF) post-processing.

    Args:
        original_image (numpy array): Original image of shape (704, 1280, 3).
        segmentation_mask (numpy array): Segmentation mask of shape (1, 704, 1280).

    Returns:
        numpy array: Post-processed segmentation mask of shape (704, 1280, 3).
    """
    num_classes = settings.NUM_CLASSES
    dense_crf = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], num_classes)

    unary = unary_from_labels(segmentation_mask[0], num_classes, gt_prob=0.90, zero_unsure=False)
    dense_crf.setUnaryEnergy(unary)

    dense_crf.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    dense_crf.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image, compat=10,
                                   kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    inference = dense_crf.inference(5)
    mapped_inference_labels = np.argmax(inference, axis=0).reshape(original_image.shape[:2])
    return settings.COLOR_MAP[mapped_inference_labels]


def apply_mask_postprocessing(raw: np.array, segmentation_mask: np.array) -> tuple:
    """
    Apply post-processing operations to a segmentation mask.

    Args:
        raw (numpy array): Original image of shape (704, 1280, 3).
        segmentation_mask (numpy array): Segmentation mask of shape (1, 704, 1280).

    Returns:
        tuple: Processed raw image and segmentation mask.
    """
    print("Applying mask postprocessing...")

    mask_rgb = settings.COLOR_MAP[segmentation_mask]

    if ui_input_variables.EROSION_ON:
        mask_rgb = apply_erosion(mask_rgb)

    if ui_input_variables.DILATION_ON:
        mask_rgb = apply_dilation(mask_rgb)

    if ui_input_variables.GAUSSIAN_SMOOTHING_ON:
        mask_rgb = apply_gaussian_smoothing(mask_rgb)

    if ui_input_variables.MEDIAN_FILTERING_ON:
        mask_rgb = apply_median_filtering(mask_rgb)

    # if ui_input_variables.CRF_ON:
    #     mask = apply_crf(raw, segmentation_mask)

    print(f'Applying Postprocessing - beta...')
    print(mask_rgb.shape)

    # Remove the batch dimension (1,) from the input array
    # rgb_array = mask_rgb[0]
    #
    # # Initialize the output array to store the class labels
    # label_array = np.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.int32)
    #
    # for i in range(rgb_array.shape[0]):
    #     for j in range(rgb_array.shape[1]):
    #         # Get the RGB color of the pixel at position (i, j)
    #         rgb_tuple = tuple(rgb_array[i, j])
    #
    #         # Look up the label for this RGB value
    #         if rgb_tuple in settings.INVERTED_COLOR_MAP:
    #             label_array[i, j] = settings.INVERTED_COLOR_MAP[rgb_tuple]
    #         else:
    #             # In case the color is not found in the map (optional: can set to a default label)
    #             label_array[i, j] = -1  # or some other default value to indicate an unmapped color

    return raw, mask_rgb
