"""
Unit tests for the src/settings.py file.
"""


import unittest
import os
from src import settings

def is_divisible(number, divisor):
        return number % divisor == 0

def device_is_in_list(device):
    return device in ['cpu','gpu']

def path_exists(path):
    return os.path.exists(path)

class TestSettingsViability(unittest.TestCase):
    """
    Class to test data cleaning methods using example cases.
    """

    def test_UNet_frame_resize_dimension_viability(self):
        """U-Net Architectures require frame dimensions must be divisible by 32. Frames are resized (trimmed) to meet this criterion."""
        self.assertTrue(is_divisible(settings.RESIZE_FRAME_HEIGHT, 32), msg=f'Frame resize height, {settings.RESIZE_FRAME_HEIGHT}, is not divisible by 32.')
        self.assertTrue(is_divisible(settings.RESIZE_FRAME_WIDTH, 32), msg=f'Frame resize width, {settings.RESIZE_FRAME_WIDTH} is not divisible by 32.')

    def test_input_frames_are_720p_rgb(self):
        """DJI Mini 4 Pro only streams via RTMP in 720p RGB."""
        self.assertEqual(settings.FRAME_WIDTH, 1280,
                         msg=f'Incorrect input frame width. Width: {settings.FRAME_WIDTH} is not a dimension of 720p.')
        self.assertEqual(settings.FRAME_HEIGHT, 720,
                         msg=f'Incorrect input frame height. Height: {settings.FRAME_WIDTH} is not a dimension of 720p.')
        self.assertEqual(settings.NUM_CHANNELS, 3,
                         msg=f'The number of channels used must be 3 for RGB images. {settings.NUM_CHANNELS} is an incorrect value.')

    def test_model_path_exists(self):
        """Ensure the application is pointing to a valid PyTorch Model."""
        self.assertTrue(path_exists(settings.MODEL_PATH), msg=f'Model Path: {settings.MODEL_PATH} does not exist.')

    def test_get_IP_function(self):
        """Test to ensure IP address does not return default IP of 127.0.0.1"""
        self.assertNotEqual(settings.ip_address, f'127.0.0.1', msg=f'IP Address: {settings.ip_address} has reverted to localhost value.')

    def test_pytorch_device(self):
        """Test to ensure that the device used in model training is either the CPU or GPU"""
        self.assertTrue(device_is_in_list(str(settings.DEVICE)), msg=f'Device: {str(settings.DEVICE)} not a valid device type.')

    def test_input_fps_greater_than_max_buffer_size(self):
        """Ensure that an appropriate input framerate is being used."""
        self.assertGreater(settings.INPUT_FPS, settings.MAX_BUFFER_SIZE, msg=f'Input FPS: {settings.INPUT_FPS} is not greater than the maximum buffer size: {settings.MAX_BUFFER_SIZE}')

    def test_num_classes_colormap_matchup(self):
        """Ensure the number of classes used is equal to the length of the colormap used in the segmentation mask."""
        self.assertEqual(settings.NUM_CLASSES, len(settings.COLOR_MAP),
                         msg=f'There is a discrepancy in the number of classes being used. Number of classes: {settings.NUM_CLASSES} is not equal to the length of the colormap: {len(settings.COLOR_MAP)}.')

    def test_resize_frame_dimension_equals_display_frame_dimensions(self):
        """Ensure the resized image is not being resized upon display."""
        self.assertEqual(settings.RESIZE_FRAME_WIDTH, settings.VIDEO_DISPLAY_WIDTH,
                         msg=f'Resized frame width: {settings.RESIZE_FRAME_WIDTH} is not equal to Display frame width: {settings.VIDEO_DISPLAY_WIDTH}.')
        self.assertEqual(settings.RESIZE_FRAME_HEIGHT, settings.VIDEO_DISPLAY_HEIGHT,
                         msg=f'Resized frame height: {settings.RESIZE_FRAME_HEIGHT} is not equal to Display frame height: {settings.VIDEO_DISPLAY_HEIGHT}.')