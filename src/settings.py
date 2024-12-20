"""
This file contains the settings and global variables for the application.
"""

import numpy as np
import cv2
import socket

from src import model_inference

def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception as e:
        IP = '127.0.0.1' # Default localhost IP
    finally:
        s.close()
    return IP

# Application Environment
ENVIRONMENT: str = 'development'
TRAIN: bool = False
SHOW_DEBUG_PROFILE: bool = True
UI_ON: bool = False
RUN_TESTS_PRIOR_TO_EXECUTION: bool = True

# RTMP/NGINX settings
LISTENING_PORT: int = 1935
IP_ADDRESS: str = get_ip()
RTMP_URL: str = f'rtmp://{IP_ADDRESS}:{LISTENING_PORT}/live/'

# Model properties
MODEL_PATH: str = '../trained_models/Unet-Mobilenet_V3.pt'
MODEL, DEVICE = model_inference.load_segmentation_model(MODEL_PATH)
MODEL_ENCODER_NAME = MODEL.encoder.__class__.__name__
MODEL_DECODER_NAME = MODEL.decoder.__class__.__name__
MODEL_ON: bool = True
COLOR_MAP: np.array = np.array([
    [0, 0, 0],        # Class 0: black
    [128, 0, 0],      # Class 1: dark red
    [0, 128, 0],      # Class 2: dark green
    [128, 128, 0],    # Class 3: dark yellow
    [0, 0, 128],      # Class 4: dark blue
    [128, 0, 128],    # Class 5: dark purple
    [0, 128, 128],    # Class 6: dark cyan
    [128, 128, 128],  # Class 7: gray
    [64, 0, 0],       # Class 8: maroon
    [192, 0, 0],      # Class 9: red
    [64, 128, 0],     # Class 10: olive
    [192, 128, 0],    # Class 11: orange
    [64, 0, 128],     # Class 12: purple
    [192, 0, 128],    # Class 13: magenta
    [64, 128, 128],   # Class 14: teal
    [192, 128, 128],  # Class 15: light gray
    [0, 64, 0],       # Class 16: dark green
    [128, 64, 0],     # Class 17: brown
    [0, 192, 0],      # Class 18: lime
    [128, 192, 0],    # Class 19: chartreuse
    [0, 64, 128],     # Class 20: navy
    [128, 64, 128],   # Class 21: medium purple
    [0, 192, 128],    # Class 22: aquamarine
], dtype=np.uint8)
NUM_CHANNELS: int = 3 # RGB
BATCH_SIZE: int = 5
NUM_CLASSES: int = 23

# Stream properties
INPUT_FPS: float = 10 # Keep low when model is on, high when model is off. Too high will cause ffmpeg buffer to fill up.
OUTPUT_FPS: float = 0.1
NUM_THREADS: int = 4
MAX_BUFFER_SIZE: int = 5
PIPE_STDOUT: bool = True
PIPE_STDERR: bool = True
THREADED_IMPLEMENTATION = True

# Frame properties
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
FRAME_SIZE: int = FRAME_WIDTH * FRAME_HEIGHT * NUM_CHANNELS

RESIZE_FRAME_WIDTH: int = 1280
RESIZE_FRAME_HEIGHT: int = 704 # U-Net architecture requires input dimensions to be divisible by 32.
VIDEO_DISPLAY_WIDTH: int = 1280  # Width of the video display area
VIDEO_DISPLAY_HEIGHT: int = 704  # Height of the video display area

# Font properties for text overlays
FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FPS_LOCATION: tuple = (10, 50)
SHAPE_LOCATION: tuple = (10, 75)
MODEL_DESCRIPTION_LOCATION: tuple = (10, 100)
FONT_SCALE: int = 1
FONT_COLOR: tuple = (255, 255, 255)
THICKNESS: int = 1
LINE_TYPE: int = 2

# DJI Mini 4 Pro Specs
FOV: float = 82.1

# Output Properties
SIDE_BY_SIDE: bool = True # Display both original and segmented frames side-by-side

# Postprocessing properties
DILATION_ON: bool = False
DILATION_KERNEL: np.array = np.ones((5, 5), np.uint8)
DILATION_ITERATIONS: int = 1

EROSION_ON: bool = False
EROSION_KERNEL: np.array = np.ones((5, 5), np.uint8)
EROSION_ITERATIONS: int = 1

MEDIAN_FILTERING_ON: bool = False
MEDIAN_FILTERING_KERNEL_SIZE: int = 11

GAUSSIAN_SMOOTHING_ON = False
GAUSSIAN_SMOOTHING_KERNEL_SHAPE: tuple = (5, 5)

CRF_ON: bool = False

# Not Working
ACTIVE_CONTOURS_ON: bool = False
WATERSHED_ON: bool = False
CANNY_DETECTION_ON: bool = False
SMALL_ITEM_FILTER_ON: bool = False