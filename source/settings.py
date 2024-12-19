"""
This file contains the settings and global variables for the application.
"""

import numpy as np
import cv2

import model_inference

# Application Environment
VERSION = '1.0.0'
ENVIRONMENT = 'development'
TRAIN = False
SHOW_DEBUG_PROFILE = True
UI_ON = False

# RTMP/NGINX settings
LISTENING_PORT=1935
ip_address = '10.0.0.105'
# ip_address = '172.20.10.2' # Hotspot
RTMP_URL=f'rtmp://{ip_address}:{LISTENING_PORT}/live/'

# Model properties
MODEL_PATH = '/Users/kylegraupe/Documents/Programming/GitHub/Computer Vision Dataset Generator/real_time_semantic_segmentation_using_dji_drone/trained_models/Unet-Mobilenet_V3.pt'
MODEL, DEVICE = model_inference.load_segmentation_model(MODEL_PATH)
MODEL_ENCODER_NAME = MODEL.encoder.__class__.__name__
MODEL_DECODER_NAME = MODEL.decoder.__class__.__name__
MODEL_ON = True
COLOR_MAP = np.array([
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
NUM_CHANNELS = 3 # RGB
BATCH_SIZE = 5
NUM_CLASSES = 23

# Stream properties
INPUT_FPS = 10 # Keep low when model is on, high when model is off. Too high will cause ffmpeg buffer to fill up.
OUTPUT_FPS = 0.1
NUM_THREADS = 4
MAX_BUFFER_SIZE = 5
PIPE_STDOUT = True
PIPE_STDERR = True

# Frame properties
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * NUM_CHANNELS

RESIZE_FRAME_WIDTH = 1280
RESIZE_FRAME_HEIGHT = 704 # U-Net architecture requires input dimensions to be divisible by 32.
VIDEO_DISPLAY_WIDTH = 1280  # Width of the video display area
VIDEO_DISPLAY_HEIGHT = 704  # Height of the video display area

# Font properties for text overlays
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_LOCATION = (10, 50)
SHAPE_LOCATION = (10, 75)
MODEL_DESCRIPTION_LOCATION = (10, 100)
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
THICKNESS = 1
LINE_TYPE = 2

# DJI Mini 4 Pro Specs
FOV = 82.1

# Output Properties
SIDE_BY_SIDE = True # Display both original and segmented frames side-by-side

# Postprocessing properties
DILATION_ON = False
DILATION_KERNEL = np.ones((5, 5), np.uint8)
DILATION_ITERATIONS = 1

EROSION_ON = False
EROSION_KERNEL = np.ones((5, 5), np.uint8)
EROSION_ITERATIONS = 1

MEDIAN_FILTERING_ON = False
MEDIAN_FILTERING_KERNEL_SIZE = 11

# GAUSSIAN_SMOOTHING_ON = False
GAUSSIAN_SMOOTHING_KERNEL_SHAPE = (5, 5)

CRF_ON = False

# Not Working
ACTIVE_CONTOURS_ON = False
WATERSHED_ON = False
CANNY_DETECTION_ON = False
SMALL_ITEM_FILTER_ON = False