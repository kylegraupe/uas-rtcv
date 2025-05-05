"""
Threaded implementation of RTMP stream. Uses Producer-Consumer Paradigm to keep stream in near real-time.
"""

import time
import ffmpeg
import numpy as np
from PIL import Image
import queue
import threading
import pstats
import cProfile
import cv2
import os
import datetime
import settings
from src import model_inference, mask_postprocessing, custom_logging
import torch


ffmpeg_process = None
is_streaming: bool = True
BUFFER_QUEUE: queue.Queue = queue.Queue(maxsize=settings.MAX_BUFFER_SIZE)
DISPLAY_QUEUE: queue.Queue = queue.Queue(maxsize=5)

def get_first_n_items_from_queue(queue_param: queue.Queue, n: int) -> list:
    """
    Retrieves the first 'n' items from the queue.

    Args:
        queue_param (queue.Queue): The queue from which items are retrieved.
        n (int): The number of items to retrieve.

    Returns:
        list: A list of the first 'n' items from the queue.
    """
    items = []
    for _ in range(n):
        if not queue_param.empty():
            item = queue_param.get()
            items.append(item)
            queue_param.task_done()
        else:
            break
    return items

def add_to_buffer(frame: np.array, buffer_queue: queue.Queue) -> None:
    """
    Adds frame to buffer. Does not exceed buffer size to keep stream in near real-time.
    :param frame: Image frame to be added to the queue.
    :param buffer_queue: Buffer queue
    :return: None
    """

    if buffer_queue.full():
        buffer_queue.get()
    buffer_queue.put(frame)

def produce_livestream_buffer(url: str) -> None:
    """
    Produces livestream buffer using FFMPEG and custom buffer in Producer-Consumer Threading Paradigm.
    :param url: RTMP URL (in settings.py)
    :return: None
    """
    global ffmpeg_process

    ffmpeg_process = (
        ffmpeg
        .input(url, an=None)  # Disable audio
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{settings.INPUT_FPS}')
        .global_args('-c:v', 'libfdk_aac', '-rtbufsize', '500k')
        .global_args('-preset', 'ultrafast', '-threads', '4')
        .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR)
    )

    try:
        while is_streaming:
            in_bytes = ffmpeg_process.stdout.read(settings.FRAME_SIZE)
            if len(in_bytes) != settings.FRAME_SIZE:
                print("Stream ended or broken frame read.")
                break
            in_frame = np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]).copy()
            add_to_buffer(in_frame, BUFFER_QUEUE)
    finally:
        # Ensure the process is killed when the loop exits
        if ffmpeg_process:
            ffmpeg_process.stdout.close()
            ffmpeg_process.stderr.close()
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=2)
            except:
                ffmpeg_process.kill()

def consume_livestream_buffer() -> None:
    """
    Consumes livestream buffer, outputs to display queue for visualization. Consumer function in Producer-Consumer Threading Paradigm.
    :return: None
    """
    time.sleep(2)

    while is_streaming:
        frame_batch = get_first_n_items_from_queue(BUFFER_QUEUE, 1)
        frame_batch_resized = []

        if settings.MODEL_ON:
            for frame in frame_batch:
                frame = cv2.resize(frame, (1280, 704), interpolation=cv2.INTER_NEAREST)
                frame_img = Image.fromarray(frame)
                frame_batch_resized.append(frame_img)

            segmentation_result_batch = model_inference.generate_segmentation_labels_np(
                frame_batch_resized,
                settings.MODEL,
                settings.DEVICE
            ).astype(np.uint8)

            _, segmentation_result_batch_processed = mask_postprocessing.apply_mask_postprocessing(frame_batch_resized,
                                                                                 segmentation_result_batch)

            if settings.SIDE_BY_SIDE:
                batch_tuple = (frame_batch_resized, segmentation_result_batch_processed)

                DISPLAY_QUEUE.put(batch_tuple)
            else:
                output_batch = segmentation_result_batch
        else:
            break

def save_batch_as_png(image, mask, save_directory, index, prefix='image'):
    """
    Saves each image in the batch as a PNG file in the specified directory.

    Args:
        images (list of np.ndarray): List of images to save.
        masks (list of np.ndarray): List of masks corresponding to the images.
        save_directory (str): Directory path where the images will be saved.
        prefix (str): Prefix for the filename of each image.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

        image_filename = os.path.join(save_directory, f'{prefix}_{index}_image.png')
        mask_filename = os.path.join(save_directory, f'{prefix}_{index}_mask.png')

        # Save images
        cv2.imwrite(image_filename, image)
        cv2.imwrite(mask_filename, mask)

        print(f'Saved {image_filename} and {mask_filename}')

def display_video() -> None:
    """
    Displays live stream and model mask to OpenCV window.
    :return: None
    """

    def display_stacked_frame(img_1, img_2):
        combined_img = np.vstack((og_img, mask_img))

        cv2.imshow("Frame", combined_img)

    def display_image(image: np.ndarray):
        """
        Display image using OpenCV.

        Args:
            window_name (str): Window name
            image (np.ndarray): Image to display
        """
        cv2.imshow('Filtered Class Overlay on Original Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def overlay_segmentation(original_image: np.ndarray, mask_image: np.ndarray, color_map: np.ndarray, alpha=0.5,
                             class_filter=None):
        """
        Overlay segmentation mask on the original image using OpenCV.

        Args:
            original_image (np.ndarray): Original image (H, W, 3), RGB, uint8
            mask_image (np.ndarray): Segmentation mask (H, W), integer class labels
            color_map (np.ndarray): (num_classes, 3) color map (RGB)
            alpha (float): Transparency of overlay [0.0 - 1.0]
            class_filter (list, optional): List of class indices to filter and display. If None, shows all classes.

        Returns:
            overlayed_image (np.ndarray): Image with overlay (BGR, uint8)
        """

        # Validate input
        if original_image.dtype != np.uint8:
            raise ValueError("Original image must be dtype=np.uint8")
        if original_image.shape[-1] != 3:
            raise ValueError("Original image must have 3 channels (H, W, 3)")

        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        color_mask_bgr = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
        overlayed_image = cv2.addWeighted(original_bgr, 1 - alpha, color_mask_bgr, alpha, 0)

        return overlayed_image

    global is_streaming, ffmpeg_process

    start_time = datetime.datetime.now()
    frame_counter = 0

    while is_streaming:
        if not DISPLAY_QUEUE.empty():
            display_queue_size = DISPLAY_QUEUE.qsize()
            og_img, mask = DISPLAY_QUEUE.get()
            np_og_img, np_mask = np.array(og_img), np.array(mask)
            # np_mask = np.array(mask)

            if mask is None:
                break

            num_images, height, width, num_channels = np_mask.shape
            # masks = [np_mask[i] for i in range(num_images)]

            for i in range(display_queue_size):
                if i >= np_mask.shape[0]:
                    print('Mismatch in image batches')
                    break

                og_img, mask_img = np_og_img[i], mask[i]

                if settings.STACKED_DISPLAY:
                    display_stacked_frame(og_img, mask_img)

                if settings.OVERLAY_DISPLAY:
                    overlayed_img = overlay_segmentation(og_img, mask_img, settings.COLOR_MAP)
                    cv2.imshow("Frame", overlayed_img)

                frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_streaming = False
            break

    # Cleanup ffmpeg process
    if ffmpeg_process:
        try:
            ffmpeg_process.stdout.close()
            ffmpeg_process.stderr.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=2)
        except:
            ffmpeg_process.kill()

    cv2.destroyAllWindows()

    # Logging
    end_time = datetime.datetime.now()
    total_stream_time_seconds = (end_time - start_time).total_seconds()
    recorded_fps = frame_counter / total_stream_time_seconds
    data_dict = {
        'recorded_fps': recorded_fps,
        'recorded_frames': frame_counter,
        'total_stream_time': total_stream_time_seconds,
        'input_fps': settings.INPUT_FPS,
        'output_fps': settings.OUTPUT_FPS,
        'ffmpeg_num_threads': settings.NUM_THREADS,
        'custom_buffer_max_buffer_size': settings.MAX_BUFFER_SIZE,
        'dilation_on': settings.DILATION_ON,
        'erosion_on': settings.EROSION_ON,
        'median_filtering_on': settings.MEDIAN_FILTERING_ON,
        'gaussian_smoothing_on': settings.GAUSSIAN_SMOOTHING_ON
    }

    custom_logging.log_event(f'Total Stream Time: {total_stream_time_seconds} seconds')
    custom_logging.log_event(f'Total Frames: {frame_counter}')
    custom_logging.log_event(f'Recorded FPS: {recorded_fps}')
    custom_logging.append_to_log_data(data_dict, 'recorded_stream_data.csv')

def stream_processing_threaded_executive() -> None:
    producer_thread = threading.Thread(target=produce_livestream_buffer, args=(settings.RTMP_URL,))
    consumer_thread = threading.Thread(target=consume_livestream_buffer)

    producer_thread.start()
    consumer_thread.start()

    display_video()

if __name__ == "__main__":
    stream_processing_threaded_executive()