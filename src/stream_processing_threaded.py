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

from src import settings, model_inference, mask_postprocessing

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

    process = (
        ffmpeg
        .input(url, an=None)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{settings.INPUT_FPS}')
        .global_args('-c:v', 'libx264', '-bufsize', '2M')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    while is_streaming:
        in_bytes = process.stdout.read(settings.FRAME_SIZE)
        if len(in_bytes) != settings.FRAME_SIZE:
            if not in_bytes:
                print("End of stream or error reading frame")
                break
            else:
                print("Error: Read incomplete frame")
                break

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]).copy()
        add_to_buffer(in_frame, BUFFER_QUEUE)


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

            segmentation_result_batch = model_inference.images_to_tensor(
                frame_batch_resized,
                settings.MODEL,
                settings.DEVICE
            ).astype(np.uint8)

            # _, segmentation_results = mask_postprocessing.apply_mask_postprocessing(buffer_frame_resized,
            #                                                                         segmentation_results_rgb)

            if settings.SIDE_BY_SIDE:
                batch_tuple = (frame_batch_resized, segmentation_result_batch)

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
    global is_streaming
    while is_streaming:
        if not DISPLAY_QUEUE.empty():
            display_queue_size = DISPLAY_QUEUE.qsize()
            og_img, mask = DISPLAY_QUEUE.get()
            np_og_img = np.array(og_img)
            np_mask = np.array(mask)

            if mask is None:
                break

            num_images, height, width = np_mask.shape
            masks = [np_mask[i] for i in range(num_images)]

            for i in range(display_queue_size):
                if i >= np_mask.shape[0]:
                    print('Mismatch in image batches')
                    break

                og_img = np_og_img[i]
                mask_img = masks[i]
                combined_img = np.vstack((og_img, settings.COLOR_MAP[mask_img]))

                cv2.imshow("Frame", combined_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_streaming = False
            break


def stream_processing_threaded_executive() -> None:
    producer_thread = threading.Thread(target=produce_livestream_buffer, args=(settings.RTMP_URL,))
    consumer_thread = threading.Thread(target=consume_livestream_buffer)

    producer_thread.start()
    consumer_thread.start()

    display_video()

if __name__ == "__main__":
    stream_processing_threaded_executive()


    # producer_thread.join()
    # consumer_thread.join()
    # cv2.destroyAllWindows()