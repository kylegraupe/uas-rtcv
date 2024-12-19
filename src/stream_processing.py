import time
import ffmpeg
import numpy as np
import cv2
from PIL import Image
import queue
import sys

import settings
import model_inference
import mask_postprocessing
import ui_input_variables


def get_first_item_from_queue(buffer_queue):
    """
    Retrieves the first (oldest) item from the queue.

    Args:
        buffer_queue (queue.Queue): The queue containing frames.

    Returns:
        np.ndarray: The oldest frame in the queue.
    """
    if not buffer_queue.empty():
        return buffer_queue.get()  # Retrieve the oldest frame (FIFO)
    else:
        return None


def custom_buffer(frame, buffer_queue):
    """
    Takes in frame as a NumPy array and adds to queue.

    Args:
        frame (np.ndarray): A NumPy array of an image frame to be added to the queue.
        buffer_queue (queue.Queue): The queue where frames will be stored.
    """
    if buffer_queue.full():
        buffer_queue.get()  # Remove the oldest frame to make space for the new frame

    buffer_queue.put(frame)  # Add new frame to the queue
    return buffer_queue


def livestream_executive_ui(url, app):
    """
    Establishes a livestream connection to the provided URL, reads video frames,
    applies a segmentation model to the frames, and displays both the original
    and segmented frames side-by-side in the UI window.

    Args:
        url (str): The URL of the livestream to connect to.
        app: The StreamApp instance for updating the UI with video frames.

    Returns:
        None
    """

    print(f'Initiated FFmpeg process at time {time.ctime()}')
    # Set up FFmpeg process
    process = (
        ffmpeg
        .input(url, an=None)  # Disable audio
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{ui_input_variables.OUTPUT_FPS}')
        .global_args('-c:v', 'libx264', '-rtbufsize', '100k')
        .global_args('-preset', 'ultrafast', '-threads', '4')
        .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR)
    )
    print(f'FFmpeg process connected at time {time.ctime()}')

    app.process = process

    buffer_queue = queue.Queue(maxsize=settings.MAX_BUFFER_SIZE)
    print(f'Buffer Queue: ')
    print(buffer_queue)

    while app.is_streaming:
        # Read frame in byte format
        in_bytes = process.stdout.read(settings.FRAME_SIZE)

        # Check for end of stream or error
        if len(in_bytes) != settings.FRAME_SIZE:
            if not in_bytes:
                print("End of stream or error reading frame")
                break
            else:
                print("Error: Read incomplete frame")
                break

        # Convert the byte array to a numpy array representing the frame
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]).copy()

        # Add the frame to the buffer queue with the custom buffer logic
        buffer_queue = custom_buffer(in_frame, buffer_queue)
        print(f'Buffer queue size: {buffer_queue.qsize()}')

        # Retrieve the first (oldest) frame from the buffer
        buffer_frame = get_first_item_from_queue(buffer_queue)

        # print(f'Buffer queue size: {buffer_queue.qsize()}')

        if settings.MODEL_ON:

            # Resize the frame before inference
            buffer_frame_resized = cv2.resize(buffer_frame, (1280, 704), interpolation=cv2.INTER_NEAREST)

            # Pass the resized frame to the model
            segmentation_results_rgb = model_inference.image_to_tensor(
                Image.fromarray(buffer_frame_resized),
                settings.MODEL,
                settings.DEVICE
            ).astype(np.uint8)

            # Post-processing the mask
            _, segmentation_results = mask_postprocessing.apply_mask_postprocessing(buffer_frame_resized,
                                                                                    segmentation_results_rgb)

            # If side-by-side display is enabled, stack the original and segmented frames
            if settings.SIDE_BY_SIDE:
                output_frame = np.hstack((buffer_frame_resized.astype(np.uint8), segmentation_results.astype(np.uint8)))
            else:
                output_frame = segmentation_results
        else:
            output_frame = buffer_frame

        # Resize the output frame to fit the display window if necessary
        display_width = settings.VIDEO_DISPLAY_WIDTH
        display_height = settings.VIDEO_DISPLAY_HEIGHT

        if output_frame.shape[1] > display_width or output_frame.shape[0] > display_height:
            # Calculate the aspect ratio
            aspect_ratio = output_frame.shape[1] / output_frame.shape[0]
            if output_frame.shape[1] > display_width:
                new_width = display_width
                new_height = int(display_width / aspect_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * aspect_ratio)

            # Resize the frame to fit within the window
            output_frame = cv2.resize(output_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Update the UI display with the output frame
        app.update_video_display(output_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    process.stdout.close()
    process.wait()
    cv2.destroyAllWindows()
