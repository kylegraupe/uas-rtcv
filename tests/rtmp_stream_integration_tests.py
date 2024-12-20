"""
RTMP Stream integration tests.
"""

import unittest
import threading
import queue
import numpy as np
import time
from src import settings
from src.stream_processing_threaded import (
    produce_livestream_buffer,
    consume_livestream_buffer,
    get_first_n_items_from_queue,
    BUFFER_QUEUE,
    DISPLAY_QUEUE,
    is_streaming,
)


class TestRTMPStreamIntegration(unittest.TestCase):
    def setUp(self):
        """
        Set up necessary preconditions for the tests.
        """
        self.test_rtmp_url = settings.RTMP_URL  # Make sure this is a valid RTMP stream URL
        self.buffer_queue = BUFFER_QUEUE
        self.display_queue = DISPLAY_QUEUE
        self.original_is_streaming = is_streaming

        self.producer_thread = threading.Thread(target=produce_livestream_buffer, args=(self.test_rtmp_url,))
        self.consumer_thread = threading.Thread(target=consume_livestream_buffer)

        while not self.buffer_queue.empty():
            self.buffer_queue.get()
        while not self.display_queue.empty():
            self.display_queue.get()

    def tearDown(self):
        """
        Clean up after the tests.
        """
        global is_streaming
        is_streaming = self.original_is_streaming

    def wait_with_timeout(self, condition_fn, timeout: int = 10):
        """
        Waits for a condition function to return True within the specified timeout.
        :param: timeout = numer of seconds before test fails due to timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_fn():
                return True
            time.sleep(0.1)
        self.fail("Test failed due to timeout.")

    def test_rtmp_stream_connection(self):
        """
        Test to ensure the RTMP stream generates frames; if not, provide an appropriate error message.
        """
        global is_streaming
        is_streaming = True
        self.producer_thread.start()
        self.consumer_thread.start()

        try:
            # Wait for frames to appear in the display queue
            self.wait_with_timeout(lambda: not self.display_queue.empty(), timeout=10)

            # Retrieve an item from the display queue
            display_item = self.display_queue.get()
            self.assertIsInstance(display_item, tuple, "Display queue item is not a tuple.")
            self.assertEqual(len(display_item), 2, "Display queue item does not contain two elements.")

            og_img, mask = display_item
            self.assertIsInstance(og_img, np.ndarray, "Original image is not a NumPy array.")
            self.assertIsInstance(mask, np.ndarray, "Mask is not a NumPy array.")
            self.assertEqual(og_img.shape, (704, 1280, 3), "Original image shape is incorrect.")
            self.assertEqual(mask.shape[:2], (704, 1280), "Mask shape is incorrect.")

        except AssertionError as e:
            # Handle failure and provide a meaningful message
            self.fail(
                f"RTMP stream test failed: {str(e)}. This likely indicates that the RTMP stream is not generating frames. "
                "Possible reasons: poor connection, no connection, or invalid RTMP URL."
            )

        finally:
            is_streaming = False
            self.producer_thread.join()
            self.consumer_thread.join()

    def test_producer_stream(self):
        """
        Test that the producer can generate frames and fill the buffer queue.
        """
        global is_streaming
        is_streaming = True
        self.producer_thread.start()

        self.wait_with_timeout(lambda: not self.buffer_queue.empty())

        frames = get_first_n_items_from_queue(self.buffer_queue, 5)
        self.assertGreater(len(frames), 0, "No frames retrieved from the buffer queue.")

        for frame in frames:
            self.assertIsInstance(frame, np.ndarray, "Frame is not a NumPy array.")
            self.assertEqual(frame.shape, (720, 1280, 3), "Frame shape is incorrect.")

        is_streaming = False
        self.producer_thread.join()

    def test_consumer_stream(self):
        """
        Test that the consumer can retrieve frames from the buffer queue and process them.
        """
        global is_streaming
        is_streaming = True
        self.producer_thread.start()
        self.consumer_thread.start()

        self.wait_with_timeout(lambda: not self.display_queue.empty())

        display_item = self.display_queue.get()
        self.assertIsInstance(display_item, tuple, "Display queue item is not a tuple.")
        self.assertEqual(len(display_item), 2, "Display queue item does not contain two elements.")

        og_img, mask = display_item
        self.assertIsInstance(og_img, np.ndarray, "Original image is not a NumPy array.")
        self.assertIsInstance(mask, np.ndarray, "Mask is not a NumPy array.")
        self.assertEqual(og_img.shape, (704, 1280, 3), "Original image shape is incorrect.")
        self.assertEqual(mask.shape[:2], (704, 1280), "Mask shape is incorrect.")

        is_streaming = False
        self.producer_thread.join()
        self.consumer_thread.join()

    def test_buffer_limits(self):
        """
        Test that the buffer queue does not exceed the maximum size.
        """
        global is_streaming
        is_streaming = True
        self.producer_thread.start()

        self.wait_with_timeout(lambda: self.buffer_queue.qsize() > 0)
        self.assertLessEqual(self.buffer_queue.qsize(), settings.MAX_BUFFER_SIZE, "Buffer queue exceeded max size.")

        is_streaming = False
        self.producer_thread.join()


if __name__ == "__main__":
    unittest.main()
