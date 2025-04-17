import ffmpeg
import time
import itertools
from pathlib import Path

import settings

def ffmpeg_gs():
    # Define test space
    codecs = ['libx264', 'libx265']
    presets = ['ultrafast', 'superfast', 'fast']
    bufsizes = ['100k', '500k']
    threads = ['1', '2', '4']
    frame_rates = ['15', '30']
    pix_fmts = ['bgr24', 'yuv420p']

    # Your input stream URL
    url = settings.RTMP_URL
    output_path = Path('/Users/kylegraupe/Documents/Programming/GitHub/Computer Vision Dataset Generator/real_time_semantic_segmentation_using_dji_drone/logs/test_logs')
    output_path.mkdir(exist_ok=True)

    # Iterate all combinations
    test_cases = itertools.product(codecs, presets, bufsizes, threads, frame_rates, pix_fmts)

    for i, (codec, preset, bufsize, thread, fps, pix_fmt) in enumerate(test_cases):
        try:
            start_time = time.time()
            process = (
                ffmpeg
                .input(url, an=None)
                .output('pipe:', format='rawvideo', pix_fmt=pix_fmt, r=fps)
                .global_args('-c:v', codec, '-rtbufsize', bufsize)
                .global_args('-preset', preset, '-threads', thread)
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            # Read a few frames to benchmark speed
            frames_to_test = 100
            frame_size = 1280 * 720 * 3  # assuming 720p and RGB
            for _ in range(frames_to_test):
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes:
                    break

            duration = time.time() - start_time
            fps_achieved = frames_to_test / duration

            print(f'Iteration {i}, {codec=}, {preset=}, {bufsize=}, {thread=}, {fps=}, {pix_fmt=}, {fps_achieved=}')

            # Log result
            with open(output_path / 'results.txt', 'a') as f:
                f.write(f"Test {i} | {codec=} {preset=} {bufsize=} {thread=} {fps=} {pix_fmt=} | Achieved FPS: {fps_achieved:.2f}\n")

            process.terminate()

        except Exception as e:
            with open(output_path / 'errors.txt', 'a') as f:
                f.write(f"Test {i} Failed: {e}\n")
            print(f'Iteration Failed {i}, {codec=}, {preset=}, {bufsize=}, {thread=}, {fps=}, {pix_fmt=}')





if __name__ == "__main__":
    ffmpeg_gs()