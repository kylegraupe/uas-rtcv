import ffmpeg
import time
import itertools
from pathlib import Path
import time, subprocess
import torch
import time
import time
import torch
import numpy as np
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

def test_ffmpeg_stream():
    frame_count = 100
    frame_size = 1280 * 720 * 3

    start = time.time()

    process = (
        ffmpeg
        .input(settings.RTMP_URL, an=None)  # Disable audio
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{settings.INPUT_FPS}')
        .global_args('-c:v', 'libfdk_aac', '-rtbufsize', '10k')
        .global_args('-preset', 'ultrafast', '-threads', '4')
        .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR)
    )

    for _ in range(frame_count):
        raw = process.stdout.read(frame_size)
        if not raw:
            break

    end = time.time()
    print(f"RTMP throughput: {frame_count / (end - start):.2f} FPS")

    process.kill()

def test_model():
    # Create random tensor shaped like your input
    dummy_input = torch.randn(1, 3, 704, 1280).to("mps" if torch.backends.mps.is_available() else "cpu")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = torch.load(settings.MODEL_PATH, map_location=device)

    # Warm up
    for _ in range(10):
        _ = model(dummy_input)

    # Measure
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
    end = time.time()

    print(f"Model FPS: {50 / (end - start):.2f}")

if __name__ == "__main__":
    # ffmpeg_gs()
    # test_ffmpeg_stream()
    test_model()