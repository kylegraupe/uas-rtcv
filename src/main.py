"""
Main script for the livestream application.
"""

import time
import cProfile
import pstats
import tkinter as tk
import threading

import settings
import custom_logging
import stream_processing
import user_interface
import stream_processing_threaded


def execute_application() -> None:

    if settings.NONTHREADED_UI_ON:
        root = tk.Tk()
        app = user_interface.StreamApp(root,
                                       lambda: stream_processing.livestream_executive_ui(settings.RTMP_URL, app),
                                       lambda: app.stop_stream())
        root.mainloop()


    if settings.THREADED_IMPLEMENTATION:
        stream_processing_threaded.stream_processing_threaded_executive()


if __name__ == "__main__":
    custom_logging.log_event(f'Application started at time: {time.ctime()}\n'
                   f'\n'
                   f'\tApplication Environment Variables: \n'
                   f'\t\tEnvironment: {settings.ENVIRONMENT}\n'
                   f'\t\tRTMP URL: {settings.RTMP_URL}\n'
                   f'\t\tIP Address: {settings.IP_ADDRESS}\n'
                   f'\t\tListening Port: {settings.LISTENING_PORT}\n'
                   f'\t\tUI On: {settings.NONTHREADED_UI_ON}\n'
                   f'\t\tRun Tests Before Application Execution: {settings.RUN_TESTS_PRIOR_TO_EXECUTION}\n'
                   f'\t\tThreaded Implementation: {settings.THREADED_IMPLEMENTATION}\n'
                   f'\n'
                   f'\tStream Properties:\n'
                   f'\t\tInput Frames per Second: {settings.INPUT_FPS}\n'
                   f'\t\tOutput Frames per Second: {settings.OUTPUT_FPS}\n'
                   f'\t\tMax Buffer Size: {settings.MAX_BUFFER_SIZE}\n'
                   f'\n'
                   f'\tMask Postprocessing:\n'
                   f'\t\tDilation On: {settings.DILATION_ON}\n'
                   f'\t\tErosion On: {settings.EROSION_ON}\n'
                   f'\t\tMedian Filtering On: {settings.MEDIAN_FILTERING_ON}\n'
                   f'\t\tGaussian Smoothing On: {settings.GAUSSIAN_SMOOTHING_ON}\n'
                   f'\t\tConditional Random Field On: {settings.CRF_ON}'
                   f'\n')

    print("Function running on thread:", threading.current_thread().name)

    # if settings.SHOW_DEBUG_PROFILE:
    #     profiler = cProfile.Profile()
    #     profiler.enable()

    execute_application()

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats()
    # else:
        # execute_application()
