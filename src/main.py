"""
Main script for the livestream application.
"""

import time
import cProfile
import pstats
import tkinter as tk

import settings
import stream_processing
import user_interface
import stream_processing_threaded

def stop_stream_callback():
    # Logic to stop the FFmpeg process or any other cleanup
    print("Stopping the stream...")
    # Example: setting a flag or calling a function to terminate the stream
    # This can be further customized based on how you manage the stream


def main():
    print(f'Application started at time {time.ctime()}')
    print(f'\tVersion: {settings.VERSION}')
    print(f'\tEnvironment: {settings.ENVIRONMENT}')
    print(f'\tRTMP URL: {settings.RTMP_URL}')
    print(f'\tIP Address: {settings.ip_address}')
    print(f'\tListening Port: {settings.LISTENING_PORT}')

    if settings.UI_ON:
        root = tk.Tk()
        app = user_interface.StreamApp(root,
                                       lambda: stream_processing.livestream_executive_ui(settings.RTMP_URL, app),
                                       lambda: app.stop_stream())
        root.mainloop()
    else:
        if settings.THREADED_IMPLEMENTATION:
            stream_processing_threaded.stream_processing_threaded_executive()




if __name__ == "__main__":

    if settings.SHOW_DEBUG_PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

        main()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
    else:
        main()
