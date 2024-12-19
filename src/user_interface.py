"""
StreamApp is a class that creates a GUI for a RTMP livestream using tkinter.

The class takes in a root window, a start_stream_callback function, and a stop_stream_callback function.

The start_stream_callback function is called when the start button is pressed. It should start the livestream.

The stop_stream_callback function is called when the stop button is pressed. It should stop the livestream.

The class will create a GUI with buttons to start and stop the stream. It will also create a label to display the video.
"""

import tkinter as tk
from PIL import Image, ImageTk
import threading
import settings
import cv2
import ui_input_variables


class StreamApp:
    def __init__(self, root, start_stream_callback, stop_stream_callback):
        # Set the theme colors
        self.bg_color = "#1e1e2e"  # Dark purple
        self.fg_color = "#ffffff"  # White
        self.button_color = "#5a5a8b"  # Dark purple-grey
        self.button_text_color = "#000000"  # Black
        self.entry_bg_color = "#2b2b3c"  # Slightly lighter purple
        self.entry_fg_color = "#ffffff"  # White
        self.video_label_bg = "#000000"  # Black for video background

        self.root = root
        self.root.title("RTMP Stream GUI")
        self.start_stream_callback = start_stream_callback
        self.stop_stream_callback = stop_stream_callback
        # self.stream_thread = None
        self.is_streaming = False
        self.process = None

        # Set initial window size to the screen's resolution
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")

        # Enable resizing for macOS full-screen functionality
        self.root.resizable(True, True)

        # Allow full screen with the green button and bind the Escape key to exit full screen
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<Escape>', self.exit_fullscreen)

        # Apply the background color to the window
        self.root.configure(bg=self.bg_color)

        # Create the sidebar frame
        self.sidebar_frame = tk.Frame(self.root, bg=self.bg_color, width=200, padx=10, pady=10)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="ns")

        # Create the input settings frame
        self.input_frame = tk.Frame(self.root, bg=self.bg_color, bd=2, relief="raised")
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create the button frame
        self.button_frame = tk.Frame(self.root, bg=self.bg_color, bd=2, relief="raised")
        self.button_frame.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Create the video display frame
        self.video_frame = tk.Frame(self.root, bg=self.video_label_bg, bd=2, relief="raised")
        self.video_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Create the stream characteristics frame
        self.stream_char_frame = tk.Frame(self.root, bg=self.bg_color, bd=2, relief="raised")
        self.stream_char_frame.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        # Create the grid layout
        self.create_input_settings()
        self.create_buttons()
        self.create_video_display()
        self.create_stream_characteristics()
        self.load_logo()

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

    def load_logo(self):
        """
        Load and display the logo.

        This function loads the logo image and displays it in the user interface.
        The image is resized to 250x150 pixels and placed in the third row, first column of the UI layout.
        The logo is displayed with a solid border and margin of 20 pixels.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logo_path = "/Users/kylegraupe/Documents/Programming/GitHub/Computer Vision Dataset Generator/real_time_semantic_segmentation_using_dji_drone/assets/graupe.io logo 1.png"
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((250, 150))  # Adjust size as needed
        self.logo_imgtk = ImageTk.PhotoImage(logo_image)

        # Create a label for the logo
        self.logo_label = tk.Label(self.root, image=self.logo_imgtk, bg=self.bg_color, bd=2, relief="solid")

        # Position the logo in the third row, first column
        self.logo_label.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")  # Margin of 20 pixels

    def create_input_settings(self):
        """
        Creates a frame for displaying the input settings for various parameters.

        The frame is split into two columns with the first column displaying the parameter names
        and the second column displaying the parameter values. The parameter values are displayed as
        Entry widgets. The user can edit the parameter values by typing into the Entry widgets.

        Additionally, this frame includes toggle buttons for enabling or disabling post-processing methods.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Title for the input settings frame
        tk.Label(self.input_frame, text="Input Settings", bg=self.bg_color, fg=self.fg_color,
                 font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

        # Output FPS setting
        tk.Label(self.input_frame, text="Output FPS:", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0,
                                                                                                sticky="w")
        self.output_fps_entry = tk.Entry(self.input_frame, bg=self.entry_bg_color, fg=self.entry_fg_color)
        self.output_fps_entry.insert(0, str(settings.OUTPUT_FPS))
        self.output_fps_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Toggle buttons for post-processing methods
        tk.Label(self.input_frame, text="Post-Processing Methods", bg=self.bg_color, fg=self.fg_color,
                 font=("Arial", 14)).grid(row=2, column=0, columnspan=2, pady=10)

        # CRF Toggle Button
        self.crf_var = tk.BooleanVar(value=ui_input_variables.CRF_ON)
        self.crf_toggle = tk.Checkbutton(self.input_frame, text="Enable Conditional Random Field",
                                         variable=self.crf_var, bg=self.bg_color,
                                         fg=self.fg_color, command=self.toggle_crf)
        self.crf_toggle.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Erosion Toggle Button
        self.erosion_var = tk.BooleanVar(value=ui_input_variables.EROSION_ON)
        self.erosion_toggle = tk.Checkbutton(self.input_frame, text="Enable Erosion", variable=self.erosion_var,
                                             bg=self.bg_color, fg=self.fg_color, command=self.toggle_erosion)
        self.erosion_toggle.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Dilation Toggle Button
        self.dilation_var = tk.BooleanVar(value=ui_input_variables.DILATION_ON)
        self.dilation_toggle = tk.Checkbutton(self.input_frame, text="Enable Dilation", variable=self.dilation_var,
                                              bg=self.bg_color, fg=self.fg_color, command=self.toggle_dilation)
        self.dilation_toggle.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Gaussian Smoothing Toggle Button
        self.gaussian_smoothing_var = tk.BooleanVar(value=ui_input_variables.GAUSSIAN_SMOOTHING_ON)
        self.gaussian_smoothing_toggle = tk.Checkbutton(self.input_frame, text="Enable Gaussian Smoothing",
                                                        variable=self.gaussian_smoothing_var, bg=self.bg_color,
                                                        fg=self.fg_color, command=self.toggle_gaussian_smoothing)
        self.gaussian_smoothing_toggle.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Gaussian Smoothing Toggle Button
        self.median_smoothing_var = tk.BooleanVar(value=ui_input_variables.MEDIAN_FILTERING_ON)
        self.median_smoothing_toggle = tk.Checkbutton(self.input_frame, text="Enable Median Smoothing",
                                                      variable=self.median_smoothing_var, bg=self.bg_color,
                                                      fg=self.fg_color, command=self.toggle_median_smoothing)

        self.median_smoothing_toggle.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")

    def update_stream_characteristics(self):
        """
        Updates the stream characteristics frame to reflect the current state of the post-processing methods.
        """
        pos_highlight_color = "medium sea green"
        neg_highlight_color = "indian red"

        # Update the post-processing section of the stream characteristics frame
        for widget in self.stream_char_frame.winfo_children():
            widget.destroy()  # Clear the existing labels

        # Recreate the labels with updated colors
        frame_titles_fg = 'black'
        frame_titles_bg = 'light slate gray'

        # Stream characteristics
        tk.Label(self.stream_char_frame, text="STREAM CHARACTERISTICS:", underline=True, bg=frame_titles_bg,
                 fg=frame_titles_fg).grid(row=0, column=0, sticky="w")
        tk.Label(self.stream_char_frame,
                 text=f"- Resolution: {settings.RESIZE_FRAME_WIDTH}x{settings.RESIZE_FRAME_HEIGHT}", bg='bisque4',
                 fg=self.fg_color).grid(row=1, column=0, sticky="w")
        tk.Label(self.stream_char_frame, text=f"- Listening Port: {str(settings.LISTENING_PORT)}", bg='bisque4',
                 fg=self.fg_color).grid(row=2, column=0, sticky="w")
        tk.Label(self.stream_char_frame, text=f"- Input FPS: {str(settings.INPUT_FPS)}", bg='bisque4',
                 fg=self.fg_color).grid(row=3, column=0, sticky="w")
        tk.Label(self.stream_char_frame, text=f"- Output FPS: {str(settings.OUTPUT_FPS)}", bg='bisque4',
                 fg=self.fg_color).grid(row=4, column=0, sticky="w")

        # Model characteristics
        tk.Label(self.stream_char_frame, text="MODEL CHARACTERISTICS:", underline=True, bg=frame_titles_bg,
                 fg=frame_titles_fg).grid(row=5, column=0, sticky="w")
        tk.Label(self.stream_char_frame, text=f"- Encoder: {str(settings.MODEL_ENCODER_NAME)}", bg='bisque4',
                 fg=self.fg_color).grid(row=6, column=0, sticky="w")
        tk.Label(self.stream_char_frame, text=f"- Decoder: {str(settings.MODEL_DECODER_NAME)}", bg='bisque4',
                 fg=self.fg_color).grid(row=7, column=0, sticky="w")

        # RGB Mask Post-Processing
        tk.Label(self.stream_char_frame, text="RGB MASK POST-PROCESSING:", underline=True, bg=frame_titles_bg,
                 fg=frame_titles_fg).grid(row=8, column=0, sticky="w")

        # Post Processing
        dilation_bg = pos_highlight_color if ui_input_variables.DILATION_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Dilation: {str(ui_input_variables.DILATION_ON).upper()}",
                 bg=dilation_bg, fg=self.fg_color).grid(row=9, column=0, sticky="w")

        erosion_bg = pos_highlight_color if ui_input_variables.EROSION_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Erosion: {str(ui_input_variables.EROSION_ON).upper()}", bg=erosion_bg,
                 fg=self.fg_color).grid(row=10, column=0, sticky="w")

        median_blur_bg = pos_highlight_color if ui_input_variables.MEDIAN_FILTERING_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Median Blur: {str(ui_input_variables.MEDIAN_FILTERING_ON).upper()}",
                 bg=median_blur_bg, fg=self.fg_color).grid(row=11, column=0, sticky="w")

        gaussian_blur_bg = pos_highlight_color if ui_input_variables.GAUSSIAN_SMOOTHING_ON else neg_highlight_color
        tk.Label(self.stream_char_frame,
                 text=f"- Gaussian Blur: {str(ui_input_variables.GAUSSIAN_SMOOTHING_ON).upper()}", bg=gaussian_blur_bg,
                 fg=self.fg_color).grid(row=12, column=0, sticky="w")

        crf_bg = pos_highlight_color if ui_input_variables.CRF_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Conditional Random Field: {str(ui_input_variables.CRF_ON).upper()}",
                 bg=crf_bg, fg=self.fg_color).grid(row=13, column=0, sticky="w")

    def toggle_crf(self):

        ui_input_variables.CRF_ON = self.crf_var.get()
        print(f"CRF set to {ui_input_variables.CRF_ON}")
        self.update_stream_characteristics()

    def toggle_erosion(self):
        ui_input_variables.EROSION_ON = self.erosion_var.get()
        print(f"Erosion set to {ui_input_variables.EROSION_ON}")
        self.update_stream_characteristics()

    def toggle_dilation(self):
        ui_input_variables.DILATION_ON = self.dilation_var.get()
        print(f"Dilation set to {ui_input_variables.DILATION_ON}")
        self.update_stream_characteristics()

    def toggle_gaussian_smoothing(self):
        ui_input_variables.GAUSSIAN_SMOOTHING_ON = self.gaussian_smoothing_var.get()
        print(f"Gaussian Smoothing set to {ui_input_variables.GAUSSIAN_SMOOTHING_ON}")
        self.update_stream_characteristics()

    def toggle_median_smoothing(self):
        ui_input_variables.MEDIAN_FILTERING_ON = self.median_smoothing_var.get()
        print(f"Gaussian Smoothing set to {ui_input_variables.MEDIAN_FILTERING_ON}")
        self.update_stream_characteristics()

    def create_buttons(self):
        # Set the button width to be slightly less than the frame width
        """
        Create the buttons for the user interface.

        This function creates the buttons for the user interface. These buttons are:

        1. Execute button: This button starts the stream based on the user input settings.
        2. Stop button: This button stops the stream if it is running.
        3. Close App button: This button stops the stream and exits the application.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        button_width = 30  # Adjust this value as needed for your layout

        # Execute button
        self.execute_button = tk.Button(self.button_frame, text="Start Stream", command=self.start_stream,
                                        bg=self.button_color, fg=self.button_text_color, bd=1, relief="raised",
                                        width=button_width)
        self.execute_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")  # Use sticky="ew" to center

        # Stop button
        self.stop_button = tk.Button(self.button_frame, text="Stop Stream", command=self.stop_stream,
                                     bg=self.button_color, fg=self.button_text_color, state=tk.DISABLED, bd=1,
                                     relief="raised", width=button_width)
        self.stop_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")  # Use sticky="ew" to center

        # Close App button
        # self.close_button = tk.Button(self.button_frame, text="Close App", command=self.exit_and_close,
        #                               bg=self.button_color, fg=self.button_text_color, bd=1, relief="raised",
        #                               width=button_width)
        # self.close_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")  # Use sticky="ew" to center

    def create_video_display(self):
        # Video display label (for the video frame)
        """
        Creates a label for displaying the video frames.

        Creates a Label widget inside the video frame with a black background color.
        The label is packed into the video frame to fill both horizontally and vertically,
        and is allowed to expand into any extra space available in the video frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.video_label = tk.Label(self.video_frame, bg=self.video_label_bg)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def start_stream(self):
        """
        Starts the livestream based on user input settings.

        Updates the output FPS setting and potentially other settings based on user input.
        Then, starts the livestream in a new thread if it is not already streaming.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Update settings based on user inputs
        settings.OUTPUT_FPS = float(self.output_fps_entry.get())
        # Update other settings similarly

        # Start the livestream in a new thread
        if not self.is_streaming:
            self.is_streaming = True
            self.stop_button.config(state=tk.NORMAL)
            self.stream_thread = threading.Thread(target=self.start_stream_callback)
            self.stream_thread.start()

    def create_stream_characteristics(self):
        """
        Creates a frame for displaying the stream characteristics (e.g. resolution, bitrate)
        and model characteristics (e.g. encoder, decoder). The frame is split into three columns
        with the first column displaying the stream characteristics, the second column displaying
        the model characteristics, and the third column displaying the post-processing operations
        (dilation, erosion, median blur, gaussian blur, and conditional random field).

        The stream characteristics are displayed as labels with the text color set to self.fg_color
        and the background color set to 'bisque4'.

        The model characteristics are displayed as labels with the text color set to self.fg_color
        and the background color set to 'bisque4'.

        The post-processing operations are displayed as labels with the text color set to self.fg_color
        and the background color set to 'medium sea green' if the operation is enabled, or 'indian red'
        if the operation is disabled.
        """
        frame_titles_fg = 'black'
        frame_titles_bg = 'light slate gray'
        # Stream characteristics (e.g., resolution, bitrate)
        tk.Label(self.stream_char_frame, text="STREAM CHARACTERISTICS:", underline=True, bg=frame_titles_bg, fg=frame_titles_fg).grid(row=0,
                                                                                                                  column=0,
                                                                                                                  sticky="w")

        # Add stream characteristics labels and values
        tk.Label(self.stream_char_frame, text=f"- Resolution: {settings.RESIZE_FRAME_WIDTH}x{settings.RESIZE_FRAME_HEIGHT}"
                 , bg='bisque4', fg=self.fg_color).grid(row=1, column=0, sticky="w")

        tk.Label(self.stream_char_frame, text=f"- Listening Port: {str(settings.LISTENING_PORT)}", bg='bisque4', fg=self.fg_color).grid(row=2,
                                                                                                             column=0,
                                                                                                             sticky="w")
        tk.Label(self.stream_char_frame, text=f"- Input FPS: {str(settings.INPUT_FPS)}",
                 bg='bisque4', fg=self.fg_color).grid(row=3, column=0, sticky="w")

        tk.Label(self.stream_char_frame, text=f"- Output FPS: {str(settings.OUTPUT_FPS)}",
                 bg='bisque4', fg=self.fg_color).grid(row=4, column=0, sticky="w")

        # Model characteristics
        tk.Label(self.stream_char_frame, text="MODEL CHARACTERISTICS:", underline=True, bg=frame_titles_bg, fg=frame_titles_fg).grid(row=5,
                                                                                                                  column=0,
                                                                                                                  sticky="w")
        tk.Label(self.stream_char_frame, text=f"- Encoder: {str(settings.MODEL_ENCODER_NAME)}",
                 bg='bisque4', fg=self.fg_color).grid(row=6, column=0, sticky="w")

        tk.Label(self.stream_char_frame, text=f"- Decoder: {str(settings.MODEL_DECODER_NAME)}",
                 bg='bisque4', fg=self.fg_color).grid(row=7, column=0, sticky="w")

        tk.Label(self.stream_char_frame, text="RGB MASK POST-PROCESSING:", underline=True, bg=frame_titles_bg, fg=frame_titles_fg).grid(row=8,
                                                                                                                  column=0, sticky="w")

        # Post Processing
        pos_highlight_color = "medium sea green"
        neg_highlight_color = "indian red"
        # Dilation
        dilation_bg = pos_highlight_color if ui_input_variables.DILATION_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Dilation: {str(ui_input_variables.DILATION_ON).upper()}",
                 bg=dilation_bg, fg=self.fg_color).grid(row=9, column=0, sticky="w")

        # Erosion
        erosion_bg = pos_highlight_color if ui_input_variables.EROSION_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Erosion: {str(ui_input_variables.EROSION_ON).upper()}",
                 bg=erosion_bg, fg=self.fg_color).grid(row=10, column=0, sticky="w")

        # Median Blur
        median_blur_bg = pos_highlight_color if ui_input_variables.MEDIAN_FILTERING_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Median Blur: {str(ui_input_variables.MEDIAN_FILTERING_ON).upper()}",
                 bg=median_blur_bg, fg=self.fg_color).grid(row=11, column=0, sticky="w")

        # Gaussian Blur
        gaussian_blur_bg = pos_highlight_color if ui_input_variables.GAUSSIAN_SMOOTHING_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Gaussian Blur: {str(ui_input_variables.GAUSSIAN_SMOOTHING_ON).upper()}",
                 bg=gaussian_blur_bg, fg=self.fg_color).grid(row=12, column=0, sticky="w")

        # Conditional Random Field (CRF)
        crf_bg = pos_highlight_color if ui_input_variables.CRF_ON else neg_highlight_color
        tk.Label(self.stream_char_frame, text=f"- Conditional Random Field: {str(ui_input_variables.CRF_ON).upper()}",
                 bg=crf_bg, fg=self.fg_color).grid(row=13, column=0, sticky="w")

    def stop_stream(self):
        if self.is_streaming:
            self.is_streaming = False
            if self.process:
                # self.process.stdin.close()  # Close the input pipe
                self.process.terminate()    # Terminate the FFmpeg process
                self.process.wait()         # Wait for the process to exit
                self.process = None
            if self.stream_thread:
                self.stream_thread.join()     # Wait for the stream thread to exit
            self.stop_button.config(state=tk.DISABLED)
            # self.start_stream().config(state=tk.NORMAL)
            print("Stream stopped.")

    def update_video_display(self, output_frame):
        # Convert BGR to RGB (if you're using OpenCV)
        """
        Update the video display with a new image.

        This function is a callback for the after method and is called
        with the image to be displayed as an argument. It sets the image
        attribute of the video label to the passed image and then
        configures the label to display the image.

        Parameters
        ----------
        output_frame : ndarray
            The BGR image to be displayed.

        """
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

        # Convert to ImageTk format
        img = Image.fromarray(output_frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video display
        self.root.after(0, self._update_image, imgtk)

    def _update_image(self, imgtk):
        """
        Update the video display with a new image.

        This function is a callback for the after method and is called
        with the image to be displayed as an argument. It sets the image
        attribute of the video label to the passed image and then
        configures the label to display the image.

        Parameters
        ----------
        imgtk : PhotoImage
            The image to be displayed in the video label.
        """
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def exit_and_close(self, event=None):
        # Stop the stream if it's running
        """
        Stop the stream if it's running and exit the application.

        This function is a callback for the "Close App" button press event.

        Parameters
        ----------
        event : None or tk.Event
            The event object passed by the button press event. If not provided,
            the function is called by the button press event.

        Returns
        -------
        None
        """
        self.stop_stream()
        # Exit the application
        self.root.quit()  # This will close the application

    def toggle_fullscreen(self, event=None):
        # Toggle full-screen mode
        """
        Toggle full-screen mode.

        This function is a callback for the F11 key press event.

        Parameters
        ----------
        event : None or tk.Event
            The event object passed by the key press event. If not provided,
            the function is called by the application.

        Returns
        -------
        None
        """
        self.root.attributes("-fullscreen", True)

    def exit_fullscreen(self, event=None):
        # Exit full-screen mode
        """
        Exit full-screen mode.

        This function is a callback for the Escape key press event, and is also
        called by the `toggle_fullscreen` function when the F11 key is pressed.

        Parameters
        ----------
        event : None or tk.Event
            The event object passed by the key press event. If not provided,
            the function is called by the `toggle_fullscreen` function.

        Returns
        -------
        None
        """
        self.root.attributes("-fullscreen", False)

