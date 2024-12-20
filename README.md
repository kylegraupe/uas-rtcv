# Real-Time Semantic Segmentation on DJI Drone via RTMP Server

This application takes a video stream from a DJI drone via RTMP Server and performs image processing and semantic segmentation on the video stream.

## 🎬 Application Trailer ⚠️

To watch a short clip of the application in use, click [here.](https://www.graupe.io/portfolio/real-time-computer-vision-streamed-via-dji-drone)

Stay tuned! Newest version solves latency issue and is in near real time! Trailer to come soon!

### ⭐ **IF YOU FIND THE PROJECT INTERESTING, PLEASE LEAVE A STAR! ⭐**

## Context 

In various industries and applications, there is a growing need for real-time, high-quality video streaming capabilities. DJI is the market-dominant supplier in consumer and industry drones. Therefore, building an application for real-time Computer Vision, leveraging DJI drones like the Mini 4 Pro, is essential to harness the full potential of these advanced imaging systems. This application provides immediate AI analysis to both consumers and professionals, eliminating the need for more costly alternatives and the necessity of DJI SDK while offering comparable control over the video feed and frames.

## What problem does this solve?

This application enables the use of computer vision on a DJI drone that does **NOT** get access to the DJI SDK. To see a list of the supported SDKs and their associated DJI drones, click [here](https://developer.dji.com/). The drone that I am using for the development of this application is the DJI Mini 4 Pro, the latest release of the <250g class of consumer drones, which is **NOT** supported in the DJI SDK.

## Features

- **Real-Time Semantic Segmentation**: Perform live semantic segmentation on drone footage.
- **Custom Model Integration**: Integrate custom U-Net models for segmentation tasks.
- **Post-Processing**: Apply advanced post-processing techniques to improve segmentation accuracy.
  - **Conditional Random Field (CRF)**: A probabilistic graphical model that refines pixel-level classification by considering spatial dependencies.
  - **Dilation**: Expands the boundaries of regions in a binary image, filling small holes and connecting adjacent regions.
  - **Erosion**: Shrinks the boundaries of regions in a binary image, removing small noise and detaching connected elements.
  - **Median Smoothing**: Reduces noise in an image by replacing each pixel with the median of neighboring pixel values.
  - **Gaussian Blur**: Applies a Gaussian function to blur an image, reducing high-frequency details and smoothing edges.
- **GUI Integration**: A user-friendly graphical interface for controlling and visualizing the segmentation process.
  - Python's TkInter is not suitable for high-frame displaying, therefore UI needs to be reworked in a new framework due to multithreading successes.


## SETUP (MacOS Apple Silicon):
- install NGINX with RTMP module: 'brew install nginx-full --with-rtmp'
- configure NGINX Configuration: 'sudo nano /opt/homebrew/etc/nginx/nginx.conf'
- set RTMP URL in settings.py file: 'RTMP_URL = "rtmp://your_ip_address:1935/live"'
  - Uses Port 1935, which is the default port, but can be changed in settings.py file.
- install OpenCV with FFMPEG Support (in IDE Terminal):
  - run: 'brew install ffmpeg'
  - run: 'brew install cmake git'
  - run: 'git clone https://github.com/opencv/opencv.git'
  - run: 'cd opencv'
  - run: 'mkdir opencv_build'
  - run: 'cd opencv_build'
  - run: 'cmake -DWITH_FFMPEG=ON -DFFMPEG_INCLUDE_DIRS=/usr/local/include/ffmpeg -DFFMPEG_LIBRARIES=/usr/local/lib/libavcodec.dylib ..'
  - run: 'make -j4'
  - run: 'sudo make install'
  - verify installation with 'print(cv2.getBuildInformation())' and check FFMPEG section.

## DEBUGGING:
 - run: 'ffplay -f flv **_your_rtmp_url_**' to verify if stream is being sent via RTMP Server. 
 - run: 'sudo nano /opt/homebrew/etc/nginx/nginx.conf' to edit nginx.conf file.
   - nginx.conf file controls the functionality of the NGINX Server. There should be a block for the RTMP Server, which will specify the location of the Listening Port (typically 1935).

## EXECUTION:
- open: Local RTMP Server application. This will facilitate the connection from the drone.
- navigate to transmission tab on DJI RC2. select 'Live Streaming Platforms' and select 'RTMP'.
  - currently working with high FPS and low latency on the following settings:
    - Frequency: 2.4 GHz
    - Channel Mode: Auto
    - Resolution: 720p (only option)
    - Bit Rate: 5 Mbps
- Endpoints:
  - runs without UI: stream_processing_threaded.py
  - runs deprecated veriion: main.py 

## REFERENCES
- Model Training Conducted in Kaggle Jupyter Notebook Environment:
  - https://www.kaggle.com/code/kylegraupe/model-training-dji-real-time-semantic-segmentation
  - Model training also included in repository: 'model_training/model-training-dji-real-time-semantic-seg-v1.ipynb'
