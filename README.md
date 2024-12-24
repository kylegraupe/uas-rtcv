<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=2000&pause=10000&color=0FFFD0&center=false&vCenter=true&width=1000&lines=Unmanned+Aerial+System+-+Real Time+Computer+Vision" alt="Typing SVG" />
</h1>

⭐ **Please leave a star!**

This tool integrates the use of FFMPEG, RTMP Servers, and PyTorch for real time computer vision via an Unmanned Aerial Systems (Drones), and is the only open-source project of the sort. 

## 🎬 Application Trailer ⚠️

To watch a short clip of the application in use, click [here.](https://www.graupe.io/portfolio/real-time-computer-vision-streamed-via-dji-drone) on graupe.io!

Stay tuned! Newest version solves latency issue and is in near real time! Trailer to come soon!

![uas_rtcv_test_flight](https://github.com/user-attachments/assets/6b717654-75de-4418-957d-0a8097e2173c)

---
<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=2000&pause=10000&startDelay=4000&color=E0AA3E&center=false&vCenter=true&width=1000&lines=LEAVE+A+STAR+OR+A+FOLLOW+IF+THIS+REPO+IS+HELPFUL!" alt="Typing SVG" />
</h1>

---

## Context 

In various industries and applications, there is a growing need for real-time, high-quality video streaming capabilities. DJI is the market-dominant supplier in consumer and industry drones. Therefore, building an application for real-time Computer Vision, leveraging DJI drones like the Mini 4 Pro, is essential to harness the full potential of these advanced imaging systems. This application provides immediate AI analysis to both consumers and professionals, eliminating the need for more costly alternatives and the necessity of DJI SDK while offering comparable control over the video feed and frames.

---

## What problem does this solve?

This application enables the use of computer vision on a DJI drone that does **NOT** get access to the DJI SDK. To see a list of the supported SDKs and their associated DJI drones, click [here](https://developer.dji.com/). The drone that I am using for the development of this application is the DJI Mini 4 Pro, the latest release of the <250g class of consumer drones, which is **NOT** supported in the DJI SDK.

---

## Features

- **Real-Time Semantic Segmentation**: Perform live semantic segmentation on aerial drone footage.
- **Custom Model Integration**: Integrate custom U-Net models for segmentation tasks.
- **Post-Processing**: Apply advanced post-processing techniques to improve segmentation accuracy.
  - **Conditional Random Field (CRF)**: A probabilistic graphical model that refines pixel-level classification by considering spatial dependencies.
  - **Dilation**: Expands the boundaries of regions in a binary image, filling small holes and connecting adjacent regions.
  - **Erosion**: Shrinks the boundaries of regions in a binary image, removing small noise and detaching connected elements.
  - **Median Smoothing**: Reduces noise in an image by replacing each pixel with the median of neighboring pixel values.
  - **Gaussian Blur**: Applies a Gaussian function to blur an image, reducing high-frequency details and smoothing edges.
- **GUI Integration**: A user-friendly graphical interface for controlling and visualizing the segmentation process.
  - Python's TkInter is not suitable for high-frame displaying, therefore UI needs to be reworked in a new framework due to multithreading successes.
- **Custom Stream Buffer**: Custom implementation of Producer-Consumer Threading paradigm to keep stream in near real-time.
  - Resolves the growing latency problem associated with more naive RTMP stream handlers.
- **Robust Logging**: The application uses logging to appropriately record application and testing runs to be used during test flights of the UAS.
  - Giving operators the ability to record all information from a given test flight is crucial for any application of drone/UAS technology as projects operate in planned stages with test flights.
  - Onboard data recording is also crucial in capturing real-time data for post-flight processing.
  - Logging currently includes separate logs generated for tests as well as core application runs.
  - Note: Ff you would like to include your logs in the GitHub repository, navigate to **.gitignore** and comment-out or remove the *.log line from the file.
--- 

## SETUP (MacOS Apple Silicon):
- navigate to **requirements.txt** and install all dependencies.
- install NGINX with RTMP module: 'brew install nginx-full --with-rtmp'
- configure NGINX Configuration: 'sudo nano /opt/homebrew/etc/nginx/nginx.conf'
  - **.. /nginx.conf** file is provided as an example configuration
- set RTMP URL in settings.py file: 'RTMP_URL = "rtmp://your_ip_address:1935/live"'
  - Uses Port 1935, which is the default port, but can be changed in settings.py file.
- install OpenCV with FFMPEG Support (in IDE Terminal) this part can be tricky:
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
- Make sure to download a Local RTMP Server
  - I use this for MacOS: https://github.com/sallar/mac-local-rtmp-server

---

## DEBUGGING:
 - run: 'ffplay -f flv **_your_rtmp_url_**' to verify if stream is being sent via RTMP Server. 
 - run: 'sudo nano /opt/homebrew/etc/nginx/nginx.conf' to edit nginx.conf file.
   - nginx.conf file controls the functionality of the NGINX Server. There should be a block for the RTMP Server, which will specify the location of the Listening Port (typically 1935).

---

## EXECUTION:
- launch: Local RTMP Server application. This will facilitate the connection from the drone.
- navigate to transmission tab on DJI RC2. select 'Live Streaming Platforms' and select 'RTMP'.
  - currently working with high FPS and low latency on the following settings:
    - Frequency: 2.4 GHz
    - Channel Mode: Auto
    - Resolution: 720p (only option)
    - Bit Rate: 5 Mbps
  - Press 'Start Live Stream'
- Endpoints:
  - primary endpoint: main.py
    - ensure environment and application variables are customized appropriately in **settings.py**
  - testing endpoint: test_executive.py
    - this will run all unit tests and eventually integration tests prior

---

## REFERENCES
- Model Training Conducted in Kaggle Jupyter Notebook Environment:
  - https://www.kaggle.com/code/kylegraupe/model-training-dji-real-time-semantic-segmentation
  - Model training also included in repository: 'model_training/model-training-dji-real-time-semantic-seg-v1.ipynb'

---

## ISSUES WITH CURRENT VERSION

- Duplicate logging events due to multiple threads.
