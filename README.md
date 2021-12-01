# object_detection_with_raspberrypi

 ## Reference
 Learned from "Computer Vision with Embedded Machine Learning" by Shawn Hymel on Coursera.  More details check below:  
  [1] ShawnHymeL's repository .   [ [Source code]( https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning.git) ]

## Introduction
This program is part of the CEG4912 project which is designed by **Group 2 of Section F00 for 2021 Fall Term**. Two cameras are used to monitor the EDM.One of the cameras will use an intelligent algorithm to identify the input numbers.The program will send an error signal when any error input is detected. 
## Components
* Raspberry Pi 4
* Pi camera
* USB web camera
## Procedure
1. Collecting Data
2. 
## Documentation
* numPy.py: Conversion between image and array.
* capture_pi.py: Capture 1 image with pi camera.  
script " for i in {1..10}; do python capture_pi.py; done " to capture 10 images.
* image_classifier.py: Classify image sets.
## Collection Data
capture_pi.py is used to capture images.
### Dataset Collection Checklist
- [x] 50 images of the same object for each class
- [ ] 3-4 classes for the problem
- [ ] Shapes will be very different between calsses
- [x] Scale photos to 96x96 pixels
- [x] Bitmap or PNG format
- [ ] Similar distance, position, lighting, and background
### Example
<img src=" " width="250" height="250">  

## DeployModel using Edge Impulse to Raspberry pi
#### Edge Impulse  
Edge Impulse for Linux is the easiest way to build Machine Learning solutions on real embedded hardware. It contains tools which let you collect data from any microphone or camera, can be used with the Node.js, Python, Go and C++ SDKs to collect new data from any sensor, and can run impulses with full hardware acceleration 

1. Install dependencies
```
curl -sL https://deb.nodesource.com/setup_12.x | sudo bash -
sudo apt install -y gcc g++ make build-essential nodejs sox gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps
npm config set user root && sudo npm install edge-impulse-linux -g --unsafe-perm
```
2. Install required Python SDK 

This library lets you run machine learning models and collect sensor data on Linux machines using Python. The SDK is open source and hosted on GitHub: edgeimpulse/linux-sdk-python.
```
sudo apt-get install libatlas-base-dev libportaudio0 libportaudio2 libportaudiocpp0 portaudio19-dev
pip3 install edge_impulse_linux -i https://pypi.python.org/simple
```
## Index
#### Updata python3 on Raspberry pi:
1. sudo apt install python3 idle3
#### Update-alternatives:
2. sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10
#### Some programs and liberaries
1. VS code: sudo apt upgrade code
2. Pip: sudo apt-get install python-pip
3. NumPy: pip install scipy
4. Opencv2: pip install opencv-python==   
notes: Need to install one package: sudo apt-get install libatlas-base-dev
5. Picamera: pip install picamera  
notes: Cannot install picamera on Windows System.
6. Tensorflow and Keras: pip install tensorflow==1.14.0
                         pip install keras==2.7.0
7. skimage: pip install-image
Image processing in python
8. sklearn: pip install scikit-learn
9. dege-impulse-linux 1.0.6: pip install edge_impulse_linux
