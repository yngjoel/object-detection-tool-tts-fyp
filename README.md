![](https://emoji.gg/assets/emoji/9879_hackerman.gif)
# A tool to support blind people in detecting objects and obstacles via voice notification 
This project aims to help visually impaired people in having a more private and convenient life by providing them their own eyes with the use of object Detection with voice feedback.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The things you need before start using the web application

* Any IDE (Preferred: Visual Studio Code)
* Install latest version of python: https://www.python.org/downloads/
* FFMPEG: https://ffmpeg.org/download.html

### Installation

A step by step guide that will tell you how to get the required libraries to run the application
```
$ python get-pip.py
//if error occurs when installing opencv, use this command instead *pip install opencv-python-headless
$ pip install opencv-python
$ pip install numpy
$ pip install os-win
$ pip install pyttsx3
$ pip install pydub
$ pip install imutils
```
* Get [YOLOv4 Weight](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view?usp=drive_open) file and place it into yolo-coco-data folder

## Usage

This command is used to start the Flask Server and web application

```
$test.py
```
