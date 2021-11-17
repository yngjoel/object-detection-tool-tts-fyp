"""

Object Detection in real time with YOLOv4 and OpenCV

"""

# import libraries
from flask import Flask, render_template, Response
import numpy as np
import cv2
import time
import os
import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment
from pydub.playback import play
from playsound import playsound
AudioSegment.converter = "C:/Program Files/ffmpeg/ffmpeg.exe"

"""
START - stream object detection camera
"""

# camera capture, 0 is for built-in camera 
# which can be changed to 1,2 or 3 depends on your system
app = Flask(__name__) #Flask Web Server
camera = cv2.VideoCapture(0)

"""
END - stream object detection camera
"""

def gen_frames():  # generate frame by frame from camera

    """
    START - Reading frames in loop
    """
    while True:
        h, w = None, None

        """
        START - YOLOv4 Network
        """
        # COCO Labels file
        # Read the labels in coco.names file
        with open('yolo-coco-data/coco.names') as f:
            labels = [line.strip() for line in f]

        network = cv2.dnn.readNetFromDarknet('yolo-coco-data\yolov4.cfg', 'yolo-coco-data\yolov4.weights')
        # get names from layers of yolov4 network
        layers_names_all = network.getLayerNames()
        #Get required output layers from yolov4 (return indexes of layer with unconnected outputs)
        layers_names_output = \
            [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

        # probability to eliminate weak predictions
        probability_minimum = 0.5

        # non-maximum supression (filter weak bounding boxes)
        threshold = 0.3
        #colors that represent every detected objects
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        """
        END - YOLOv4 Network
        """

        _, frame = camera.read()
        # all frames has same dimension, hence slicing tuple from only 2 elements
        # Capture frame-by-frame
        if not _:
            break
        else:
            if w is None or h is None:
                h, w = frame.shape[:2]

            """
            START - Blob of current frames
            """
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)
            
            """
            END - Blob of current frames
            """

            """
            START - Forward pass implementation
            """
            network.setInput(blob)  # setting blob as input to the network
            start = time.time()
            output_from_network = network.forward(layers_names_output)
            end = time.time()

            #Time spent on per frame
            print('Current frame took {:.5f} seconds'.format(end - start))

            """
            END - Forward pass implementation
            """

            """
            START - Get bounding boxes
            """
            #List for detected bounding boxes, obtained confidences and calss_numbers
            bounding_boxes = []
            confidences = []
            class_numbers = []

            # after feed forward pass, it goes thru all output layers
            for result in output_from_network:
                # going thru all detections from current output layer
                for detected_objects in result:
                    # Get 80 possible classes for detected object
                    scores = detected_objects[5:]
                    # Get the index class with max probability value
                    class_current = np.argmax(scores)
                    # Get the value of probability for defined class
                    confidence_current = scores[class_current]
                    #eliminate weak predictions using probability_minimum
                    if confidence_current > probability_minimum:
                        #scale bounding box to initial frame size
                        #YOLO data format keeps coordinates for center of bounding boxes of current width and height
                        #The width and height of original frame get coordinates for center of bounding boxes
                        box_current = detected_objects[0:4] * np.array([w, h, w, h])
                        #YOLO format, get coordinates from top left corner (x_min and y_min)
                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        # Adding results into prepared lists
                        bounding_boxes.append([x_min, y_min,
                                            int(box_width), int(box_height)])
                        confidences.append(float(confidence_current))
                        class_numbers.append(class_current)

            """
            END - Get bounding boxes
            """

            """
            START - Non-maximum supression
            """
            # Implementing non-maximum suppression of given bounding boxes
            #bounding boxes are excluded if it has low cc or same region bb has higher confidence
            results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                    probability_minimum, threshold)

            texts = []

            """
            END - Non-maximum supression
            """

            """
            START - Draw bounding boxes and labels
            """
            #after non-maximum supression, check for detected objects
            if len(results) > 0:
                #Goes thru the indexes of results
                for i in results.flatten():
                    #get bounding box coordinates for its width and height
                    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                    #colour for bounding boxes & convert numpy to array list
                    colour_box_current = colours[class_numbers[i]].tolist()

                    # Drawing bounding box on the original current frame
                    cv2.rectangle(frame, (x_min, y_min),
                                (x_min + box_width, y_min + box_height),
                                colour_box_current, 2)
                    
                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                        confidences[i])
                
                    # Putting text with label and confidence on the original image
                    cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                    
                    #Position of detected objects, Approx position
                    if x_center <= w/3:
                        w_pos = "left "
                    elif x_center <= (w/3 * 2):
                        w_pos = "center "
                    else:
                        w_pos = "right "
                    
                    if y_center <= h/3:
                        h_pos = "top "
                    elif y_center <= (h/3 * 2):
                        h_pos = "mid "
                    else:
                        h_pos = "bottom "
                    
                    texts.append(h_pos + w_pos + labels[class_numbers[i]])
           
            print(texts) #In the console, objects detected will be printed as text

            #Text to speech function 
            if texts:
                description = ', '.join(texts)
                tts = gTTS(description, lang='en') #Text to speech language is set to English
                tts.save('gTTS.mp3') #Speech is saved as a .mp3 file
                tts = AudioSegment.from_mp3("gTTS.mp3") #Audio Segment will call the .mp3 file
                subprocess.call(["ffplay", "-nodisp", "-autoexit", "gTTS.mp3"]) #ffplay will automatically play the audio in gTTS.mp3 file
            
            """
            END - Draw bounding boxes and labels
            """
            ret, buffer = cv2.imencode('.jpg', frame) #call the object detection system to flask web application
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    """Object Detection home page."""
    return render_template('home.html')

@app.route('/obj')
def index():
    """Video streaming page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)