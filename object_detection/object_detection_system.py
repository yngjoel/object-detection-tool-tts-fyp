"""

Object Detection in real time with YOLOv4 and OpenCV

"""

# import libraries
import numpy as np
import cv2
import time

"""
START - stream object detection camera
"""

# camera capture, 0 is for built-in camera 
# which can be changed to 1,2 or 3 depends on your system
camera = cv2.VideoCapture(0)

#variables
h, w = None, None

"""
END - stream object detection camera
"""

"""
START - YOLOv4 Network
"""

# COCO Labels file
# Read the labels in coco.names file
with open('yolo-coco-data/coco.names')as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolo-coco-data\yolov4.cfg', 'yolo-coco-data\yolov4.weights')
# get names from layers of yolov4 network
layers_names_all = network.getLayerNames()

layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# probability to eliminate weak predictions
probability_minimum = 0.7

# non-maximum supression (filter weak bounding boxes)
threshold = 0.3 

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

"""
END - YOLOv4 Network
"""

"""
START - Reading frames in loop
"""
while True:
    _, frame = camera.read()
    # all frames has same dimension, hence slicing tuple from only 2 elements
    if w is None or h is None:
        h, w = frame.shape[:2]

    """
    START - Blob of current frames
    """
    blob = cv2.dnn.blobFromImage(frame, 1 /255.0, (416,416), 
                                swapRB=True, crop = False)
    
    """
    END - Blob of current frames
    """

    """
    START - Forward pass implementation
    """
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    print('Current Frame took {:.5f} seconds'.format(end -start))

    """
    END - Forward pass implementation
    """

    """
    START - Get bounding boxes
    """
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
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
                y_min = int(y_center - (box_width / 2))

                # Adding results into prepared lists
                bounding_boxes,append([x_min, y_min,
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
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    """
    END - Non-maximum supression
    """

    """
    START - Draw bounding boxes and labels
    """
    #after non-maximum supression, check for detected objects
    if len(results) > 0:
        for i in results.flatten():
            #get bb coordinates for its width and height
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
            cv2.putText(frames, text_box_current, (x_min, y_min - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    """
    END - Draw bounding boxes and labels
    """

    """
    START - Show processed frames in OpenCV window
    """
    cv2.namedWindow('YOLO v4 Object Detection RT', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO v4 Object Detection RT', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    """
    END - Show processed frames in OpenCV window
    """
"""
END - Reading frames in loop
"""
#Close OpenCV Windows & Camera
camera.release()
cv2.destroyAllWindows()