import cv2
import numpy as np
import os
import time

from matplotlib import pyplot as plt

def yolo_detect(img_,img,idx,verbose=False,save=False):
    # Load names of classes and get random colors for them.
    classes = open('/home/anjali/carla_sim/models/coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNetFromDarknet('/home/anjali/carla_sim/models/yolov3.cfg', '/home/anjali/carla_sim/models/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img_, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    if verbose:
        t = time.time()
        print('It took %.3f seconds to process the image.' % (t-t0))
    
    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)
    if save:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]

                #  -- Arguments for CV2 rectangle:
                # cv2.rect   (img,  x, y,   width, height, color, line width)
                cv2.rectangle(img_, (x, y), (x + w, y + h), color, 4)

                # Labels and confidences for the image
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(img_, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imwrite(f'/home/anjali/carla_sim/data/output_{idx}.png', img_)
        print('Image preview:')
        plt.imshow(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB), interpolation='none')
    return classIDs,boxes,confidences