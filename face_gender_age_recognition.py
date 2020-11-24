# Face, Gender and Age recognition in real time from webcam capture
# Eugene Kykyna

import face_recognition
import numpy as np
import cv2
import math
import argparse
import os


# Draws a rectangle around face
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (255, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


# Finds encodings of the faces from the images list
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Path to known faces to recognize
path = 'resources/known'
pathList = os.listdir(path)

# List of known faces images
images = []

# List of names from images
knownFaceNames = []

# Distance between face box and rectangle to draw
padding = 20

# Argparse init
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

# Import from neural network database
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Weights and classifiers
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
# Reading from deep neural network
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Gets names from the images
for im in pathList:
    curImg = cv2.imread(f'{path}/{im}')
    images.append(curImg)
    knownFaceNames.append(os.path.splitext(im)[0])

# Create arrays of known face encodings
knownFaceEncodings = findEncodings(images)

# Initialize current frame marker
processThisFrame = True

# Get a reference to webcam
videoCapture = cv2.VideoCapture(args.image if args.image else 0)

# Main loop
while cv2.waitKey(1) < 0:
    hasFrame, frame = videoCapture.read()
    # Break loop if face is not in frame
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    # Resize frame of video to 1/4 size for faster face recognition processing
    smallerFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgbSmallerFrame = cv2.cvtColor(smallerFrame, cv2.COLOR_BGR2RGB)

    # Check if calculation is being proceed once for a single frame
    if processThisFrame:
        # Find all the faces and get face encodings in the current frame
        faceLocations = face_recognition.face_locations(rgbSmallerFrame)
        faceEncodings = face_recognition.face_encodings(
            rgbSmallerFrame, faceLocations)

        facesNames = []
        for faceEncode, _ in zip(faceEncodings, faceLocations):
            # See if face mathes known one
            matches = face_recognition.compare_faces(
                knownFaceEncodings, faceEncode)
            # Leave blank if no match
            name = ""

            # Get euclid distance between known faces encodings and current one in a frame
            faceDistances = face_recognition.face_distance(
                knownFaceEncodings, faceEncode)

            # Calculate best match index
            bestMatchIndex = np.argmin(faceDistances)

            # Connect name to an encoded face in a frame
            if matches[bestMatchIndex]:
                name = knownFaceNames[bestMatchIndex]
            facesNames.append(name)

    # Move to the next frame
    processThisFrame = not processThisFrame

    for faceBox in faceBoxes:
        # Get face cut from frame
        face = frame[max(0, faceBox[1]-padding):
                     min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        # Draw name
        for (top, _, _, left), name in zip(faceLocations, facesNames):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            # right *= 4
            # bottom *= 4
            left *= 4
            print('proceded')

            cv2.putText(resultImg, name, (left, top - 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

        # Draw gender and age
        cv2.putText(resultImg, f'{gender}, {age}', (
            faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        # Display drawings
        cv2.imshow("Detecting ...", resultImg)


# Release memory
videoCapture.release()
cv2.destroyAllWindows()
