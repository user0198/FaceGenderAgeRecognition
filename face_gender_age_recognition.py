import face_recognition
import numpy as np
import cv2
import math
import argparse
import os


# Draws frame around detected face
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


# Get names from the images
path = 'resources/known'
images = []
known_face_names = []
pathList = os.listdir(path)

for im in pathList:
    curImg = cv2.imread(f'{path}/{im}')
    images.append(curImg)
    known_face_names.append(os.path.splitext(im)[0])

# Create arrays of known face encodings
known_face_encodings = findEncodings(images)

# Initialize current frame marker
process_this_frame = True

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video_capture.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    # Resize frame of video to 1/4 size for faster face recognition processing
    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_smallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_smallFrame)
        face_encodings = face_recognition.face_encodings(
            rgb_smallFrame, face_locations)

        face_names = []
        for face_encode, face_loc in zip(face_encodings, face_locations):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encode)
            name = "-"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encode)

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):
                     min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        # Draws name
        for (top, _, _, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            # right *= 4
            # bottom *= 4
            left *= 4
            print('proceded')

            cv2.putText(resultImg, name, (left, top - 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

        # Draws gender and age
        cv2.putText(resultImg, f'{gender}, {age}', (
            faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Detecting ...", resultImg)


video_capture.release()
cv2.destroyAllWindows()
