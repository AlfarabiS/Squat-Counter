import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
import tempfile
from function import *
import csv
import time


# Scaffolding
file = st.sidebar.file_uploader('Upload File')

repetisi_text, repetisi_number = st.columns(2)

with repetisi_text:
    st.sidebar.text('Repetisi')
with repetisi_number:
    repetisi = st.sidebar.markdown('0')

st.sidebar.text('Referensi Gerakan')
st.sidebar.image('squat.gif')


if file:
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    tmpfile.write(file.read())

cap = cv2.VideoCapture(0)
stframe = st.empty()

# Initial csv
classname = ''
num_coords = 33

landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val),
                  'z{}'.format(val), 'v{}'.format(val),]

with open('coordinate.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(
        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

# Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mpDraw = mp.solutions.drawing_utils

# reps = 0
# stage = None
pTime = 0
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    counter = 0
    stage = None

    while cap.isOpened():
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        try:
            landmarks = results.pose_landmarks.landmark

            pose_row = list(np.array([
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ]for landmark in landmarks]))
            row = pose_row
            pose_row.insert(0, classname)
            # cek array
            # cv2.putText(img, str(pose_row), (50, 170),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)

            with open('coordinate.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(
                    f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(pose_row)
            # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)
                classname = classifier(hip_angle, knee_angle)

                if knee_angle > 135 and hip_angle > 135:
                    stage = "top"
                elif knee_angle < 100 and hip_angle < 100 and stage == 'top':
                    stage = "down"
                elif knee_angle <= 90 and hip_angle <= 90 and stage == 'down':
                    stage = "bottom"
                elif knee_angle > 100 and hip_angle > 100 and stage == 'bottom':
                    stage = "up"
                    counter = counter + 1

        except:
            pass

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, 'Repetisi = ' + str(int(counter)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(img, 'Penilaian Gerakan = ' + str(classname), (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(img, 'Stage = ' + str(stage), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(img, 'hip_angle = ' + str(hip_angle), (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(img, 'knee_angle = ' + str(knee_angle), (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame)
        cv2.waitKey(1)
        repetisi.write(counter)
