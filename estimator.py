import cv2
import mediapipe as mp
import numpy as np
import csv
from function import calculate_angle
import time


def pose(source):
    num_coords = 33

    landmarks = ['class']
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val),
                      'z{}'.format(val), 'v{}'.format(val),]

    with open('coodinate.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

    classname = ''
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(source)
    pTime = 0
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        try:
            landmarks = results.pose_landmarks.landmark

            pose_row = list(np.array(
                [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]))
            row = pose_row
            row.insert(0, classname)

            with open('coodinate.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

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

        except:
            pass

        print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, str(int(fps)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        while True:
            return img
