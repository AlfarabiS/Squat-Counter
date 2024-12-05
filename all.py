import cv2
import cv2.large_kinfu
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

num_coords = 33


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val),
                  'z{}'.format(val), 'v{}'.format(val),]

with open('coodinate.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(
        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


classname = ''
cap = cv2.VideoCapture('0916.mp4')

if (cap.isOpened() == False):
    print('eror')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
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

        # Giving Classname by condition
        if hip_angle < 85:
            if knee_angle < 130:
                classname = 'perfect'
                cv2.putText(image, "Perfect ",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        elif hip_angle < 70:
            if knee_angle < 135:
                classname = 'hip terlalu menekuk'
                cv2.putText(image, "hip terlalu menekuk ",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            # coordinate and dotting
        shoulder_coordinate = tuple(np.multiply(
            shoulder, [480, 854]).astype(int))
        hip_coordinate = tuple(np.multiply(hip, [480, 854]).astype(int))
        knee_coordinate = tuple(np.multiply(knee, [480, 854]).astype(int))
        ankle_coordinate = tuple(np.multiply(
            ankle, [480, 854]).astype(int))

        cv2.line(image, shoulder_coordinate,
                 hip_coordinate, (255, 255, 255), 3)
        cv2.line(image, knee_coordinate,
                 hip_coordinate, (255, 255, 255), 3)
        cv2.circle(image, shoulder_coordinate,
                   20, (17, 217, 34), cv2.FILLED)
        cv2.circle(image, hip_coordinate, 20, (17, 217, 34), cv2.FILLED)
        cv2.circle(image, knee_coordinate, 20, (17, 217, 34), cv2.FILLED)
        cv2.circle(image, ankle_coordinate, 20, (17, 217, 34), cv2.FILLED)
        cv2.circle(image, shoulder_coordinate,
                   15, (0, 0, 0), 3)
        cv2.circle(image, ankle_coordinate,
                   15, (0, 0, 0), 3)
        cv2.circle(image, hip_coordinate, 15, (0, 0, 0), 3)
        cv2.circle(image, knee_coordinate, 15, (0, 0, 0), 3)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

# destroyAllWindows()
