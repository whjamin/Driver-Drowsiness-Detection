from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

print("== 얼굴 좌표 데이터 불러 오는 중... ==")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

print("== 카메라 인식 중... ==")

vs = VideoStream(src=1).start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576

font = cv2.FONT_HERSHEY_SIMPLEX

image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corner 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.14
MOUTH_AR_THRESH = 0.89
EYE_AR_CONSEC_FRAMES = 5
MOUTH_AR_CONSEC_FRAMES = 5
EYE_COUNTER = 0
MOUTH_COUNTER = 0
cap = cv2.VideoCapture(0)

(mStart, mEnd) = (49, 68)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)

    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20), font, 0.5, (0, 0, 255), 2)

    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            EYE_COUNTER += 1

            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (500, 20), font, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0

        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), font, 0.7, (0, 0, 255), 2)

        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1

            if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Yawning!", (800, 20), font, 0.7, (0, 0, 255), 2)

        else:
            MOUTH_COUNTER = 0

        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), font, 0.35, (0, 255, 0), 1)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), font, 0.35, (0, 255, 0), 1)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), font, 0.35, (0, 255, 0), 1)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), font, 0.35, (0, 255, 0), 1)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), font, 0.35, (0, 255, 0), 1)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), font, 0.35, (0, 255, 0), 1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), font, 0.35, (0, 0, 255), 1)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, 
            end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()