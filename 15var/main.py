import imutils
import cv2 as cv
import numpy as np
from collections import deque

videoCapture = cv.VideoCapture(0)
videoMe = cv.VideoCapture(1)
planeList = cv.imread("123.jpg", 1)

lowerRed = np.array([0, 109, 20], dtype="uint8")
upperRed = np.array([15, 255, 255], dtype="uint8")

lowerVioletRed = np.array([165, 109, 20], dtype="uint8")
upperVioletRed = np.array([180, 255, 255], dtype="uint8")

lowerYellow = np.array([20, 109, 20], dtype="uint8")
upperYellow = np.array([40, 255, 255], dtype="uint8")

lowerOrange = np.array([12, 85, 110], dtype="uint8")
upperOrange = np.array([23, 255, 255], dtype="uint8")

lowerViolet = np.array([130, 50, 50], dtype="uint8")
upperViolet = np.array([160, 255, 255], dtype="uint8")

lowerBlack = np.array([0, 0, 0], dtype="uint8")
upperBlack = np.array([0, 255, 255], dtype="uint8")

lowerBlue = np.array([98, 109, 20], dtype="uint8")
upperBlue = np.array([112, 255, 255], dtype="uint8")

lowerGreen = np.array([0, 50, 50], dtype="uint8")
upperGreen = np.array([180, 255, 255], dtype="uint8")

color = ""
x = 0.0
y = 0.0
area = 0
isRealesed = False
isEraser = False
mask = None
points = deque(maxlen=2)

while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    ret2, me = videoMe.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    redTmp = cv.inRange(hsv, lowerRed, upperRed)
    violetRed = cv.inRange(hsv, lowerVioletRed, upperVioletRed)
    yellow = cv.inRange(hsv, lowerYellow, upperYellow)
    blue = cv.inRange(hsv, lowerBlue, upperBlue)
    red = redTmp + violetRed

    tmpColor = ""
    tmpMask = None
    tmpX = 0.0
    tmpY = 0.0
    maxArea = 0

    for k in range(1):
        moments = cv.moments(yellow, 1)
        xMoment = moments['m10']
        yMoment = moments['m01']
        tmpArea = moments['m00']
        if color == "yellow":
            if tmpArea > 750:
                x = xMoment
                y = yMoment
                area = tmpArea
                mask = yellow
                break
            else:
                color = ""
                mask = None
                area = 0
        elif tmpArea > 750 and tmpArea > maxArea:
            tmpMask = yellow
            maxArea = tmpArea
            tmpColor = "yellow"
            tmpX = xMoment
            tmpY = yMoment

        moments = cv.moments(red, 1)
        xMoment = moments['m10']
        yMoment = moments['m01']
        tmpArea = moments['m00']
        if color == "red":
            if tmpArea > 750:
                x = xMoment
                y = yMoment
                area = tmpArea
                mask = red
                break
            else:
                color = ""
                mask = None
                area = 0
        elif tmpArea > 750 and tmpArea > maxArea:
            tmpMask = red
            maxArea = tmpArea
            tmpColor = "red"
            tmpX = xMoment
            tmpY = yMoment

        moments = cv.moments(blue, 1)
        xMoment = moments['m10']
        yMoment = moments['m01']
        tmpArea = moments['m00']
        if color == "blue":
            if tmpArea > 750:
                x = xMoment
                y = yMoment
                area = tmpArea
                mask = blue
                break
            else:
                color = ""
                mask = None
                area = 0
        elif tmpArea > 750 and tmpArea > maxArea:
            tmpMask = blue
            maxArea = tmpArea
            tmpColor = "blue"
            tmpX = xMoment
            tmpY = yMoment

    if color == "":
        color = tmpColor
        mask = tmpMask
        area = maxArea
        x = tmpX
        y = tmpY

    if area > 750:
        x = int(x / area)
        y = int(y / area)
        if isRealesed:
            if isEraser:
                cv.putText(frame, color, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 41), 2)
            else:
                cv.putText(frame, color, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            cv.putText(frame, color, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        centre = (x, y)
        conturs = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        conturs = imutils.grab_contours(conturs)
        c = max(conturs, key=cv.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        print("Centre : " + str(centre) + ", top point : " + str(extTop))
        cv.circle(frame, extTop, 4, [255, 0, 255], -1)

        if isRealesed:
            points.appendleft(extTop)

            for i in range(1, len(points)):
                d = list(points[i - 1])
                tmp = list(points[i])
                dist = (tmp[0] - d[0]) * (tmp[0] - d[0]) + (tmp[1] - d[1]) * (tmp[1] - d[1])
                print("Distanation : " + str(dist))
                if dist > 4900:
                    points.popleft()
                    isRealesed = False
                    isEraser = False
                    points.clear()
                    continue

                colorDraw = (0, 0, 255)
                thickness = 2
                if isEraser:
                    cv.circle(planeList, points[i], 10, [255, 255, 255], -1)
                else:
                    if color == "yellow":
                        colorDraw = (0, 255, 255)
                    elif color == "blue":
                        colorDraw = (255, 0, 0)
                    cv.line(planeList, points[i - 1], points[i], colorDraw, thickness=thickness)

    blueScreen = cv.bitwise_and(frame, frame, mask=blue)
    redScreen = cv.bitwise_and(frame, frame, mask=red)
    yellowScreen = cv.bitwise_and(frame, frame, mask=yellow)
    cv.imshow("frame", frame)
    yellowRed = np.hstack([yellowScreen, redScreen])
    cv.imshow('yellow and red', yellowRed)
    cv.imshow('blue', blueScreen)
    cv.imshow('draw', planeList)
    # cv.imshow('me', me)

    pressedKey = cv.waitKey(1)
    if pressedKey & 0xFF == ord('z'):
        print("Exit")
        break
    elif pressedKey & 0xFF == ord('r'):
        if isEraser:
            isRealesed = True
        else:
            isRealesed = not isRealesed
        isEraser = False
        points.clear()
        print("Release")
    elif pressedKey & 0xFF == ord('c'):
        isRealesed = False
        isEraser = False
        planeList = cv.imread("123.jpg", 1)
        points.clear()
        print("Clear")
    elif pressedKey & 0xFF == ord('d'):
        isEraser = not isEraser
        isRealesed = isEraser
        points.clear()
        print("Eraser")
    elif pressedKey & 0xFF == ord('s'):
        cv.imwrite("save.jpg", planeList)
        print("Saved")

cv.destroyAllWindows()
videoCapture.release()
