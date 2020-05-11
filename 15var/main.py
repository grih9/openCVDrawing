import imutils
import cv2 as cv
import numpy as np
from collections import deque

videoCapture = cv.VideoCapture(0)
videoMe = cv.VideoCapture(1)
planeList = cv.imread("123.jpg", 1)

lower_red = np.array([0, 109, 20], dtype = "uint8")
upper_red = np.array([15, 255, 255], dtype = "uint8")

lower_violet_red = np.array([165, 109, 20], dtype = "uint8")
upper_violet_red = np.array([180, 255, 255], dtype = "uint8")

lower_yellow = np.array([20, 109, 20], dtype = "uint8")
upper_yellow = np.array([40, 255, 255], dtype = "uint8")

lower_orange = np.array([12, 85, 110], dtype = "uint8")
upper_orange = np.array([23, 255, 255], dtype = "uint8")

lower_violet = np.array([130, 50, 50], dtype = "uint8")
upper_violet = np.array([160, 255, 255], dtype = "uint8")

lower_black = np.array([0, 0, 0], dtype = "uint8")
upper_black = np.array([0, 255, 255], dtype = "uint8")

lower_blue = np.array([98, 109, 20], dtype = "uint8")
upper_blue = np.array([112, 255, 255], dtype = "uint8")

lower_green = np.array([0, 50, 50], dtype = "uint8")
upper_green = np.array([180, 255, 255], dtype = "uint8")

color = ""
x = 0.0
y = 0.0
area = 0
isRealesed = False
isEraser = False
mask = None
pts = deque(maxlen = 2)

while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    ret2, me = videoMe.read()
    #frame = cv.flip(frame, 1)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red_tmp = cv.inRange(hsv, lower_red, upper_red)
    violet_red = cv.inRange(hsv, lower_violet_red, upper_violet_red)
    yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    blue = cv.inRange(hsv, lower_blue, upper_blue)
    green = cv.inRange(hsv, lower_green, upper_green)
    red = red_tmp + violet_red

    tmpColor = ""
    tmpMask = None
    tmpx = 0.0
    tmpy = 0.0
    maxArea = 0

    for k in range(1):
        moments = cv.moments(yellow, 1)
        x_moment = moments['m10']
        y_moment = moments['m01']
        tmpArea = moments['m00']
        if color == "yellow":
            if tmpArea > 750:
                x = x_moment
                y = y_moment
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
            tmpx = x_moment
            tmpy = y_moment

        moments = cv.moments(red, 1)
        x_moment = moments['m10']
        y_moment = moments['m01']
        tmpArea = moments['m00']
        if color == "red":
            if tmpArea > 750:
                x = x_moment
                y = y_moment
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
            tmpx = x_moment
            tmpy = y_moment

        moments = cv.moments(blue, 1)
        x_moment = moments['m10']
        y_moment = moments['m01']
        tmpArea = moments['m00']
        if color == "blue":
            if tmpArea > 750:
                x = x_moment
                y = y_moment
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
            tmpx = x_moment
            tmpy = y_moment

    if color == "":
        color = tmpColor
        mask = tmpMask
        area = maxArea
        x = tmpx
        y = tmpy

    if (area > 750) :
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
        cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        print("Centre : " + str(centre) + ", top point : " + str(extTop))
        cv.circle(frame, extTop, 4, [255, 0, 255], -1)

        if isRealesed:
            pts.appendleft(extTop)

            for i in range(1, len(pts)):
                d = list(pts[i - 1])
                tmp = list(pts[i])
                dist = (tmp[0] - d[0]) * (tmp[0] - d[0]) + (tmp[1] - d[1]) * (tmp[1] - d[1])
                print("Distanation : " + str(dist))
                if dist > 4900:
                    pts.popleft()
                    isRealesed = False
                    isEraser = False
                    pts.clear()
                    continue

                colorDraw = (0, 0, 255)
                thickness = 2
                if isEraser:
                    cv.circle(planeList, pts[i], 10, [255, 255, 255], -1)
                else:
                    if color == "yellow":
                        colorDraw = (0, 255, 255)
                    elif color == "blue":
                        colorDraw = (255, 0, 0)
                    cv.line(planeList, pts[i - 1], pts[i], colorDraw, thickness = thickness)

    blueScreen = cv.bitwise_and(frame, frame, mask = blue)
    redScreen = cv.bitwise_and(frame, frame, mask = red)
    yellowScreen = cv.bitwise_and(frame, frame, mask = yellow)
    cv.imshow("frame", frame)
    yellowRed = np.hstack([yellowScreen, redScreen])
    cv.imshow('yellow and red', yellowRed)
    cv.imshow('blue', blueScreen)
    cv.imshow('draw', planeList)
    #cv.imshow('me', me)

    pressedKey = cv.waitKey(1)
    if pressedKey & 0xFF == ord('z'):
        break
    elif pressedKey & 0xFF == ord('r'):
        if isEraser:
            isRealesed = True
        else:
            isRealesed = not isRealesed
        isEraser = False
        pts.clear()
    elif pressedKey & 0xFF == ord('c'):
        isRealesed = False
        isEraser = False
        planeList = cv.imread("123.jpg", 1)
        pts.clear()
    elif pressedKey & 0xFF == ord('d'):
        isEraser = not isEraser
        isRealesed = isEraser
        pts.clear()
    elif pressedKey & 0xFF == ord('s'):
        cv.imwrite("save.jpg", planeList)

cv.destroyAllWindows()
videoCapture.release()
