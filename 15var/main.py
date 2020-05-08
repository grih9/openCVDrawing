import argparse
import imutils
import cv2 as cv
import numpy as np
from collections import deque

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=2,
	help="max buffer size")
args = vars(ap.parse_args())

videoCapture = cv.VideoCapture(1)
videoMe = cv.VideoCapture(0)
planeList = cv.imread("123.jpg", 1)

lower_red = np.array([0, 85, 110], dtype = "uint8")
upper_red = np.array([15, 255, 255], dtype = "uint8")

lower_violet_red = np.array([165, 85, 110], dtype = "uint8")
upper_violet_red = np.array([180, 255, 255], dtype = "uint8")

lower_yellow = np.array([20, 85, 110], dtype = "uint8")
upper_yellow = np.array([40, 255, 255], dtype = "uint8")

lower_orange = np.array([12, 85, 110], dtype = "uint8")
upper_orange = np.array([23, 255, 255], dtype = "uint8")

lower_violet = np.array([135, 85, 110], dtype = "uint8")
upper_violet = np.array([150, 255, 255], dtype = "uint8")

lower_black = np.array([0, 0, 0], dtype = "uint8")
upper_black = np.array([0, 255, 255], dtype = "uint8")

lower_blue = np.array([100, 85, 110], dtype = "uint8")
upper_blue = np.array([115, 255, 255], dtype = "uint8")

lower_green = np.array([30, 85, 110], dtype = "uint8")
upper_green = np.array([80, 255, 255], dtype = "uint8")

color = ""
x = 0.0
y = 0.0
area = 0
isRealesed = False
mask = None
pts = deque(maxlen=args["buffer"])

while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    ret2, me = videoMe.read()
    frame = cv.flip(frame, 1)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red_tmp = cv.inRange(hsv, lower_red, upper_red)
    violet_red = cv.inRange(hsv, lower_violet_red, upper_violet_red)
    yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    orange = cv.inRange(hsv, lower_orange, upper_orange)
    violet = cv.inRange(hsv, lower_violet, upper_violet)
    blue = cv.inRange(hsv, lower_blue, upper_blue)
    black = cv.inRange(hsv, lower_black, upper_black)
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
            if tmpArea != 0:
                x = x_moment
                y = y_moment
                area = tmpArea
                mask = yellow
                break
            else:
                color = ""
                mask = None
                area = 0
        elif tmpArea != 0 and tmpArea > maxArea:
            tmpMask = yellow
            maxArea = tmpArea
            tmpColor = "yellow"
            tmpx = x_moment
            tmpy = y_moment

    for k in range(1):
        moments = cv.moments(red, 1)
        x_moment = moments['m10']
        y_moment = moments['m01']
        tmpArea = moments['m00']
        if color == "red":
            if tmpArea != 0:
                x = x_moment
                y = y_moment
                area = tmpArea
                mask = red
                break
            else:
                color = ""
                mask = None
                area = 0
        elif tmpArea != 0 and tmpArea > maxArea:
            tmpMask = red
            maxArea = tmpArea
            tmpColor = "red"
            tmpx = x_moment
            tmpy = y_moment


    if color == "":
        color = tmpColor
        mask = tmpMask
        area = maxArea
        x = tmpx
        y = tmpy

    if (area > 100) :
        x = int(x / area)
        y = int(y / area)
        cv.putText(frame, color, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        centre = (x, y)
        cnts = cv.findContours(mask, cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        print("Centre : " + str(centre) + ", farthest Point : " + str(extBot))
        cv.circle(frame, extBot, 5, [255, 0, 255], -1)
        lst = list(extBot)
        lst[1] = 450 - lst[1]
        extBot = tuple(lst)
        if isRealesed:
            pts.appendleft(extBot)

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue

                colorDraw = (0, 0, 255)
                if color == "yellow":
                    colorDraw = (0, 255, 255)
                cv.line(planeList, pts[i - 1], pts[i], colorDraw)

    cv.imshow("frame", frame)
    cv.imshow('yellowmask', yellow)
    cv.imshow('redmask', red)
    cv.imshow('draw', planeList)

    pressedKey = cv.waitKey(1)
    if pressedKey & 0xFF == ord('z'):
        break
    elif pressedKey & 0xFF == ord('r'):
        isRealesed = not isRealesed
        pts.clear()
    elif pressedKey & 0xFF == ord('c'):
        planeList = cv.imread("123.jpg", 1)
        pts.clear()

cv.destroyAllWindows()
videoCapture.release()
