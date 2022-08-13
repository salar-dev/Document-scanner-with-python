# Instagram - https://www.instagram.com/salardev/
# Facebook - https://www.facebook.com/salar.brefki/

import cv2
import numpy as np

cap = cv2.VideoCapture(1)

def rescale_frame(frame, percent=75):
   width = int(frame.shape[1] * percent / 100)
   height = int(frame.shape[0] * percent / 100)
   dim = (width, height)
   return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def processing(img):
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imggray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)

    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), 30)
    return biggest

def reorder(points):
    points = points.reshape((4,2))
    newPoints = np.zeros((4,1,2),np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    newPoints[1]= points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints

def warp(img, biggest, imgSize):
    widthImg = imgSize[0]
    heightImg = imgSize[1]
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgCropped


while True:
    _, img = cap.read()
    imgSize = img.shape
    imgContour = img.copy()
    processrdImg = processing(img)

    biggest = getContours(processrdImg)

    print(getContours(processrdImg))

    if biggest.size != 0:
        imgWarped = warp(img, biggest, imgSize)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('myDoc.jpg', imgWarped)
        rescaleImgWarped = rescale_frame(imgWarped, percent=40)
        cv2.imshow('Doc', rescaleImgWarped)
    else:
        pass

    rescaleImg = rescale_frame(imgContour, percent=40)
    rescaleProcessrdImg = rescale_frame(processrdImg, percent=40)

    cv2.imshow('img', rescaleImg)
    cv2.imshow('processrdImg', rescaleProcessrdImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Salar Dev