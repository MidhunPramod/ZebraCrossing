import cv2
import numpy as np

image = cv2.imread('input/zebra5.jpeg')
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayimage = cv2.bilateralFilter(grayimage, 9, 75, 75)
kernel = np.ones((2, 2), np.uint8)

edges = cv2.Canny(grayimage, 100, 200)
dilation = cv2.dilate(edges, kernel, iterations=2)
dilation = cv2.erode(dilation, kernel, iterations=1)

_, contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

new_contours = []


for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            new_contours.append(cnt)

cv2.drawContours(image, new_contours, -1, (0, 255, 0), 2)

cv2.imshow('original', image)
cv2.imshow('canny', dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
