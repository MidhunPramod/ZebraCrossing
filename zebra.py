import cv2
import numpy as np

image = cv2.imread('zebra4.jpeg')
hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
kernel = np.ones((3, 3), np.uint8)

# Works properly
# lower_white = np.array([0, 0, 195])
# upper_white = np.array([179, 40, 255])

lower_white = np.array([0, 0, 170])
upper_white = np.array([179, 40, 255])


hsvimage = cv2.bilateralFilter(hsvimage, 9, 75, 75)
mask = cv2.inRange(hsvimage, lower_white, upper_white)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# erosion = cv2.erode(mask, kernel, iterations=1)
# dilation = cv2.dilate(erosion, kernel, iterations=1)
# blur = cv2.bilateralFilter(dilation, 9, 75, 75)

_, contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

new_contours = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) < 7:
            # rect = cv2.minAreaRect(cnt)
            # box = cv2.boxPoints(rect)
            # box = box.astype('int')
            # new_contours.append(box)
            new_contours.append(cnt)
            # extLeft = cnt[cnt[:, :, 0].argmin()][0]
            # extRight = cnt[cnt[:, :, 0].argmax()][0]
            # extTop = cnt[cnt[:, :, 1].argmin()][0]
            # extBot = cnt[cnt[:, :, 1].argmax()][0]

            # print(extTop, end=' ')
            # print(extRight, end=' ')
            # print(extBot, end=' ')
            # print(extLeft, end=' ')

            # new_contours.append(
            #     np.array([[extTop, extRight, extBot, extLeft]]))
            # print()


cv2.drawContours(image, new_contours, -1, (0, 255, 0), 2)

cv2.imshow('zebra', mask)
cv2.imshow('original', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
