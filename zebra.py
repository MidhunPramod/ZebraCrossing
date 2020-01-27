import cv2
import numpy as np

image = cv2.imread('input/zebra5.jpeg')
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
final_contours = []
centroid_y = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) < 7:
            new_contours.append(cnt)
            M = cv2.moments(cnt)
            # cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroid_y = centroid_y + cy

centroid_y = centroid_y/len(new_contours)


for cnt in new_contours:
    M = cv2.moments(cnt)
    cy = int(M['m01']/M['m00'])
    if abs(cy - centroid_y) < 50:
        final_contours.append(cnt)


cv2.drawContours(image, final_contours, -1, (0, 255, 0), 2)

cv2.imshow('zebra', mask)
cv2.imshow('original', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
