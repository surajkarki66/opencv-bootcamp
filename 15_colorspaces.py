import os
import cv2

img = cv2.imread(os.path.join("images", "van.jpg"))
img = cv2.resize(img, (600, 600))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


cv2.imshow("Original Image", img)
cv2.imshow("RGB Image", img_rgb)
cv2.imshow("Grayscale Image", img_gray)
cv2.imshow("HSV Image", img_hsv)

cv2.waitKey(0)
