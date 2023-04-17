"""
Sample example detection line, but it has some limitations.
eg:
    1. acc is not good
        we want to detection line on the rode, it detection all lines
    2. fine-tune is a problem

"""

import cv2
import numpy as np

image = cv2.imread('solidWhiteCurve.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                        threshold=100, minLineLength=100, maxLineGap=10)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
