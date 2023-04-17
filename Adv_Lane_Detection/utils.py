""""
Calibration

"""
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import pickle


images = glob.glob('Adv_Lane_Detection/camera_cal/calibration*.jpg')
img = mpimg.imread(images[0])
# cv2.imshow("img",img)
# cv2.waitKey(0)


# store chessboard coordinates
chess_points = []
# store points from transformed img
image_points = []

# board is 6 rows by 9 columns. each item is one (xyz) point 
chess_point = np.zeros((9*6,3),np.float32)
chess_point[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

for image in images:
    img = mpimg.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    success, corners = cv2.findChessboardCorners(gray,(9,6),None)

    if success:
        image_points.append(corners)
        chess_points.append(chess_point)
    else:
        print('corners not found{}'.format(image))
"""
corners not foundAdv_Lane_Detection/camera_cal\calibration1.jpg
corners not foundAdv_Lane_Detection/camera_cal\calibration4.jpg
corners not foundAdv_Lane_Detection/camera_cal\calibration5.jpg
"""

image = mpimg.imread('Adv_Lane_Detection/camera_cal/calibration2.jpg')

plt.figure()
f,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
ax1.imshow(image)
ax1.set_title('Captured Image', fontsize=30)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
ret , corners = cv2.findChessboardCorners(gray,(9,6),None)    
if ret == False:
    print('corners not found')
img1 = cv2.drawChessboardCorners(image,(9,6),corners,ret) 

ax2.imshow(img1)
ax2.set_title('Corners drawn Image', fontsize=30)
plt.tight_layout()
plt.savefig('Adv_Lane_Detection/out_images/chess_corners.png')


points_pkl = {}
points_pkl["chesspoints"] = chess_points
points_pkl["imagepoints"] = image_points
points_pkl["imagesize"] = (img.shape[1], img.shape[0])
pickle.dump(points_pkl,open("Adv_Lane_Detection/object_and_image_points.pkl", "wb" ))