import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


points_pickle = pickle.load( open( "Adv_Lane_Detection/object_and_image_points.pkl", "rb" ) )
chess_points = points_pickle["chesspoints"]
image_points = points_pickle["imagepoints"]
img_size = points_pickle["imagesize"]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(chess_points, image_points, img_size, None, None)

camera = {}
camera["mtx"] = mtx
camera["dist"] = dist
camera["imagesize"] = img_size
pickle.dump(camera, open("Adv_Lane_Detection/camera_matrix.pkl", "wb"))

def distort_correct(img,mtx,dist,camera_img_size):
    img_size1 = (img.shape[1],img.shape[0])
    #print(img_size1)
    #print(camera_img_size)
    assert (img_size1 == camera_img_size),'image size is not compatible'
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


img = mpimg.imread('Adv_Lane_Detection/camera_cal/calibration2.jpg')
img_size1 = (img.shape[1], img.shape[0])

undist = distort_correct(img, mtx, dist, img_size)

### Visualize the captured 
plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Captured Image', fontsize=30)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=30)
plt.tight_layout()
plt.savefig('Adv_Lane_Detection/out_images/undistorted_chess.png')