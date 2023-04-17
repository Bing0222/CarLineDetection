# Review some steps using detection line
<!-- TOC -->

- [Review some steps using detection line](#review-some-steps-using-detection-line)
    - [Simple Lane Dtection](#simple-lane-dtection)
        - [Some knowledge points](#some-knowledge-points)
        - [steps](#steps)
    - [Adv Lane Detction](#adv-lane-detction)
        - [Some knowledge points](#some-knowledge-points)
    - [Vehicel Detection and Line](#vehicel-detection-and-line)
    - [Features Detection](#features-detection)

<!-- /TOC -->

## Simple Lane Dtection

### Some knowledge points

    - grayscale
    - canny detection
    - gaussian blur
    - region of interset
    - get slope
    - draw line
    - hough line
    - weighted image

### steps

    - grayscale
    - rgb-->hsv
    - modify color range
    - get mask
    - combine and gussian blur
    - canny edges
    - get region of interset
    - hough_line --> get line
    - pic add

## Adv Lane Detction

### Some knowledge points

    - Camera Calibration
         We need to store the number of object points, or locations of interest on the chessboard. Only points that are not on the outermost edge are considered. For this board, there are 9 columns on the X-axis and 6 rows on the 7 axis. We will want a list of (x, y, z) coordinates for each combination (i.e. unique location on the board). Every z-value will be zero since it is a 2D image.
      success, corners = cv.findChessboardCorners(gray, (9,6), None)

    - Distortion Correction
        get undistorted
        pickle.dump(camera, open("camera_matrix.pkl", "wb"))

    - Gradient & Color Thresholding
        For this project we will be using several filters to generate a binary (black & white) image that we can detect and track the lane lines with.
            . Sobel gradients in the x & y directions
                steps:
                1. get distort_correct image (Undistorted Image)
                2. sobel gray image --> rescale --> binary image
            . Gradient magnitude
                steps:
                1. Calculate the xy magnitude
                2. Scale to 8-bit (0 - 255) and convert to type = np.uint8
            . Gradient direction
                steps:
                1. Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
            . Color space transform and filtering
                The gradient filters above all convert the original image into grayscale and a lot of useful information is lost. Lane lines are either yellow or white, and we can use that to our advantage trying to locate and track them. The Hue Saturation Lightness color space will help. In particular, the S channel of an HSL image retains a lot information about lane lines - especially when there are shadows on the road. The Red channel of RGB also does a good job of creating binary images of lane lines
                steps:
                    1.  Apply a threshold to the S channel
                    2. Apply a threshold to the R channel

    - Combining Filter Methods


    - Perspective transform

    - Detecting Lane Lines
        Now that our snapshot of the road has been transformed, we are ready to find and track the actual lane lanes. To do this, we can identify areas with a high count of white pixels. Specifically, we will use a histogram to bin the pixels into columns and look for peaks in the distribution.
        We can clearly see where the lanes are in this histogram. By implementing a sliding window, we can track the peaks as they veer left/right due to the slight curvature of the road.

        The function below works as follows:
        Create a search window on the bottom of the image whose height is 1/9 of the image's height.
        Split the window into left and right halves.
        Locate the pixel column with the highest value via histogram.
        Draw a box around that area using a margin variable.
        Identify all of the non-zero pixels in that box. If there are enough, center the box on their mean position for the next window.
        Fit a quadradtic equation to all of the non-zero pixels identified in each half of the image (left lane and right lane)


    Main steps:
        1. load camera matrix and distortion matrix
        2. correct lens distortion
        3. get binary image
        4. prespective transform
        5. window search
        6. draw polygon
        7. update
        8. print/show

## Vehicel Detection and Line

## Features Detection
