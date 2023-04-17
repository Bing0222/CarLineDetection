import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_iamge = cv2.bitwise_and(img, mask)
    return masked_iamge


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# check line in binary draw


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    img: binary image
    rho: 表示以像素为单位的距离精度，一般取1。
    theta：表示以弧度为单位的角度精度，一般取$\pi/180$
    threshold：表示累加器中的值大于或等于此值时才认为是一条直线。
    minLineLength：表示检测到的直线的最小长度，比这个短的直线会被忽略。
    maxLineGap：表示同一直线上的两点之间允许的最大间隔。

    return: line_img
    line_img: 画出直线
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    add by weighted 
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_frame(img):
    global first_frame

    gray_img = grayscale(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    # 图像限制在lower和upper 之间
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_img, 200, 255)
    mask_yw = cv2.bitwise_or(mask_yellow, mask_white)
    mask_yw_img = cv2.bitwise_and(gray_img, mask_yw)

    kernel_size = 5
    guss_gray = gaussian_blur(mask_yw_img, kernel_size)

    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(guss_gray, low_threshold, high_threshold)

    imshape = img.shape
    lower_left = [imshape[1]/9, imshape[0]]
    lower_right = [imshape[1]-imshape[1]/9, imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/8, imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/8, imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left, top_left, top_right,
                         lower_right], dtype=np.int32)]
    roi_image = region_of_interest(canny_edges, vertices)

    rho = 4
    theta = np.pi/180

    threshold = 30
    min_line_len = 100
    max_line_gap = 280

    line_image = hough_lines(roi_image, rho, theta,
                             threshold, min_line_len, max_line_gap)
    result = weighted_img(line_image, img, α=0.8, β=1., λ=0.)
    return result


for source_img in os.listdir("Basic_Lane_Detection/test_images/"):
    image = mpimg.imread("Basic_Lane_Detection/test_images/"+source_img)
    processed = process_frame(image)
    mpimg.imsave("Basic_Lane_Detection/test_images/annotated_" +
                 source_img, processed)
