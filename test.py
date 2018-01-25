#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def get_slop(line):
    x1,y1,x2,y2 = line
    return (y2-y1)/(x2-x1)

def get_center(line):
    x1,y1,x2,y2 = line
    return (x1+x2)/2, (y1+y2)/2

def connect_lines(lines, xsize, ysize):
    """
    connect/average/extrapolate line segments
    """
    # expected maximum difference of slopes between a lane and a line in the lane
    merge_threshold = 20

    # find major line 1
    major_line1 = []
    max_score = 0
    for line1 in lines:
        score = 0
        slop = get_slop(line1[0])
        x,y,_,_ = line1[0]
        for line2 in lines:
            x1,y1,x2,y2 = line2[0]

            # If a point is located in a line m(x-x1) = y-y1,
            # we expect y-y1-m(x-x1) == 0. This error values check
            # how far the point is from the line.
            err1 = abs(y1-y-slop*(x1-x))
            err2 = abs(y2-y-slop*(x2-x))
            if err1 < merge_threshold and err2 < merge_threshold:
                score = score + 1
        if score > max_score:
            max_score = score
            major_line1 = line1[0]

    # make the lane cover all its lines
    major_line1_slop = get_slop(major_line1)
    x,y,_,_ = major_line1
    maxx, minx = 0, xsize
    for line in lines:
        x1,y1,x2,y2 = line[0]
        err1 = abs(y1-y-major_line1_slop*(x1-x))
        err2 = abs(y2-y-major_line1_slop*(x2-x))
        if err1 < merge_threshold and err2 < merge_threshold:
            if x1 > x2:
                x2,y2,x1,y1 = x1,y1,x2,y2
            if maxx < x2:
                maxx = x2
            if minx > x1:
                minx = x1
    major_line1 = [minx, y+major_line1_slop*(minx-x),
            maxx, y+major_line1_slop*(maxx-x)]

    # extend the lane to the bottom of the image (i.e., y = ysize-1)
    x1,y1,x2,y2 = major_line1
    if y1 > y2:
        x2,y2,x1,y1 = x1,y1,x2,y2
    x2,y2 = (ysize-1-y2)/major_line1_slop+x2, ysize-1
    major_line1 = [x1,y1,x2,y2]

    # expected minimum slope difference between two lanes
    diff_threshold = 100

    # find major line 2
    major_line2 = []
    max_score = 0
    major_line1_slop = get_slop(major_line1)
    mx,my,_,_ = major_line1
    for line1 in lines:
        x1,y1,x2,y2 = line1[0]
        err1 = abs(y1-my-major_line1_slop*(x1-mx))
        err2 = abs(y2-my-major_line1_slop*(x2-mx))
        if err1 < diff_threshold and err2 < diff_threshold:
            continue
        score = 0
        for line2 in lines:
            slop = get_slop(line1[0])
            x,y,_,_ = line1[0]
            x1,y1,x2,y2 = line2[0]
            err1 = abs(y1-y-slop*(x1-x))
            err2 = abs(y2-y-slop*(x2-x))
            if err1 < merge_threshold and err2 < merge_threshold:
                score = score + 1
        if score > max_score:
            max_score = score
            major_line2 = line1[0]

    # make the lane cover all its lines
    major_line2_slop = get_slop(major_line2)
    x,y,_,_ = major_line2
    maxx, minx = 0, xsize
    for line in lines:
        x1,y1,x2,y2 = line[0]
        err1 = abs(y1-y-major_line2_slop*(x1-x))
        err2 = abs(y2-y-major_line2_slop*(x2-x))
        if err1 < merge_threshold and err2 < merge_threshold:
            if x1 > x2:
                x2,y2,x1,y1 = x1,y1,x2,y2
            if maxx < x2:
                maxx = x2
            if minx > x1:
                minx = x1

    # extend the lane to the bottom of the image (i.e., y = ysize-1)
    major_line2 = [minx, y+major_line2_slop*(minx-x),
            maxx, y+major_line2_slop*(maxx-x)]
    x1,y1,x2,y2 = major_line2
    if y1 > y2:
        x2,y2,x1,y1 = x1,y1,x2,y2
    x2,y2 = (ysize-1-y2)/major_line2_slop+x2, ysize-1
    major_line2 = [x1,y1,x2,y2]

    return np.array([[major_line1], [major_line2]], dtype=np.int32)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    major_lines = connect_lines(lines, img.shape[1], img.shape[0])
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, major_lines, [255,0,0], 12)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

import os
os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    ysize = image.shape[0]
    xsize = image.shape[1]
    gray = grayscale(image)

    kernel_size = 5
    blur = gaussian_blur(gray, kernel_size)

    low_threshold = 50
    high_threshold = 150
    edges = canny(blur, low_threshold, high_threshold)

    # FIXME
    interested = region_of_interest(edges,
            [np.array([[(0, ysize-1),(450, 320), (490, 320), (xsize-1, ysize-1)]],
                dtype=np.int32)]
            )

    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_length = 15
    max_line_gap = 20
    hough = hough_lines(interested, rho, theta, threshold, min_line_length, max_line_gap)

    result = weighted_img(image, hough)
    return result

for file_name in os.listdir("test_images/"):
    #reading in an image
    image = mpimg.imread('test_images/' + file_name)
    # plt.savefig('test_images_output/' + file_name)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
