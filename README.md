# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[solidwhite]: ./test_images_output/solidWhiteRight.jpg
[solidyellow]: ./test_images_output/solidYellowLeft.jpg
[curved1]: ./test_images_output/solidWhiteCurve.jpg
[curved2]: ./test_images_output/solidYellowCurve.jpg

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.
- Converting the images to grayscale.
- Applying Gaussian Blur to the grayscaled image.
- Detecting edges based on Canny Edge Detection.
- Getting only interested region.
    - This region must be a polygon that fully contains both left and right lanes and
      at the same time it must exclude other region potentially causing confusion
      (e.g., region considered as the lane like white car).
- Finding lines using Hough Transform with the input of detected edges
  (i.e., output of Canny Edge Detection).

The main challenge of the pipeline implementation is understanding high level concepts
to the implementation details including API usages and the determination of parameters.
To decide parameters, we must consider given conditions and meaning of each argument of
APIs.
- For Canny Edge Detection, since it takes advantage of the gradient and the range
of gray pixels must be \[0 - 255\], the threshold must be one of values in \[1 - 254\].
We can assume that bright lanes that humans can recognize (e.g., white, yellow) have
big enough gradients. Thus, we can set big thresholds (e.g., 150 for high\_threshold).
According to the know-how ratio between low\_threshold and high\_threshold (i.e., 1:3),
I set the low\_threshold as 1/3 of high\_threshold.
- For Hough Transform to find lines, I used (1, numpy.pi/180, 10, 15) as (rho, theta,
threshold, min\_line\_length). This setting decides it is a line if there are more than
10 points in the grid of (1, numpy.pi/180) and the line is bigger than 15. I assumed that
the line is a set of continuous points and "continuous points" must be close enough.
In other words, even small grid (e.g., 1, numpy.pi/180) must contain many points if it
is a line. I found the exact value based on trials and errors. I adopted the similar
intuition to set the min\_line\_length (i.e., how many "continuous points" are needed
for a line).

In order to draw a single line on the left and right lanes, I added the
connect\_lines() function. The basic assumptions are
- There are only two lanes (i.e., left and right).
- Each lane is a straight line (i.e., not curved)
- If we make a lane longer, it must be close to lines that is a part of the lane.
- Lines not contained in a lane must have different slopes from the slope of the lane
and the difference should be big enough.

Those assumptions are considered with virtual lanes, but we only have information for
lines found by Hough Transform (i.e., the result of the pipeline).
To get the lane, the function first finds the major line included in a lane. I considered
the major line must contain the maximum number of lines when extending it. Since two
lines are rarely intersected but can be considered as parts of the same lane, I allowed a
small error. For each line _L_, the function gets y-distance from the extended _L_ to
two end points (_x1_, _y1_), (_x2_, _y2_) of other lines.
For example, if _L_ is _slop * (x-x0) = (y-y0)_, the y-distance from (_x1_, _y1_) from
the extended _L_ is _abs(y1-y0-slop*(x1-x0))_.
After getting the first major line, it extends the major line to lines that have similar
slopes to the major line (by getting maximum and minimum x-values).
Then, it extends the major line to the bottom of the image (i.e., y-value == y-size).
It does the same thing to find the second major line, but only difference is to skip
lines that have not big enough slope difference from the first major line when searching
the second major line.

Results are shown as:

![alt text][solidwhite]
![alt text][solidyellow]


### 2. Identify potential shortcomings with your current pipeline

First, I assumed many things and it means there are many limitations in the current
implementation. For example, I assumed lanes are straight. It does not work well for
curved lanes.

Second, I fixed parameters (e.g., for Canny Edge Detection and Hough Transform). I
guess this works fine for a small set of images, but the performance will be seriously
bad when testing it against a large set of images. For example, the region of interest
is very critical to the result. If some images having many white objects (e.g., white
cars) in the fixed region of interest, then there will be so many detected edges from
Canny Edge Detection.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to find a better transform to detect curved lanes.
I am not sure but I think we need another mathematical theory to handle curved lanes.
In particular, when the angle of curve is changed quickly (e.g., small circle), I
guess Hough Transform cannot find the curved lane (even when we set the parameters
very well).

Another potential improvement could be the adaptive parameter selection. We need an
algorithm to automatically determine better parameters for the given input.
