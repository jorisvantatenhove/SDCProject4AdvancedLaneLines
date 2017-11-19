## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/chessboard_corners.png "Chessboard corners 1"
[image2]: ./examples/chessboard_corners2.png "Chessboard corners 2"
[image3]: ./examples/frame1_input.png "Frame 1 input"
[image4]: ./examples/frame1_corrected.png "Frame 1 corrected"
[image5]: ./examples/frame1_warped.png "Frame 1 warped"
[image6]: ./examples/frame1_sobel.png "Frame 1 sobel"
[image7]: ./examples/frame1_thresholded.png "Frame 1 saturation"
[image8]: ./examples/frame1_binary.png "Frame 1 merged binary image"
[image9]: ./examples/frame1_output.png "Frame 1 output"


### Project Overview

In the fourth project, the goal is similar as the first project: we again try to automatically define lane lines. This time however, we use more advanced techniques and try to take road curvature into account as well.

We first preprocess the input images in order to get rid of some camera artifacts, and then transform the image and fit a second degree polynomial to the two most well defined lines in the image.

In the preproccesing step, we load calibration images of a chessboard patterns shot from different angles, in order to calculate the distortion.

After calculating the distortion, we know how we can warp the image our camera shoots to more accurately reflect reality. On these warped images, we detect lane lines using the following procedure:

* Perform a perspective transform on these images to get rid of unnecessary footage and also make it more easier to fit polynomials to the lane lines
* Use thresholded color transformations and Sobel gradients to create a binary image
* Detect lane lines in the binary image
* Determine curvature of the lane, and the vehicles position relative to the lane
* Wrap the detected lane lines back to the original image
* Display the detected lane lines, curvature and vehicle position

The result of this procedure on a sample video can be found in `lane_lines.mp4`. The code is found in `project.py`.

### Step by step description

First, we calibrate the camera using 20 input photos of chessboards. Since we know that these should be flat, we can calculate the distortion in the sides of the picture. We can automatically detect the chessboard corners. Here are two examples of the images where we've detected the corners of the chess fields:

![Chessboard example 1][image1]

![Chessboard example 2][image2]

After calibration the camera, we can start the real work! We show the process by following what happens to the first frame:

![Input frame][image3]

After correcting for the distortion, we find the following image:

![Corrected frame][image4]

Then, we warp the section of the image where we expect the lane lines to be to a birds eye view:

![Warped frame][image5]

We apply two kinds of lane detection. The first one is a Sobel gradient in the x-direction:

![Sobel transformation][image6]

The second one uses the Saturation channel of the image converted to the HLS colour scheme:

![Saturation transformation][image7]

Then, we merge the two binary images into a single binary image:

![Merged binary image][image8]

Using the histogram and sliding window search as described in slides 33 and 34 of the Advanced Lane Finding course, we detect the lane lines, warp them back and plot them in the original image.

Also, we measure the offset of the car by averaging the position of the right and left lane lines to detect the middle of the lane, and compare this to the middle frame on 640 pixels in the x-axis.

Lastly, we also measure the radius of curvature as defined by the formula in slide 35 in the Advanced Lane Finding course. 

Taking this together, we find the following frame:

![Output frame][image9]

Applying this procedure to every single frame gives us the result that can be found in the video!