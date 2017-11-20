import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# This method gets as input a list of filenames of pictures of a chessboard, and outputs an array of object points
# (the chessfield corner coordinates) and their corresponding pixel coordinates per image. Does this using
# cv2#findChessboardCorners.

objpoints = []
imgpoints = []


def find_all_chessboard_corners(images, plot=False):
    nx = 9
    ny = 6

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    global objpoints
    global imgpoints

    for filename in images:
        img = cv2.imread(filename)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny))

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            if plot:
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.imshow(img)
                plt.show()

    return objpoints, imgpoints


# This function returns the undistorted version of the image in found in the provided filename path. Undistorts
# according to the input objpoints and imgpoints, using cv2#undistort.
def undistort(img, objpoints, imgpoints, plot=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.show()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if plot:
        plt.imshow(img)
        plt.show()
        plt.imshow(undist)
        plt.show()
    return undist


def perspective_transform(undist, plot=True):
    src = np.float32([[600, 440], [675, 440], [1120, 675], [200, 675]])
    dst = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]))
    if plot:
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.show()
    return warped


# Abs_sobel_thresh, as defined in the Udacity SDC course (Advanced Lane Finding, slide 21).
def abs_sobel_thresh(img, orient, thresh_min=20, thresh_max=100, plot=False):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(binary_output, cmap='gray')
        ax2.set_title('Thresholded Sobel Gradient', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # Return the result
    return binary_output


def saturation_filter(img, thresh_min=120, thresh_max=255, plot=False):
    S = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh_min) & (S <= thresh_max)] = 1

    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(binary_output, cmap='gray')
        ax2.set_title('Thresholded Saturation Gradient', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # Return the result
    return binary_output


# Histogram/sliding windows lane lines procudure, as defined in the Udacity SDC course
# (Advanced Lane Finding, slide 33).
def find_lane_lines(image):
    undist = undistort(image, objpoints, imgpoints)

    transformed = perspective_transform(undist)

    sobel = abs_sobel_thresh(transformed, 'x')
    sat = saturation_filter(transformed)

    binary_warped = np.zeros_like(sat)
    binary_warped[(sat == 1) | (sobel == 1)] = 1

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 12
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = (
                        (1 + (
                        2 * left_fit_cr[0] * np.max(lefty) * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (
        2 * right_fit_cr[0] * np.max(righty) * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(out_img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    src = np.float32([[560, 440], [730, 440], [1120, 675], [230, 675]])
    dst = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])

    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (out_img.shape[1], out_img.shape[0]))
    # Combine the result with the original image
    result = cv2.cvtColor(cv2.addWeighted(undist, 1, newwarp, 0.3, 0), cv2.COLOR_BGR2RGB)

    offset = (640 - ((rightx[0] + leftx[0]) / 2)) * xm_per_pix

    cv2.putText(result, 'Camera {:.2f} meter to the right'.format(offset), (80, 80), fontFace=16, fontScale=1,
                color=(0, 0, 0), thickness=3)
    cv2.putText(result, 'Radius of curvature of {:.2f} meter'.format((left_curverad + right_curverad) / 2), (80, 120),
                fontFace=16, fontScale=1, color=(0, 0, 0), thickness=3)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # plt.imshow(result)

    return result


#########################
#  START OF MAIN LOGIC  #
#########################

nx = 9  # Number of inside corners along the x-axis
ny = 6  # Number of inside corners along the y-axis

# Prepare the chessboard corners for the objpoints: (0, 0, 0), (1, 0, 0), (2, 0, 0), ..., (nx - 1, ny - 1, 0)
images = glob.glob('camera_cal/*.jpg')

objpoints, imgpoints = find_all_chessboard_corners(images)

# testimages = glob.glob('test_images/*.jpg')
# for img in testimages:
#    find_lane_lines(cv2.imread(img))

clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(find_lane_lines)
white_clip.write_videofile('lane_lines.mp4', audio=False)
