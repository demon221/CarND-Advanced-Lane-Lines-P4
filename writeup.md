**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/writeup_images/undistort.png "Undistorted"
[image2]: ./output_images/writeup_images/undistort_test_image.png "Road Transformed"
[image3]: ./output_images/writeup_images/binary_combination.jpg "Binary Combination Example"
[image4]: ./output_images/writeup_images/binary_color.jpg "Binary Color Example"
[image5]: ./output_images/writeup_images/binary_result1.jpg "Binary Result for Test Image 1"
[image6]: ./output_images/writeup_images/binary_result2.jpg "Binary Result for Test Image 2"
[image7]: ./output_images/writeup_images/binary_result3.jpg "Binary Result for Test Image 3"
[image8]: ./output_images/writeup_images/binary_result4.jpg "Binary Result for Test Image 4"
[image9]: ./output_images/writeup_images/binary_result5.jpg "Binary Result for Test Image 5"
[image10]: ./output_images/writeup_images/binary_result6.jpg "Binary Result for Test Image 6"
[image11]: ./output_images/writeup_images/binary_color_harder_challenge.png "Binary Color Example in harder challenge"
[image12]: ./output_images/writeup_images/opening_harder_challenge.png "Openging Example in harder challenge"
[image13]: ./output_images/writeup_images/warp_test_image.png "Warp Example"
[image14]: ./output_images/writeup_images/warp_test_image_challenge.png "Warp Example in challenge"
[image15]: ./output_images/writeup_images/warp_test_image_hard.png "Warp Example in harder challenge"
[image16]: ./output_images/writeup_images/binary_line_fit.png "Binary Line Fit Example"
[image17]: ./output_images/writeup_images/result_test_image.png "Result Example"
[image18]: ./output_images/writeup_images/result_test_image_debug.png "Result Example with Debug"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
### Writeup / Submission

#### 1. Writeup

Here I will introduce the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.
#### 2. Submission Files
You can find all project with submission files in this Github Repository. (https://github.com/demon221/CarND-Advanced-Lane-Lines-P4.git)

You can also find the link of what it includs:

1. writeup (https://github.com/demon221/CarND-Advanced-Lane-Lines-P4/blob/master/writeup.md)
2. code in Jupyter notebook (https://github.com/demon221/CarND-Advanced-Lane-Lines-P4/blob/master/P4.ipynb)
3. example output images (https://github.com/demon221/CarND-Advanced-Lane-Lines-P4/tree/master/output_images)
4. output video (https://github.com/demon221/CarND-Advanced-Lane-Lines-P4/tree/master/output_videos)

In the example output images, you can find all the middle processed images for test images and images captured from the video.
- [/camera_cal](https://github.com/demon221/CarND-Advanced-Lane-Lines-P4/tree/master/output_images/camera_cal) :      result images of camera calibration
- [/challenge_images](https://github.com/demon221/CarND-Advanced-Lane-Lines-P4/tree/master/output_images/challenge_images): final and middle step results of images captured from challenge video
- [/hard_images](https://github.com/demon221/CarND-Advanced-Lane-Lines-P4/tree/master/output_images/hard_images):      final and middle step results of images captured from harder challenge video
- [/test_images](https://github.com/demon221/CarND-Advanced-Lane-Lines-P4/tree/master/output_images/test_images):      final and middle step results of test images

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Before starting the implementation of the lane detection pipeline, the first thing that should be done is camera calibration.
The code for this step is contained in the 2nd code cell of the IPython notebook located in "P4.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

From the calibration images I set the chessboard size to `9x6` for the project.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

The calibration values are stored in file "camera_dist_pickle.p". For one camera it is only needed to calibrate once, these calibration values can be used for further usage. After the matrix and coeeficients are calculated, we will use them to undistort an image to demonstrate their correctness.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one.

1. Load the distortion matrix from file "camera_dist_pickle.p" which is stored in previous "Camera Calibration".
2. Apply this distortion correction to the test image using the cv2.undistort() function and obtained this result: 

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This part include two phase of work in this project. For the harder challenge video, some more image processing functions are used.

#### - Test images, 1st and 2nd videos:
The code for these operations is presented under "Create of thresholded binary image" header, which in the 4th code cell of the IPython notebook.

Different gradient threshold methods are used:
1. Sobel transformation (including X with Y threshold, and magnitude with direction threshold)
2. Color in range of HSV color space threshold
3. Saturation, hue, lightness of HLS color threshold
4. Red channels binary threshold
5. Combination of color threshold to distinguish yellow and white lines:
    - A combination of the yellow in HSV color space and saturation channels for the yellow line.
    - A combination of the white in HSV color space and red channels for the white line.

A function called `combined_thresh()` makes a combination with the sobel gradient function `abs_sobel_thresh()`, `mag_sobel_thresh`, and `dir_sobel_thresh`, as well as the color threshold function `color_range()`, `s_thresh()`, and `r_thresh()`.

Another function called `color_thresh()` only takes the color threshold for yellow color and white color combination function `combined_color()`, it makes a liner combination of color range, saturation and red channel for each color of yellow and white.
```python
# Combine color binary of yellow and white
def combined_color(img):
    # Get S channel from HLS space
    s_channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    # Get R channel from RGB space
    r_channel = img[:,:,0]
    # Get yellow and white in range from HSV space
    y_bin, w_bin = color_range(img)

    combined_channel = (s_channel + 0.5 * r_channel) / 1.5
    yellow_channel = 0.3 * y_bin + 0.6* s_channel + 0.3 * r_channel
    white_channel = 0.2 * w_bin + 0.8 * r_channel
```


The different effects of the two functions are compared as following:

- Combination of sobel and color threshold method.

![alt text][image3]
- Color threshold for yellow and white method, without sobel gradient.

![alt text][image4]

In good situation of test images and project test video, it can be found that with color threshold can already detect the lane lines very well and nearly no noise compared to with sobel one.

Here's the results of all the test images for this step:

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

In the result of binary images of test images, color threshold method can also recognize the lane lines with good effort. Mostly, the edges of road extracted by sobel threshold and noised of the surface of road are eliminated.

Unfortunately, the best combination of these filters which is able to separate pixels of lane line from background on snapshots for all three videos was not found.
Optimized methods shall be applied to solve following problems:
- Shadows and glares are quite challenging. (especially for the harder challenge video)
- Sobel transformation may import noise (such as edge of the road or different material of road surface in the challenge video).

So in the video pipeline, the sobel threshold method will not be used.

#### - Harder challenge video:
The color threshold is further optimized for harder challenge images.(captured from the harder challenge video)

I pick the threshold values in each range of intense lightness and dark lightness.

| Color Space        | Dark yellow   | Light yellow  |  Dark white   |  Light white  |
|:------------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| HSV                |[20, 30, 50]~[30, 255, 180]|[20, 30, 50]~[30, 255, 255]|[0, 0, 40]~[255, 60, 100]|[0, 0, 200]~[255, 40, 255]|
| HSL: L             | 0 ~ 80                    | 80 ~ 255                  | 80 ~ 255                | 200 ~ 255                |
| HSL: S             | 0 ~ 90                    | 90 ~ 255                  | N/A                     | 90 ~ 255                 |
| RGB: R             | N/A                       | N/A                       | N/A                     | 200 ~ 255                |

Here I chose a threshold binary result from a image captured from the harder challenge video, which indicates result of a shadow road:

![alt text][image11]

To eliminate glares which brings some large region of white color, I chose a opening calculation of the warped binary images. It can extract the shape of line, eliminate large glares noises.

```python
# Shape of line extraction
def extrat_line(img, ksize=(55,55)):
    # Opening calculation to remove small objects
    blur = cv2.medianBlur(img, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    open_img = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    img = cv2.addWeighted(img, 1, open_img, -1, 0)

    return img, open_img
```

Here is a result for the opening calculation. Only lane lines are reserved in the glares.

![alt text][image12]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transform is presented under "Perspective transform to a bird-eye view" header, includes a function called `perspective_transform()`, which in the 8th code cell of the IPython notebook.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.

#### - Test images:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 188, 720      | 310, 720      |
| 1125, 720     | 1010, 720     |
| 593, 450      | 310, 0        |
| 687, 450      | 1010, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
The two straight images with perspective transform are shown in following. With the same matrix the two images can both transform to mostly straight lines in the bird-eye view.

![alt text][image13]

#### - Challenge video:

For the later video processing of challenge video, the perspective transform matrix needs to adapt to the lane position. After several trial the source and destination points for challenge video is chosen to:

| Source        | Destination   |
|:-------------:|:-------------:|
| 285, 720      | 310, 720      |
| 1125, 720     | 1010, 720     |
| 595, 480      | 310, 0        |
| 725, 480      | 1010, 0       |

The straight images is captured during the end of challenge video, the warped image is shown as following.

![alt text][image14]

#### - Harder challenge video:

For the video processing of harder challenge video, the perspective transform matrix needs to eliminate the head of the car. So we shall decrease the bottom limitation of the source edge.
After several trial the source and destination points for challenge video is chosen to:

| Source        | Destination   |
|:-------------:|:-------------:|
| 313, 680      | 310, 720      |
| 1057, 680     | 1010, 720     |
| 572, 480      | 310, 0        |
| 775, 480      | 1010, 0       |

The straight images is captured during the middle of harder challenge video, the warped image is shown as following.

![alt text][image15]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This part is presented under the header of "Identify lane-line pixels and fit their positions with a polynomial", from cell 11.

The method how I has fitted my lane lines with a 2nd order polynomial, has similar principle and structure with codes proposed by Udacity.

There are three parts functions for the lane pixel fingding.

1. Search Lane pixels from scratch

To search lane pixels from scratch, we use a sliding window function called `sliding_window_search()`. We basically find two prominent peaks from a histogram within each sliding window and use those peak as the location of the left and right lane. The image is divided to several layers from bottom to top.
After one window is detected due to enough nonzero points found, the window will slide to the next layer. This sliding window detect method is repeated from the bottom layer to the top layer of the image. Each window switch a margin around the previous window. The nonzero points are stored in the line indices.
A 2-D polynomial is used to fit the lane line with these nonzero points.

2. Polynomial search from previous coefficients

If polynomial coefficients is checked to be effective from the previous lane line, we can just search in a margin around the previous line position. The function name is called `prev_search()`. With the nonzero points around the previous line, the 2-D polynomial is used again to fit the new lane line.

3. Visualization the line and ROI region

With calculated polynomial coefficients, the poly fit curve of line and lane region will be visulizated on the warped binary images. It locates in function `draw_line_fit()` and `draw_line_region`.

The identify result of the test image is shown like this:

![alt text][image16]

In this part, the parameter of `margin` is best to set as an adaptive parameter. In the case of highway, such as test video and challenge video, `margin` can be adapted bit smaller. But in the case of country road such as harder challenge video, `margin` shall be adapted larger.
In my project this parameter is defined by the value of 100 in first two videos, and 150 in the harder challenge video.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 13 of the jupyter notebook.
Assuming that the lane is about 40 meters long and 3.7 meters wide, conversion factors in x and y from pixels space the real world meters are defined in `ym_per_pix` and `xm_per_pix`.
```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 40/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

With the polynomial fitting factor A, B and C in the 2-D polynomial curve f(y)=Ay^2+By+C, the radius of curvature calculation is followed as [tutorial](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).

Assuming the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines. The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane.
So using the bottom y pixel value of the warped image in the 2-D polynomial curve f(y)=Ay^2+By+C, we can calculate the bottom x pixel value and its offset with center point.

The result of radius of curvature and offset distance are calculated by the mean value of results of left line and right line.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 14 in the function `draw_lane_region()` and `put_curvature_and_offset()`.
Function `draw_lane_region()` is used to transform the fitted curve of lane and sliding windows or region of interests with invert matrix of the perspective transform.
Function  `put_curvature_and_offset()` is used to put the curvature and offset onto the images.

At last in cell 15 the function `find_lane_pipeline()` is the pipeline for lane detection.

In general, the detect lane pipeline includes these steps:
1. Undistort the original image
2. Binary gradient of the undistorted image
3. Perspective transform of the binary image
4. Find lane from the perspective transform image
5. Plot the lane result with curvature and offset back to the undistorted image

Here is an example of my result on a test image:

![alt text][image17]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

In order to produce a more stable and robust video output, I create an class called `Line()` in cell 17. Its responsibilities include:
1. Make a queue of N iterations to keep track of lane polynomial points and coefficients for previous N frame
2. Update the current lane polynomial points and coefficients from sliding window search function or previous search function
3. Calculate current radius of curvature and distance from center of road for sanity check
4. After sanity check passes, the rational lane polynomial points and coefficients can be added to the queue of high-confidence results, otherwise one most previous results will be pop out
5. Calculate an average polynomial coefficients over the last N iterations which passes the sanity check


```python
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, n=10):
        # number of queue to store data of last iterations
        self.n = n
        # number of fits in last iterations buffer
        self.n_buffered = 0
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n iterations
        self.recent_xfitted = deque([], maxlen=n)
        # polynomial coefficients of the last n iterations
        self.recent_fit = deque([], maxlen=n)
        # average x values of the fitted line over the last n iterations
        self.best_xval = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0])
        # x values of the most recent fit
        self.current_xval = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # pixels of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
```

I also create two functions of `sanity_check()` and `find_lane()` in cell 18. For coding optimization, class `Line()` is used for left line and right line.

`sanity_check()`

Each time a new measurement is got, a sanity check should be used to check the detection makes sense. To confirm that the detected lane lines are real, the sanity check will consider:
- Checking that they have similar curvature
- Checking that they are separated by approximately the right distance horizontally
- Checking that they are roughly parallel

```python
curv_ratio_threshold = 10 # Threshold of the curvature ration between left and right line [1]
distance_max_threshold = 1000 # Threshold of the max distance between left and right line [pixel]
distance_min_threshold = 400 # Threshold of the min distance between left and right line [pixel]
distance_deviation_threshold = 300 # Threshold of the standard deviation of distance between left and right line [pixel]

def sanity_check(left_line, right_line):
    # Confirm that the detected lane lines are real
    # Checking that they have similar curvature
    left_curverad = left_line.radius_of_curvature
    right_curverad = right_line.radius_of_curvature
    diff_ratio = left_curverad / right_curverad
    if (diff_ratio >= curv_ratio_threshold) | (diff_ratio <= 1./curv_ratio_threshold):
        return False

    # Checking that they are separated by approximately the right distance horizontally
    left_fitx = left_line.current_xval
    right_fitx = right_line.current_xval
    distance = right_fitx - left_fitx
    distance_max = max(distance)
    distance_min = min(distance)
    if (distance_max > distance_max_threshold) | (distance_min < distance_min_threshold):
        return False

    # Checking that they are roughly parallel
    distance_deviation = np.std(distance)
    if distance_deviation >= distance_deviation_threshold:
        return False

    return True
```

`find_lane()`

Each time a new measurement passes the sanity check, it will be appended to the queue of recent measurements and then an average over the n past measurements is taken to obtain the lane position. Otherwise the result will not be appended to the queue of recent measurements, meanwhile the earliest result will be popped out from this list.

Regarding to the mechanism above, the queue of the recent measurements may get null if lines are lost for several frames in a row. When the queue is null, it shall start to search from scratch using a histogram and sliding window to re-establish the measurements. When the queue still has recent measurements, it will retain the previous positions from the previous frame and step to the next frame to search again.

```python
def find_lane(binary_warped, left_line, right_line, margin=100):

    # Update line data
    if left_line.n_buffered > 0 & right_line.n_buffered > 0:
        # If line buffer exists then use previous best search
        left_fit_last = left_line.best_fit
        right_fit_last = right_line.best_fit
        out_img, left_fit, right_fit = prev_search(binary_warped, left_fit_last, right_fit_last)
    else:
        # If line buffer loses then use sliding window search
        out_img, left_fit, right_fit = sliding_window_search(binary_warped, margin=margin)

    if left_fit.any() & right_fit.any():
        left_line.update(out_img, left_fit)
        right_line.update(out_img, right_fit)

        # Sanity Check
        if sanity_check(left_line, right_line):
            # If sanity check passes, add current fit to buffer
            left_line.set_detected(out_img, inp=True)
            right_line.set_detected(out_img, inp=True)
        else:
            # If sanity check not passes, ignore the current fit
            left_line.set_detected(out_img, inp=False)
            right_line.set_detected(out_img, inp=False)
    else:
        left_line.set_detected(out_img, inp=False)
        right_line.set_detected(out_img, inp=False)

    left_best_fit = left_line.best_fit
    right_best_fit = right_line.best_fit

    return out_img, left_best_fit, right_best_fit
```

Some more tricks I used in the video pipline:
1. For better analyzing of the result, the debug images are also screened on the output video.
- Color warped images put on the right top.
- Binary warped images put on the right middle.
- Line finding and fit images put on the right bottom.

Here is an example of the image with debug.

![alt text][image18]

2. To eliminate the error when no points can be detected, null check of fit arrays is added in the line searching and line drawing functions.

```python
if left_fit.any():
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
if right_fit.any():
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
```


Here is the link to my video result:

 1. [project_video](./output_videos/project_video_result.mp4)
 2. [challenge_video](./output_videos/challenge_video_result.mp4)
 3. [harder_challenge_video](./output_videos/harder_challenge_video_result.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most challenging problems locate in two aspects, one is image processing to get a clear line binary image, another is mechanism for line smoothing or prediction. Adaptive parameters is also an important point, how to get a common algorithm for the pipeline.

- Light extract with image process

The biggest issue by far for me is sudden changes of light conditions. In the harder challenge video case, the lines get either completely lost (going from bright to dark) or image gets filled with noise coming from the white spots.
Although I have added some tricks, including yellow and white extract in intense lightness and dark lightness, opening calculation to extract shape of line, to make pipeline robust against that kind of changes, they still can cause major problems. More advanced filtering and brightness equalization techniques have to be examined. Or methods and technical which I don't know.

Glare on the road, appearing, for example, under trees, may lead to noisy results. Additional issues could happen due to poor condition of road marking, intersection of different lines. It could partly be resolved by additional line filtering between video frames.
However, the main problem is the absence of road lines or their invisibility. Lines on the road could by invisible due to dust or, as it happens more often, snow coverage or autumn leaves on the road. Partly snow coverage could confuse the algorithm as well because wind may form intricate snow patterns on the road. More sophisticated algorithms (such as deep neural networks) should be applied in the case of line absence in order to predict and determinate where the lane position should be.

- Mechanism for line smoothing or prediction

In this project I have used several tricks, such as line tracking, sanity check, search for the new line within +/- some margin around the old line center, average smoothing of recent high-confidential n past measurements. But it still doesn't work well on the harder challenge video when the line cannot be detected rationally.
In case of lines with noise or even invisibility, line smoothing and prediction shall be essential. When a detected line is rational on one side, a prediction for the line on the other side may be useful. But it will have a big influence for the sanity check and queue of recent measurements mechanism. If only one side lane is detected, how to execute the sanity check may also be a problem. Even when two side lanes are both invisible.
Using some memory algorithm could be helpful.

- Adaptive parameters

In the pipeline for the three video pipeline, there are some parameters shall be adapted, such as color range threshold for yellow and white, matrix of perspective transform, margin for sliding window searching.
It is better to change the pipeline algorithm to use self-adaptive for these parameters. Take perspective transform matrix for example, the source and destination points may change when the car is driving on a plain, uphill or downhill road. Getting the vehicle steering angle and speed information may help our algorithm to self-adapt this transform.
The color range thresholds are definitely different on an intense sunlight road or in a tunnel road. If there is a lightness sensor, it may help for the lane-line color extract.


At last, lane line detection is very important for self-driving. A high performance with stable, robust, redundant algorithm may need large amount of technical skills, experience based optimization and test feedback.