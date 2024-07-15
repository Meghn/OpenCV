# OpenCV

## What are images?

- Images are numpy arrays.
 ```python
 import cv2

 image = cv2.imread('some_image.png')
 print(type(image))
 ```
 ```<class 'numpy.ndarry'>```

- An image shape is given by its height, width and number of channels.
 ```python
 print(image.shape)
 ```
 ```(720,1280,3)```

- An image is made by **pixels**
    - In "most cases" pixel value range from **0** to **255**.
    - In binary images, pixel value is in **[0,1]** ( or **[0,255]**).
    - In 16 bits images pixel value range from **0** to **65535**. (Generally 8 bits)

## Input/Output

- __Image__
```python
# read image
img = cv2.imread(image_path)
# write image
cv2.imwrite(image_out_path, img)
# Visualization
cv2.imshow('Bird', img)
cv2.waitKey(0)
```
The ```waitKey``` keeps the image open indefinitely untill a key is pressed.

- **Video**
```python
# read video
video = cv2.VideoCapture(video_path)
# visualize video
ret = True
while ret:
    ret, frame = video.read()
    if ret:
        cv2.imshow('Elephant', frame)
        cv2.waitKey(20)
```
Writing a video is slightly more complicated.
The ```ret``` boolean variable is ``` True``` if there is a frame that can be read else it is ```False```
The ```waitKey``` is given a number so each frame is open only for that amout of miliseconds. for 25 frames per second that number would be 40 miliseconds.

To release the memory space allocated to the video, always have the below code.
```python
video.release()
cv2.destroyAllWindows()
```

- **Webcam**
```python
# read webcam
video = cv2.VideoCapture(1)
# visualize webcam
while True:
    ret, frame = video.read()
    if ret:
        cv2.imshow('WebCam', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
```

The number in the ```VideoCapture``` is the webcam number you want to use. 

## Basic Operations

- **Resizing**
```python
resized_img = cv2.resize(img, (640, 640))
```

- **Cropping**
```python
cropped_img = img[120:240, 120:260]
```

## Colorspaces

All images loaded by OpenCV are in the **BGR** Format.

```python
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```
The ```HSV``` colorspace is very popular among other colorspaces offered by openCV and have a very important application (egs. color detection) in the computer vision field.

## Blurring

Helpful to remove noise in an image.
- __blur__ : Each pixel is the mean of its kernel neighbours
- __gaussian blur__ : Convolve weach pixel with a gaussian kernel
- __median blur__ : Central element is replaced by the median of the kernel neighbours. This operation processes the edges while removing noise.

```python
k_size = 7
img_blur = cv2.blur(img, (k_size, k_size))
img_gaussian_blur = cv2.GaussianBlur(img, (k_size, k_size), 5)
img_median_blur = cv2.medianBlur(img, k_size)
```

## Threshold

![Thresholds](./data/Threshold.png)

We use thresholding for semantic segmentation.

- **Simple Thresholding** : 

    For every pixel the same threshold value is applied.

    We must first convert the colr image into grayscale.

    ```python
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    ```
    where ```80``` is the threshold we are going to use where ant pixel value above 80 will become 255(or white) and below 80 will become 0(or black) and ```255``` is the maximum value of a pixel.

    The output may not always be perfect so we just blur the resulting binary image and again send it through a threshold to get a better result.
    ```python
    thresh = cv2.blur(thresh, (10, 10))
    ret, thresh = cv2.threshold(thresh, 80, 255, cv2.THRESH_BINARY)
    ```

- **Adaptive Thresholding** : 
    Sometimes there would be shadows and highlights in an image where we cannot use just a single threshold. Hence we use the adaptive threshold.

    ```python
    adaptive_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)
    ```
    With ```ADAPTIVE_THRESH_GAUSSIAN_C``` the adaptive method and ```THRESH_BINARY``` the threshold type.

## Edge Detection

Many types of edge detection, namely:
- Sobel Operator
- Laplacian Operator
- Canny Edge Operator
    ```python
    img_edge = cv2.Canny(img, 100, 200)
    ```
    where ```100, 200``` are the min and max threshold we send to the canny edge detector.

After edge detection you can erode or dilate the image.
- **Erode** : The pixel is turned black if there are black pixels in its neighborhood
- **Dilate** : The pixel is turned white if there are white pixels in its neighborhood

## Drawing

We'll draw using the help of OpenCV. Four most popular drawings are:
- line
    ```python
    cv2.line(img, (100, 150), (300, 450), (0, 255, 0), 3)
    ```
    - Starting point: ```(100, 150)```
    - Ending point: ```(300, 450)```
    - Color: ```(0, 255, 0)```
    - Thickness: ```3```

- rectangle
    ```python
    cv2.rectangle(img, (200, 350), (450, 600), (0, 0, 255), -1)
    ```
    - Upper left corner: ```(200, 350)```
    - Lower right corner: ```(450, 600)```
    - Color: ```((0, 0, 255)```
    - Thickness: ```-1``` this fills up the rectangle with solid color. Any positive non-zero value will give only the boundary of the rectangle.

- circle
    ```python
    cv2.circle(img, (800, 200), 75, (255, 0, 0), 10)
    ```
    - Centre point: ```(800, 200)``` x-value is associated with the width and the y-value with the height.
    - Radius: ```75```
    - Color: ```(255, 0, 0)```
    - Thickness: ```10```

- text
    ```python
    cv2.putText(img, 'Hey you!', (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 10)
    ```
    - Text: ```'Hey you!'```
    - Location: ```(600, 450)```
    - Font: ```cv2.FONT_HERSHEY_SIMPLEX```
    - Text Size: ```2```
    - Color: ```(255, 255, 0)```
    - Thickness: ```10```
