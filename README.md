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





 