NOTES
===================

[Computer vision is an interdisciplinary scientific field that deals with how computers can be made to gain high-level understanding from digital images or videos. ](https://en.wikipedia.org/wiki/Computer_vision)

## Breakdown
1. Image processing and transformation
2. Video analysis
3. Cool stuff 😀

## Process
* Image capture - from physical world
* Raw data - Greyscale or RGB
* Preprocessing - low level feature extraction
* Processing - Detection, Segmentation, Object recognition, decision

## Application
1. Computer graphics - structural input of data to create 2D image
2. Image processing - extract data from image
3. Image Analysis - generate statistics from image
4. Computer vision 
5. Machine vision - Robotics

- - - -   
### Opencv
Core {numpy}, imgproc and highgui(rendering)

## Image processing
### Application
<details>
    <summary>Enhancement</summary>
    <p>changing saturation, contrast and look</p>
</details>
<details>
    <summary>Understanding</summary>
    <p>object detection and classification</p>
</details>
<details>
    <summary>Perturbation</summary>
    <p>blurring, rotating</p>
</details>

- - - -


When doing this processes the pixels are stored as a uint8 dtype and maximum is 256. So when using opencv it handles the overflow but if you are manually operating you will have to handle it.

* Interpolation - resizing an image requires reassigning pixels in the original shape

    Methods - nearest, bilinear, bicubic, quadric, gaussian, mitchell
* Colour spaces - grayscale
  
    Colour   
    1. RGB/BGR
    2. HSV/HSI
    3. CMYK/CMY

* Thresholding 
    1. Binary
    2. Binary inverse
    3. Truncate
    4. Truncate inverse
    5. To zero

* Geometric transformation
    1. [x] Resizing - increasing and decreasing resolution
    2. [x] Translation
    3. [x] Rigid - rotation
    4. [x] Scale 
    5. [ ] Perspective

## Image derivative
Amount if image pixel changing at a given point

### Application
Fast calculation of haar wavelets in face recognition
Convolution function h produced by a function g operating on another function f

## Histogram equilization
Histogram Equalization is a computer image processing technique used to improve contrast in images. It accomplishes this by effectively spreading out the most frequent intensity values, i.e. stretching out the intensity range of the image.

## Auto contrast

## Histogram matching

## Scaling
Resizing images

Interpolation
1. INTER_AREA - shrinking
2. INTER_CUBIC - slow
3. INTER_LINEAR - zooming

## Rotation

## Translation

## 2D Convolution (Filtering)
Just as 1D signals images also can be fitted with various LPF and HPF
* LPF - removes noise bluring images
* HPF - finding edges
Using kernels for averaging the pixel value

## Power law transformation
Enhancing images

Used in monitor displays

> S = C * (r ^ y) 

```
S - output pixel value
r - input pixel value
C and Y - real numbers (constants)
```


## READ 
* cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected
* cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
* cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
* cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. If 0 is passed, it waits indefinitely for a key stroke.
* cv2.ADAPTIVE_THRESH_MEAN_C: The threshold value is the mean of the neighbourhood area minus the constant C.
* cv2.ADAPTIVE_THRESH_GAUSSIAN_C: The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C.


## References
1. https://theailearner.com/2019/01/26/power-law-gamma-transformations/
https://benchpartner.com/power-law-transformations-gamma-correction-in-image-processing/
2. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_inpainting/py_inpainting.html
3. https://www.ijsr.net/archive/v7i1/ART20179794.pdf
4. https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
5. https://en.wikipedia.org/wiki/Computer_vision
6. https://www.youtube.com/playlist?list=PLh6SAYydrIpctChfPFBlopqw-TGjwWf_8

