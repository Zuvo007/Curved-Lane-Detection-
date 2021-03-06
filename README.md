# Curved-Lane-Detection-
Introduction : Autonomous Driving Car is one of the most disruptive innovations in AI. Fuelled by Deep Learning algorithms, they are continuously driving our society forward and creating new opportunities in the mobility sector. An autonomous car can go anywhere a traditional car can go and does everything that an experienced human driver does. But it’s very essential to train it properly. One of the many steps involved during the training of an autonomous driving car is lane detection, which is the preliminary step. Today, we are going to learn how to perform lane detection using videos.

There are some very important key factors which I have to take care off while doing this project.so these key steps are:

1. **Read and decode video files into proper frames.**
2. **2Gray scale conversion of continuous images that it's getting.**
3. **Reducing noises by applying various filters.**
4. **Detect edges and mask the canny image.**
5. **Find the co-ordinates of the lane.**
6. **Fit the co-ordinates into the canny image.**

So after doing this important parts finally the project waas done and doing these tasks were quite challenging and so let's share a few lines what are these actually steps are :

- **Capturing and decoding video file**: We will capture the video using VideoCapture object and after the capturing has been initialized every video frame is decoded (i.e. converting into a sequence of images).
- **Grayscale conversion of image**: The video frames are in RGB format, RGB is converted to grayscale because processing a single channel image is faster than processing a three-channel colored image.
- **Reduce noise:** Noise can create false edges, therefore before going further, it’s imperative to perform image smoothening. Gaussian filter is used to perform this process.
- **Canny Edge Detector**: It computes gradient in all directions of our blurred image and traces the edges with large changes in intesity. For more explanation please go through this article: Canny Edge Detector
- **Region of Interest**: This step is to take into account only the region covered by the road lane. A mask is created here, which is of the same dimension as our road image. Furthermore, bitwise AND operation is performed between each pixel of our canny image and this mask. It ultimately masks the canny image and shows the region of interest traced by the polygonal contour of the mask.
- **Hough Line Transform**: The Hough Line Transform is a transform used to detect straight lines. The Probabilistic Hough Line Transform is used here, which gives output as the extremes of the detected lines.

I am sharing my outputs so that you can have a better intuition how actually this gonna looks like.

<img src="https://github.com/Zuvo007/Curved-Lane-Detection-/blob/master/Outputs/output1.png">
<br>
<br>

<img src="https://github.com/Zuvo007/Curved-Lane-Detection-/blob/master/Outputs/output2.png">
<br>
<br>
<img src="https://github.com/Zuvo007/Curved-Lane-Detection-/blob/master/Outputs/output3.png">






