# Superhero Mask Overlay
<img src="https://github.com/kranok-dev/Superhero_Mask_Overlay/blob/main/thumbnail.png" width="500">

**Description**                                                               
> Simple project made for fun. It consists of placing a mask (a PNG image with transparent background) on your face. It uses MediaPipe to detect face landmarks and applies homography to warp the mask image to your face's orientation.

**Installation**
> Clone this repository and the implemented code requires MediaPipe, OpenCV and Numpy to be installed in Python (Python 3 was used):
  ```
  $ pip3 install mediapipe
  $ pip3 install opencv-contrib-python
  $ pip3 install numpy
  ```

**Execution**
> The application was designed to process the webcam live feed and shows the result image. You can move your mouse to select another mask:
```
$ python3 app.py
```

> Try the demo implemented, test your own images, and have fun!

**Demo Video**
> https://www.youtube.com/watch?v=UYiplvH7-qc
