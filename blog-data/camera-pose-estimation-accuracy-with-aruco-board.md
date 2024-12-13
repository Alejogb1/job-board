---
title: "camera pose estimation accuracy with aruco board?"
date: "2024-12-13"
id: "camera-pose-estimation-accuracy-with-aruco-board"
---

Okay so camera pose estimation accuracy using an ArUco board eh been there done that bought the t-shirt well more like built the entire system from scratch only to realize my t-shirt was inside out the whole time but we'll get to that

Let's dive right in you're talking about the classic problem of figuring out where a camera is in 3D space relative to a known ArUco marker board and yeah accuracy is paramount otherwise your augmented reality dream becomes a shaky distorted mess and nobody wants that believe me

See in my early days back when I was still rocking a CRT monitor and had a dial up connection faster than my grandma I was working on this project this robot arm project I had to implement accurate pose estimation for the arm to pick and place objects from a conveyer belt using computer vision The first version well I used a chessboard honestly it was awful the accuracy was like hitting a baseball with a pool noodle very bad then I tried a simple square marker and yeah it did not go well either it was just awful I thought i could make it work with enough camera calibrations but nope that wasnt it So I finally settled on an Aruco board it did the trick

Now the key here isn't just slapping an ArUco board in front of the camera and hoping for the best you really need to think about a bunch of things to nail this down

First thing first let's talk about ArUco itself You got your marker size marker dictionary and the board layout these are crucial parameters If your board size is too small compared to your camera resolution the pose estimation is going to be noisy as hell the markers will be tiny in the frame The same thing goes to the dictionary size if the dictionary is small the distance where you can detect the markers will be lower and it might also lead to mismatches

I’ve seen people thinking bigger is always better but hold your horses if the marker board is massive well your camera has to see all of it all the time or you are going to have a partial detection or no detection at all Also if the board is too big you also have to deal with potential distortions from your lens specially if your FOV is very wide this could be a real problem

Okay so for example let's say you are using OpenCV which is a usual suspect here I’ll be using python as my language because well python

```python
import cv2
import cv2.aruco as aruco
import numpy as np

# Define the aruco dictionary and parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters_create()

# Define the board parameters this was important for me
marker_size = 0.05 # in meters for example 5cm
marker_separation = 0.01 #in meters 1cm
board_size = (5,5) # number of markers in x and y
board = aruco.GridBoard_create(board_size[0], board_size[1], marker_size, marker_separation, aruco_dict)

# The camera matrix is super important as well
# These are the intrinsic parameters from your camera calibration
camera_matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float32)

# These are the distortion coefficients from your camera calibration
dist_coeffs = np.array([k1,k2,p1,p2,k3], dtype=np.float32)

# Example capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Estimate pose of the board
        rvec, tvec, _ = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs)

        # Draw the board outline
        if rvec is not None and tvec is not None:
          # draw axes in the corner of the marker to visualize orientation
          aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        
        # Draw detected markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)


    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

In this code snippet you'll see that I start by setting the correct dictionary as well as the parameters Then I set the board parameters like marker size marker spacing and finally board size as well as how to create the board Next I show you that you need your camera parameters to make the pose calculation work this are your camera intrinsic matrix and your distortion coefficients Finally I show how you detect the markers how to estimate the pose with `estimatePoseBoard` which is the function that performs all the heavy lifting and draw the detected markers and the axis in the corner of each marker for visualization
So camera calibration is non-negotiable you need accurate camera intrinsic parameters and distortion coefficients if you want an accurate pose estimation It's like trying to cook a perfect meal without measuring the ingredients you can’t do it and your meal will be garbage. There are methods to calibrate your camera like the one provided in OpenCV or with specialized calibration tools I used to do it with the checkerboard but there are more sophisticated methods that will improve the final pose

Okay then let's talk about image processing because this is important as well If your image is blurry or if the lightning is very bad the detection is going to be awful the pose estimation will be wrong so preprocessing can really help you deal with this situation and increase your pose accuracy so think about using things like Gaussian blur or histogram equalization they're simple to implement but they do help to deal with noise in the image

Another thing I learned the hard way is that you have to be careful with the distance of your board to the camera closer is better because the marker will be bigger but not too close because you might have perspective issues and distortions the edges of the board can appear curved it is very tricky to get that right

I will also say that if you have a lot of reflections or occlusions with objects the marker detection is going to be problematic you may have partial detections or no detections at all and this will make the pose estimation a disaster

Now in terms of accuracy you are going to be fighting with errors and there are mainly two types one is from the marker detection and second from the pose estimation errors the marker detection errors can be reduced by a good preprocessing step and good camera parameters the pose estimation error will be reduced with a good camera calibration you will never eliminate errors entirely but if you work on reducing both you will have a good pose

Here's another code snippet showing you how you can use reprojection errors as a metric to measure the error in your detection

```python
import cv2
import cv2.aruco as aruco
import numpy as np

# Same setup as before
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters_create()
marker_size = 0.05
marker_separation = 0.01
board_size = (5,5)
board = aruco.GridBoard_create(board_size[0], board_size[1], marker_size, marker_separation, aruco_dict)
camera_matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1,k2,p1,p2,k3], dtype=np.float32)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs)
        
        if rvec is not None and tvec is not None:
            # Project the board points
            obj_points = board.objPoints
            projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
            projected_points = np.squeeze(projected_points, axis=1) # remove extra dim
            
            # Extract detected corners points in image plane
            detected_corners = np.concatenate(corners).reshape(-1,2)

            # Calculate the reprojection error per marker
            total_error=0
            for i in range(len(detected_corners)):
              if i < len(projected_points):
                error = np.linalg.norm(detected_corners[i] - projected_points[i])
                total_error += error
            
            if len(detected_corners) > 0:
              avg_error=total_error / len(detected_corners)
            else:
              avg_error=0

            
            print(f"Reprojection error: {avg_error}")
           
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
Here you can see that I’m calculating the reprojection error which is done by projecting the 3D points of the ArUco board onto the image plane using the estimated pose (rvec tvec) and your camera calibration parameters the projected points will be compared with your detected corners of the ArUco board and the norm error is calculated
This is a good way to check your errors if they are very big your pose calculation is probably not very good or if they are very small your pose calculation is good.

And one more thing you can also use a different method to calculate the pose like PnP or Perspective n points which is a classical algorithm to estimate a camera pose given correspondences between 3D points and their 2D projections. Now I used the board estimation function instead because it is very handy

```python
import cv2
import cv2.aruco as aruco
import numpy as np

# Same setup as before
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters_create()
marker_size = 0.05
marker_separation = 0.01
board_size = (5,5)
board = aruco.GridBoard_create(board_size[0], board_size[1], marker_size, marker_separation, aruco_dict)
camera_matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1,k2,p1,p2,k3], dtype=np.float32)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Get the 3D object points corresponding to detected IDs
        obj_points = []
        for id in ids:
            idx=np.where(board.ids == id[0])[0][0] #gets the index from board ids
            obj_points.append(board.objPoints[idx])
        obj_points = np.array(obj_points,dtype=np.float32)
        
        corners = np.array(corners,dtype=np.float32)
        corners = np.squeeze(corners, axis=1)

        # Estimate the pose using PnP
        rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)[1:3] #get only rvec and tvec
        
        #Draw axis
        if rvec is not None and tvec is not None:
          aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

So here we are using `solvePnP` instead of `estimatePoseBoard` this is another way to solve the pose problem and is used widely as well you can try both and see which one works better for you In this code I grab the 3D points from the board and the detected corners and pass them to `solvePnP`

Finally for the resources I would avoid tutorials or videos I found them lacking focus on the theory instead look for academic papers or books like the classic "Multiple View Geometry in Computer Vision" by Richard Hartley and Andrew Zisserman it is a bible for this kind of problems it is very math intensive but it will explain you all you need to know about pose calculation and camera calibration Another resource you can use is the documentation in OpenCV of the ArUco module to see what are the parameters that you can use and tune to improve your pose results
Also make sure you read academic papers about aruco pose estimation there's a ton of research in that area so it might help you implement something even more robust and with a better error metric

Okay I think that's about it If you had more specific questions just shoot and I'll see what I can do I really enjoy this kind of stuff it is like my version of a good detective novel
