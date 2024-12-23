---
title: "opencv project points camera transformation?"
date: "2024-12-13"
id: "opencv-project-points-camera-transformation"
---

so camera transformations with OpenCV yeah I’ve been down that rabbit hole more times than I care to admit. It's like one of those things that sounds simple on paper but then reality hits you with all its coordinate systems and matrices and suddenly you're debugging for hours.

So you wanna get points from one camera view to another right? Project them from one perspective to another that’s the gist of it. I’ve seen it a bunch especially in multi-camera setups or robotics projects where you've gotta figure out where something is in the real world based on different views of it.

Let's cut through the fluff and talk actual implementations and solutions yeah? The core idea is that we’re dealing with different coordinate systems. A camera's image plane gives you pixel coordinates but to get anything meaningful in the real world we gotta bridge that gap. We need those translation and rotation matrices that link camera 1 and 2 as well as intrinsic camera parameters for every camera used in the system.

First off what we should be aware of is that we almost always work with homogeneous coordinates when dealing with transformations because they make all the matrix math easier. The whole concept is so easy to mess up so make sure you’re super aware of every coordinate space you are dealing with at each given point.

Let's say you’ve got points detected by camera one and you want to know their 3D location relative to camera two or simply the new 2D projection of them as camera two would see them. This involves several steps which almost always lead to some kind of headache. We typically go from 2D pixel points in the first camera through its intrinsic parameters to a 3D representation then apply the relative rotation and translation from camera 1 to camera 2 and then project back to the camera 2 pixel view with its intrinsic matrix.

I’ve had my fair share of pain trying to nail down these transforms especially when camera calibration isn't super solid.  One time I was working on a VR system and my head tracking was drifting all over the place turns out my intrinsic parameters were off just a tiny bit but that was enough to break everything. Debugging those subtle errors really builds character you know. Anyway let's dig into the actual code because that’s usually what people really care about here.

Here’s a snippet of Python code using OpenCV and Numpy to give you a feel of how it looks like. Assume we have all the needed rotation translation and intrinsic matrices.

```python
import cv2
import numpy as np

def transform_points(points2d_cam1, K1, K2, R, t):
    """
    Transforms 2D points from camera 1 to camera 2.

    Args:
        points2d_cam1 (np.array): 2D points in camera 1's pixel coordinates [N, 2]
        K1 (np.array): Intrinsic matrix of camera 1 [3, 3]
        K2 (np.array): Intrinsic matrix of camera 2 [3, 3]
        R (np.array): Rotation matrix from camera 1 to camera 2 [3, 3]
        t (np.array): Translation vector from camera 1 to camera 2 [3, 1]

    Returns:
        np.array: 2D points in camera 2's pixel coordinates [N, 2]
    """
    points2d_cam1 = np.array(points2d_cam1).reshape(-1, 1, 2).astype(np.float32)
    points_homo_cam1 = cv2.convertPointsToHomogeneous(points2d_cam1)[:, 0, :]
    points_3d_cam1 = np.linalg.solve(K1, points_homo_cam1.T).T

    points_3d_cam1_homo = np.hstack((points_3d_cam1, np.ones((points_3d_cam1.shape[0], 1))))

    Rt = np.hstack((R, t))
    points_3d_cam2_homo = np.dot(Rt, points_3d_cam1_homo.T).T

    points_3d_cam2 = points_3d_cam2_homo[:, :3]

    points_homo_cam2 = np.dot(K2, points_3d_cam2.T).T

    points2d_cam2_projected = points_homo_cam2[:, :2] / points_homo_cam2[:, 2:]
    
    return points2d_cam2_projected
if __name__ == '__main__':
    # Dummy data for demonstration
    K1 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    K2 = np.array([[900, 0, 330], [0, 900, 250], [0, 0, 1]], dtype=np.float32)
    R = np.array([[0.9, 0.1, 0], [-0.1, 0.9, 0], [0, 0, 1]], dtype=np.float32)
    t = np.array([[0.1], [0.1], [0]], dtype=np.float32)
    points2d_cam1 = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)

    points2d_cam2 = transform_points(points2d_cam1, K1, K2, R, t)
    print("Projected points in camera 2:\n", points2d_cam2)
```

This function `transform_points` takes 2D points from camera 1 along with intrinsic camera matrices for camera 1 and 2 `K1` and `K2` the rotation matrix `R` and the translation vector `t` which transforms points from camera one’s coordinates to camera two's coordinates and spits out the transformed points in camera 2 pixel coordinates. It uses the matrix algebra that’s the bread and butter of computer vision. First the 2d points from camera 1 are converted to 3d by using the inverse of the K1 matrix. Then we convert the 3d points in homogeneous coordinates and transform the 3d points from camera one space to camera two’s using the rotation and translation matrix we then remove the homogeneous coordinate and we project the 3d points to camera 2’s view with the K2 camera matrix.

Now there's more to it than that you could be dealing with distortion parameters and more advanced transformations but this gives you a simple baseline to start with. You might also want to use the `cv2.projectPoints` function but I found that sometimes it hides the nitty-gritty details when debugging and you don’t know what is actually going on in the hood.

One thing you always always need to think about is the coordinate frame of your rotation and translation matrix. The R and t matrix are transforming the points from frame 1 to frame 2. So, remember that your rotation matrix R and translation vector t need to express the transform of frame 1 **as seen by** frame 2. I know it seems redundant to say it so many times but people often mess this part up.

And here’s a C++ version for you guys that prefer it.

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<Point2f> transformPoints(const vector<Point2f>& points2d_cam1, const Mat& K1, const Mat& K2, const Mat& R, const Mat& t) {
    vector<Point2f> points2d_cam2;
    
    Mat points_homo_cam1;
    convertPointsToHomogeneous(points2d_cam1, points_homo_cam1);
    
    Mat points_3d_cam1;
    solve(K1, points_homo_cam1.t(), points_3d_cam1, DECOMP_LU);
    
    Mat points_3d_cam1_homo;
    vconcat(points_3d_cam1.t(), Mat::ones(1, points_3d_cam1.cols, CV_32F), points_3d_cam1_homo);

    Mat Rt;
    hconcat(R, t, Rt);
    
    Mat points_3d_cam2_homo = Rt * points_3d_cam1_homo;
    
    Mat points_3d_cam2 = points_3d_cam2_homo(Rect(0, 0, points_3d_cam2_homo.cols, 3));
    
    Mat points_homo_cam2 = K2 * points_3d_cam2;

    for(int i = 0; i < points_homo_cam2.cols; ++i) {
      Point2f p;
      p.x = points_homo_cam2.at<float>(0, i) / points_homo_cam2.at<float>(2, i);
      p.y = points_homo_cam2.at<float>(1, i) / points_homo_cam2.at<float>(2, i);
      points2d_cam2.push_back(p);
    }
    return points2d_cam2;
}


int main() {
    // Dummy data for demonstration
    Mat K1 = (Mat_<float>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
    Mat K2 = (Mat_<float>(3, 3) << 900, 0, 330, 0, 900, 250, 0, 0, 1);
    Mat R = (Mat_<float>(3, 3) << 0.9, 0.1, 0, -0.1, 0.9, 0, 0, 0, 1);
    Mat t = (Mat_<float>(3, 1) << 0.1, 0.1, 0);
    vector<Point2f> points2d_cam1 = {Point2f(100, 100), Point2f(200, 200), Point2f(300, 300)};

    vector<Point2f> points2d_cam2 = transformPoints(points2d_cam1, K1, K2, R, t);

    cout << "Projected points in camera 2:" << endl;
    for(const auto& p : points2d_cam2) {
        cout << p << endl;
    }

    return 0;
}
```

This C++ version is essentially the same logic as the Python one. The most annoying part for me when writing C++ is the Mat type and it's manipulation but the logic is pretty much the same.

Here’s an extra thing if you’re dealing with camera poses rather than individual transformations you might be looking at something like this snippet which has become so familiar to me it makes me cry a little.

```python
import numpy as np

def transform_points_pose(points3d_cam1, pose_cam1, pose_cam2, K2):
  """Transforms 3D points from the world/camera 1 coordinate frame to camera 2's pixel coordinates using poses
    Args:
      points3d_cam1 (np.array): 3D points in the world/camera 1's coordinate frame [N, 3]
      pose_cam1 (np.array): Pose (4x4 transform matrix) of camera 1 relative to the world/reference frame
      pose_cam2 (np.array): Pose (4x4 transform matrix) of camera 2 relative to the world/reference frame
      K2 (np.array): Intrinsic matrix of camera 2 [3, 3]

    Returns:
      np.array: 2D points in camera 2's pixel coordinates [N, 2]
    """
  points_3d_cam1_homo = np.hstack((points3d_cam1, np.ones((points3d_cam1.shape[0], 1))))

  # Calculate the transformation from world to camera 2
  world_to_cam2 = np.linalg.inv(pose_cam2)

  # Calculate the transformation from camera 1 to camera 2
  cam1_to_cam2 = np.dot(world_to_cam2, pose_cam1)

  # Apply the transformation
  points_3d_cam2_homo = np.dot(cam1_to_cam2, points_3d_cam1_homo.T).T
  points_3d_cam2 = points_3d_cam2_homo[:, :3]

  points_homo_cam2 = np.dot(K2, points_3d_cam2.T).T
  points2d_cam2_projected = points_homo_cam2[:, :2] / points_homo_cam2[:, 2:]

  return points2d_cam2_projected

if __name__ == '__main__':
  # Dummy data
  K2 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
  points3d_cam1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

  # Camera poses (4x4 transform matrices)
  pose_cam1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32) # Identity matrix (Camera 1 at origin)
  pose_cam2 = np.array([[0.9, 0.1, 0, 0.2], [-0.1, 0.9, 0, 0.2], [0, 0, 1, 0.2], [0, 0, 0, 1]], dtype=np.float32)

  points2d_cam2 = transform_points_pose(points3d_cam1, pose_cam1, pose_cam2, K2)
  print("Projected points in camera 2:\n", points2d_cam2)
```

This version is the one I usually end up implementing because it’s the most generic form of this problem. Camera poses are 4x4 transform matrices that represent both rotation and translation and the points are 3d in world coordinates. First we calculate the transform of each camera pose to world coordinates so we can compute the transform of camera 1 to camera 2. Then we just do the same projection as before.

The trickiest part here is often making sure your pose matrices are in the correct format. And as you can see this is all just matrix multiplication. Honestly, after a while these transform matrices feel like a second language. It's just moving things from one coordinate system to another, but it’s got to be right because a little slip-up and your points will fly off the screen.

For learning more I suggest starting with Hartley and Zisserman’s “Multiple View Geometry in Computer Vision” it is the Bible of this type of problem or “An Invitation to 3-D Vision: From Images to Geometric Models” by Yi Ma which is a bit easier to digest.

So in short be very aware of your coordinate systems and practice a lot of matrix operations. Also don’t overcomplicate things it might seem complex at first but it all boils down to basic linear algebra operations.
