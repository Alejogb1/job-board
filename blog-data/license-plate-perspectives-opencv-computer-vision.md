---
title: "license plate perspectives opencv computer vision?"
date: "2024-12-13"
id: "license-plate-perspectives-opencv-computer-vision"
---

Okay so license plate perspectives OpenCV right I've been wrestling with that beast for what feels like ages honestly You wouldn't believe the head scratching I went through back in the day trying to make sense of those skewed rectangles

I mean everyone thinks image recognition is some magical click-and-boom thing but let me tell you the real world is a messy place Especially when you're dealing with license plates at weird angles It’s like the universe is actively trying to sabotage your computer vision algorithms

So you're looking at getting those license plates back to a straight front-on view right Like you want to undistort them from whatever perspective they got captured in? Classic problem Been there done that probably bought the T-shirt from a shady convention somewhere

First things first you gotta get your hands on OpenCV thats like the bread and butter of this whole process If you haven't already you need to install it I'd assume you've already got python setup since you are asking about computer vision with opencv I'm not going into that

Then comes the fun part you know finding those license plates within the image itself Before you even think about perspective correction you need to know where the plate is This usually involves a bunch of different techniques I went down the rabbit hole with haar cascades a long time ago but they are too finicky especially if you are trying to work with different countries with varying plates

I ended up settling on a combination of edge detection contour finding and filtering out the most likely rectangle shapes to work effectively for my needs It's not bulletproof but it does the job mostly

Here's some python code that show this that I wrote probably circa 2018 before all the deep learning hype really took off you'll notice the lack of fancy neural nets in here its all basic image manipulation

```python
import cv2
import numpy as np

def find_license_plate(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_plates = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 2 <= aspect_ratio <= 6 and w * h > 1000:
                potential_plates.append(approx)
    
    # Here we would have a function to choose the best from potential plates but for
    # this example we assume the first found is the correct one
    if len(potential_plates) > 0 :
        return potential_plates[0]
    else:
        return None

if __name__ == '__main__':
    plate_corners = find_license_plate('license_plate_image.jpg') #Replace with your own path

    if plate_corners is not None:
      print("License plate located")
    else:
      print("Could not find the license plate")
```

You can see that it’s a fairly standard process but the key is to experiment and refine parameters like the Canny edge thresholds and contour area filters You’ll probably find yourself tweaking them like a mad scientist until you get something usable

After you find the corners of the license plate the real fun begins perspective transformation You need to define the source and destination points Source points are obviously the four corners you found Destination points are the points of a rectangle with the correct dimensions for a license plate it is usually a rectangle

The `cv2.getPerspectiveTransform` function is your friend here It calculates the transformation matrix needed to warp the image then `cv2.warpPerspective` applies it and boom you got a straight plate Now obviously I'm making it sound easy but trust me I spent days in a dark room muttering to myself to figure out the right parameters

Here's that part I wrote around the same time I had to deal with different plate sizes back then this was not so automated

```python
def correct_perspective(image_path, plate_corners):
    img = cv2.imread(image_path)
    
    # Check if a plate was found if not we don't do anything
    if plate_corners is None:
        return None
    
    # Reshape the plate_corners into a numpy array format suitable for the functions to work with
    plate_corners_reshaped = plate_corners.reshape((4, 2)).astype(np.float32)

    #Calculate the width and height of the plate.
    width = max( np.linalg.norm( plate_corners_reshaped[0] - plate_corners_reshaped[1] ) ,
                np.linalg.norm( plate_corners_reshaped[2] - plate_corners_reshaped[3] ) )
    
    height = max( np.linalg.norm( plate_corners_reshaped[0] - plate_corners_reshaped[3] ) ,
                 np.linalg.norm( plate_corners_reshaped[1] - plate_corners_reshaped[2] ) )


    # Define the destination points as a rectangle
    dest_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    transform_matrix = cv2.getPerspectiveTransform(plate_corners_reshaped, dest_pts)
    warped_img = cv2.warpPerspective(img, transform_matrix, (int(width), int(height)))
    return warped_img

if __name__ == '__main__':
    plate_corners = find_license_plate('license_plate_image.jpg') # Replace this with the correct path
    warped_plate = correct_perspective('license_plate_image.jpg', plate_corners)

    if warped_plate is not None:
        cv2.imshow('Corrected License Plate', warped_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not correct the perspective no license plate found")

```

Remember the order of source points is super important They should correspond to the order of destination points Usually top left top right bottom right bottom left If you get this mixed up the image will look like it was put through a blender It happened to me countless of times when I was writing the code in its first iteration

And just like that you've got a license plate with the proper perspective But its not over now you might want to do a bit more pre processing for an OCR algorithm to read the text. I remember my first OCR attempt worked about as well as a caffeinated sloth so dont get discouraged

Another thing to keep in mind is that you might need to deal with various levels of distortion caused by the cameras lens You might have to apply some lens distortion correction before the perspective transform to get the best results for that you would need to calibrate the camera to get the matrix and distortion coefficients there is plenty of information out there about it so I am not going into that

Here is another example of some basic code I wrote for cropping it for the OCR model

```python
def crop_plate(warped_img):
  gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Assuming the largest contour is the plate
  if contours:
    plate_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(plate_contour)
    cropped_plate = warped_img[y:y+h, x:x+w]
    return cropped_plate
  else:
      return None
if __name__ == '__main__':
    plate_corners = find_license_plate('license_plate_image.jpg')
    warped_plate = correct_perspective('license_plate_image.jpg', plate_corners)

    if warped_plate is not None:
       cropped_plate = crop_plate(warped_plate)
       if cropped_plate is not None:
        cv2.imshow('Cropped License Plate', cropped_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
       else:
         print("Could not find a suitable cropping")
    else:
        print("Could not correct the perspective no license plate found")
```

Now I know it's tempting to jump straight into the fancy deep learning stuff but sometimes the most straightforward approach is the most reliable and also requires less computing power You really should start with the basics before adding complexity or at least that’s what I learned the hard way after weeks of debugging complex deep learning code to find that a simple thresholding and contour detection worked way better for me. I mean sometimes simpler is better it saves you time and headache

As for further reading you should check out the classic "Computer Vision Algorithms and Applications" by Richard Szeliski Its been a while since I read it but it's got some good stuff in it on image transforms and geometrical models Also “Digital Image Processing” by Rafael C. Gonzalez and Richard E. Woods is another great resource It goes deeper into fundamental image processing techniques and it helped me a lot when I was starting

Now I'll tell you a little joke one time I spent like two days debugging a problem only to realize the problem was that the image I was working with was just black Turns out it wasn't even a license plate photo it was just a picture of my cat taken with the lens cap on I mean I deserve it for not checking but it serves to illustrate you need to start with the most obvious stuff before going deep into the code

Good luck with your project and remember to test thoroughly and don’t be afraid to tweak things until it works for your use case it's never a one size fits all solution in this domain You have to adapt your approach and your code to your own data and environment its the only way to do it. Let me know if you run into any more problems I'm always up for a good challenge
