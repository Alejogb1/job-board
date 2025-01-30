---
title: "How to fix a Faceswap screen error?"
date: "2025-01-30"
id: "how-to-fix-a-faceswap-screen-error"
---
The persistent "screen error" during Faceswap processing typically signals a misalignment between the extracted face data and the target image's projection parameters, stemming from inconsistent or insufficient facial landmarks. This often manifests as a distorted or incomplete face overlay, particularly when dealing with significant head rotations or differing aspect ratios. My experience optimizing multiple Faceswap workflows has shown that a multi-pronged approach addressing input quality, alignment consistency, and model training is crucial.

A primary source of this error arises from poor quality input images. Faceswap models rely on precise facial landmarks – points defining the eyes, nose, mouth, and jawline – to accurately map one face onto another. If the source images are blurry, low-resolution, or occluded (partially hidden by hair, hands, or other objects), the landmark detection process becomes unreliable. This introduces errors in the affine transformation used to warp the source face, leading to the ‘screen error’. For example, if the detected eye landmarks are inaccurate, the resulting swap will appear as if the eyes are either floating away or misaligned.

Secondly, inconsistencies in landmark detection *across* the source and target datasets are problematic. Even if individual source and target images are clear, variations in pose, lighting, or expression can result in different landmark placements. These differences compound when the model attempts to synthesize the target face using the source's identified features, resulting in a visible ‘screen error’. This is particularly pronounced when swapping faces between drastically different angles, say from a frontal face to a profile view, or between vastly differing head sizes. Standard approaches involve meticulously cleaning the dataset by removing problem cases or preprocessing by adjusting the pose and lighting before processing through Faceswap. However, inconsistencies can also stem from variations in the landmark detector's behaviour.

The model itself can contribute to such errors. An insufficiently trained model, or one trained on a limited dataset, may not generalize well to new, unseen faces or pose variations. This will lead to inadequate facial representation capabilities. If the model has not learned to robustly handle variations in face shape, it struggles to project the source face onto the target. Furthermore, a model trained exclusively on frontal faces, for example, will likely have difficulty handling profile faces, thus causing the ‘screen error’.

Here are a few common mitigation strategies with code examples. These are to be implemented prior to the main Faceswap training or usage.

**Example 1: Data Cleaning and Preprocessing**

This example demonstrates how to perform basic image quality checks and preprocessing, essential for obtaining high-quality input for Faceswap. This utilizes OpenCV library, which needs to be installed before execution. This step also assumes the input is organised in a source and target folder.

```python
import cv2
import os
import numpy as np

def preprocess_images(folder_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(folder_path, filename)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {filename}. Skipping.")
                continue

            # Check image dimensions and remove small images
            h, w, _ = image.shape
            if h < 100 or w < 100:
                print(f"Error: Image {filename} too small. Skipping.")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Sharpen image slightly
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
           

            # Resize to consistent 256 x 256
            resized = cv2.resize(sharpened, (256, 256), interpolation=cv2.INTER_AREA)

            output_file_path = os.path.join(output_path, filename)
            cv2.imwrite(output_file_path, resized)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

source_images_folder = 'source_images'
target_images_folder = 'target_images'

preprocessed_source_folder = 'preprocessed_source'
preprocessed_target_folder = 'preprocessed_target'

preprocess_images(source_images_folder, preprocessed_source_folder)
preprocess_images(target_images_folder, preprocessed_target_folder)
print("Preprocessing Complete.")
```

This script iterates through each image in the provided folders, reads the image, checks its dimensions, converts it to grayscale, sharpens the image using a kernel filter and then resizes it to a standard 256x256 resolution. It writes the resulting images into a preprocessed output folder. This helps address inconsistent image sizes and qualities that can cause landmark errors. The resizing to a constant size also assists with model convergence in later stages.

**Example 2: Consistent Landmark Detection**

Landmark detection inconsistencies can be reduced by standardizing the input pose, specifically using face alignment techniques. This example relies on dlib. This library should also be installed prior to execution. This uses the preprocessed input folders from the previous example.

```python
import dlib
import cv2
import os
import numpy as np

def align_face(image_path, output_path, detector, predictor):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if not faces:
        print(f"No face detected in {os.path.basename(image_path)}.")
        return

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)], dtype=np.float32)

        # Define target landmark coordinates based on a standard frontal face
        target_points = np.array([
                [106.17, 116.90], [106.17, 135.23], [109.17, 153.62], [116.77, 170.33], [125.55, 186.06],
                [137.85, 197.11], [151.18, 203.04], [167.40, 202.90], [179.85, 197.11], [192.47, 186.06],
                [201.25, 170.33], [208.85, 153.62], [211.85, 135.23], [211.85, 116.90], [158.01, 86.22],
                [156.56, 105.75], [155.64, 128.89], [155.64, 148.44], [156.56, 171.58], [158.01, 191.12],
                [116.80, 103.54], [124.05, 93.09], [133.61, 90.57], [143.16, 91.72], [149.79, 99.05],
                [151.24, 104.63], [117.00, 148.03], [124.25, 141.74], [133.77, 141.09], [143.53, 142.14],
                [149.95, 150.33], [151.50, 157.89], [167.91, 104.63], [174.54, 99.05], [182.71, 91.72],
                [192.27, 90.57], [201.98, 93.09], [209.22, 103.54], [166.71, 148.03], [173.95, 150.33],
                [182.37, 142.14], [192.13, 141.09], [201.65, 141.74], [208.90, 148.03], [151.59, 175.46],
                [151.59, 184.77], [151.59, 193.86], [156.14, 177.64], [161.50, 174.79], [167.06, 175.24],
                [172.01, 177.64], [176.20, 177.49], [181.88, 177.64], [187.50, 175.24], [192.76, 174.79],
                [197.65, 177.64], [148.70, 187.50], [163.29, 189.55], [195.27, 189.55], [206.13, 187.50],
                [161.45, 194.79], [185.16, 194.79]
                ], dtype=np.float32)

        # Calculate the affine transformation
        transformation_matrix = cv2.getAffineTransform(points[17:48], target_points[17:48])
        
        # Apply the transformation
        aligned_face = cv2.warpAffine(image, transformation_matrix, (image.shape[1], image.shape[0]))
        cv2.imwrite(output_path, aligned_face)
    

# Path to the trained landmark detection model from dlib
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" # Download this file and put it in the same directory
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

source_images_folder = 'preprocessed_source'
target_images_folder = 'preprocessed_target'

aligned_source_folder = 'aligned_source'
aligned_target_folder = 'aligned_target'

os.makedirs(aligned_source_folder, exist_ok=True)
os.makedirs(aligned_target_folder, exist_ok=True)

for filename in os.listdir(source_images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_images_folder, filename)
        output_path = os.path.join(aligned_source_folder, filename)
        align_face(image_path, output_path, detector, predictor)
for filename in os.listdir(target_images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(target_images_folder, filename)
        output_path = os.path.join(aligned_target_folder, filename)
        align_face(image_path, output_path, detector, predictor)
print("Face alignment completed.")
```

This script uses dlib's face detector and landmark predictor to align the faces in both the source and target folders. It first detects the face, then identifies 68 facial landmarks, then computes an affine transform from the detected landmarks to a target set of landmarks that correspond to a frontal pose.  The aligned faces are then written to new output folders. By ensuring the landmarks are relatively consistent across the dataset, the warp transformations during swap processing will be more accurate. The target landmarks here are a manually tuned set for a specific face orientation. It is essential to replace this target landmark set if alignment is being made to a different orientation or a different landmark count is being used.

**Example 3: Data Augmentation**

Augmenting the training data with varied pose and lighting conditions can improve the model's ability to handle variations in the target images. The following demonstrates a basic augmentation strategy using OpenCV.

```python
import cv2
import os
import numpy as np
import random

def augment_images(folder_path, output_path, num_augmentations_per_image=2):
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read image {filename}. Skipping.")
            continue

        for i in range(num_augmentations_per_image):
            augmented_image = image.copy()

            # Random rotation (-25 to 25 degrees)
            angle = random.uniform(-25, 25)
            rows, cols, _ = augmented_image.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (cols, rows))

            # Random brightness adjustment
            alpha = random.uniform(0.8, 1.2)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=0)

            # Random horizontal flip (50% chance)
            if random.random() < 0.5:
                augmented_image = cv2.flip(augmented_image, 1)

            output_file_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_aug_{i}.png")
            cv2.imwrite(output_file_path, augmented_image)

source_aligned_folder = "aligned_source"
target_aligned_folder = "aligned_target"
augmented_source_folder = "augmented_source"
augmented_target_folder = "augmented_target"

augment_images(source_aligned_folder, augmented_source_folder)
augment_images(target_aligned_folder, augmented_target_folder)

print("Image augmentation completed")
```
This script takes images from a folder, performs random rotations, adjusts brightness, and applies a random horizontal flip for a specified number of augmentations per image, then saves the output into a new folder. These variations improve the model's ability to handle diverse input images.

By rigorously preprocessing the data with quality checks and alignment, and augmenting it before model training, the occurrence of "screen errors" during face swapping can be considerably reduced. Resource recommendations for further investigation include the official documentation of libraries such as dlib, OpenCV, and general resources on image preprocessing and data augmentation within computer vision. Additional exploration into advanced model training techniques can further improve results.
