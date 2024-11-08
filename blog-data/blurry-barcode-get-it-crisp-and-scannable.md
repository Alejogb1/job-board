---
title: "Blurry Barcode?  Get it Crisp and Scannable!"
date: '2024-11-08'
id: 'blurry-barcode-get-it-crisp-and-scannable'
---

```python
import cv2
import numpy as np

# Load the image
image = cv2.imread('barcode_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Enhance contrast using adaptive thresholding
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (likely the barcode)
largest_contour = max(contours, key=cv2.contourArea)

# Decode the barcode
decoded_data = pyzbar.decode(image, symbols=[pyzbar.pyzbar.ZBarSymbol.EAN128])

# Print the decoded data
for barcode_data in decoded_data:
  print(barcode_data.data.decode('utf-8'))
```
