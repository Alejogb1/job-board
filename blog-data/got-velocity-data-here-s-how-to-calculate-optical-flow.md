---
title: "Got Velocity Data?  Here's How to Calculate Optical Flow"
date: '2024-11-08'
id: 'got-velocity-data-here-s-how-to-calculate-optical-flow'
---

```python
import numpy as np
import cv2

# Load velocity data
data = np.fromfile('Velocity/ns_1000_v.dat', dtype=np.float32)
data = np.reshape(data, (128, 128, 128, 3))

# Select a slice
slice_num = 5
vx = data[:, :, slice_num, 0]
vy = data[:, :, slice_num, 1]

# Create optical flow visualization
flow = np.stack((vx, vy), axis=-1)
flow_magnitude = np.sqrt(vx**2 + vy**2)
flow_angle = np.arctan2(vy, vx)

# Normalize magnitude for visualization
flow_magnitude_normalized = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Convert to HSV for visualization
hsv = np.zeros((128, 128, 3), dtype=np.uint8)
hsv[..., 1] = 255  # Saturation
hsv[..., 0] = (flow_angle * 180 / np.pi) % 180  # Hue (angle)
hsv[..., 2] = flow_magnitude_normalized.astype(np.uint8)  # Value (magnitude)

# Convert to BGR for display
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Display the optical flow visualization
cv2.imshow('Optical Flow', bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
