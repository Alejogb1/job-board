---
title: "Normalize ImageJ Image Stack: Quick & Easy Unity Adjustment"
date: '2024-11-08'
id: 'normalize-imagej-image-stack-quick-easy-unity-adjustment'
---

```java
// Assuming you have an ImagePlus object named "imp" containing your image stack

// Convert to 32-bit mode
imp.getProcessor().convertToFloat();

// Loop through each slice in the stack
for (int i = 1; i <= imp.getStackSize(); i++) {
  imp.setSlice(i);
  // Get the current slice as a Processor
  ImageProcessor ip = imp.getProcessor();

  // Normalize the slice to unity
  ip.normalize();
}
```
