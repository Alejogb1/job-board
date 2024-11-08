---
title: "Android Camera2: Preview Overexposed with YUV ImageReader - Quick Fix?"
date: '2024-11-08'
id: 'android-camera2-preview-overexposed-with-yuv-imagereader-quick-fix'
---

```java
// prepare list of surfaces to be used in capture requests
List<Surface> sfl = new ArrayList<Surface>();

sfl.add(mCameraSurface); // surface for viewfinder preview

sfl.add(mPreviewImageReader.getSurface()); //preview data
sfl.add(mImageReader.getSurface()); // surface for image capture

// configure camera with all the surfaces to be ever used
camDevice.createCaptureSession(sfl, new sessionListener(), null);
```
