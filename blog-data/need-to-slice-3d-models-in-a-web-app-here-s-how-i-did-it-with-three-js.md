---
title: "Need to slice 3D models in a web app?  Here's how I did it with Three.js"
date: '2024-11-08'
id: 'need-to-slice-3d-models-in-a-web-app-here-s-how-i-did-it-with-three-js'
---

```javascript
// Slices
function drawIntersectionPoints() {
    var contours = new THREE.Group();
    for(i=0;i<10;i++){
        // ... (rest of your code)

        plane.position.y = i;
        plane.updateMatrixWorld(true); // Add this line
        scene.add(plane);

        // ... (rest of your code)
    };

    // ... (rest of your code)
}
```
