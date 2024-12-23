---
title: "How to pin a plane to specific altitude in MapLibre GL JS?"
date: "2024-12-23"
id: "how-to-pin-a-plane-to-specific-altitude-in-maplibre-gl-js"
---

Alright, let's unpack how we'd tackle pinning a plane to a specific altitude using MapLibre GL JS. This is a situation I've encountered quite a few times in past projects, particularly when visualizing flight paths or creating detailed 3d simulations. It's not as straightforward as just setting a Z value, as we need to consider how MapLibre interprets its coordinate systems, specifically when dealing with 3d models like our plane.

Essentially, you're not working with a direct height value in the same way you might in a typical 3d graphics API like WebGL directly. Instead, MapLibre builds upon the concept of 'terrain'. If you’re adding a 3d model to the map, and wish to position that model at a specific altitude, you need to work with the map's terrain elevation. The core process revolves around translating your desired altitude into a relative offset to the elevation of the underlying map geometry at the plane’s location.

I remember this one project, a flight tracker, where the client wanted the planes to appear as if they were actually flying at a specific flight level, completely detached from the earth surface but with the earth surface shown too. Initially, our plane models kept oscillating with the terrain, looking like they were having a bumpy ride rather than flying smoothly. The solution involved a bit of calculation and utilizing MapLibre's ability to fetch terrain elevation data.

Let's dive into a step-by-step approach. We will focus on ensuring the model stays consistently at the desired altitude regardless of the terrain changes. The approach generally involves the following:

1.  **Retrieving Terrain Elevation:** We use `map.queryTerrainElevation` to obtain the elevation of the terrain at the plane's current geographical coordinates. This gives us a baseline.

2.  **Calculating the Offset:** We subtract the terrain elevation from our target altitude. The result is the vertical offset that we must apply to the model’s position in the scene.

3.  **Updating the Model’s Position:** We will then adjust the model matrix, or in some cases the `translate` property of the layer, to position it vertically correctly based on that offset.

Here's the first code snippet which demonstrates fetching terrain elevation and printing to console (for demonstration purposes, you'd ultimately be incorporating this into a more comprehensive update logic):

```javascript
map.on('load', () => {
  const planeCoordinates = [-100, 40]; // Example coordinates for our plane
  const targetAltitude = 1000;       // Desired altitude in meters

  map.queryTerrainElevation(planeCoordinates, {}, (err, elevation) => {
    if (err) {
      console.error("Error fetching elevation:", err);
      return;
    }
    console.log(`Terrain Elevation: ${elevation} meters`);
    const altitudeOffset = targetAltitude - elevation;
    console.log(`Calculated altitude offset: ${altitudeOffset} meters`);

     // Here is where you would eventually adjust your model's position using the altitudeOffset
     //...
  });
});
```

This first block is purely for demonstration – note that we are *logging*, not actually *moving* anything. We're showing you how to correctly get the elevation and derive the offset. Now, the meat of the solution lies in continuously applying this to the model as the map changes.

Let’s assume you have loaded the 3d model for our plane already. Then our next code snippet adds the function to the `render` event which updates the plane's position, this is where the actual offset calculation and movement occurs:

```javascript
let planeModelLayerId; // Store the layer ID of your plane model
let targetAltitude = 1000; // Example desired altitude

// Function to update the plane's altitude:
function updatePlaneAltitude(coordinates) {
  map.queryTerrainElevation(coordinates, {}, (err, elevation) => {
    if (err) {
      console.error("Error fetching elevation:", err);
      return;
    }

    const altitudeOffset = targetAltitude - elevation;

    map.setLayerPaintProperty(planeModelLayerId, 'translate', [0, 0, altitudeOffset]);
    // OR if working with matrix:
     // If your model uses a matrix and you are adjusting that on render rather than
    // a translate property, you would update that matrix here.
    //...

  });
}

map.on('render', function() {
  if (map.loaded() && planeModelLayerId) {
       const planeCoordinates = [-100, 40]; // Get current plane coords from your source
       updatePlaneAltitude(planeCoordinates);
  }
});
map.on('load', function(){
 //...code to load your plane model using map.addLayer and store the id in planeModelLayerId
  planeModelLayerId = 'my-plane-layer-id' // example placeholder

});
```

This second snippet demonstrates how to make the plane respond to the render loop, calculating the correct offset and applying it on every frame render, however, it still does not consider the need to move the object. We are focusing solely on the vertical offset here.  It also illustrates the need to use `map.setLayerPaintProperty` if working with the `translate` property, or to directly manipulate the 3d model’s transformation matrix in the render loop if your 3d model is not using a `translate`.

The final snippet introduces movement along a path and demonstrates how the offset is applied with changing coordinates of the object:

```javascript
let planeModelLayerId;
let targetAltitude = 1000;
const pathCoordinates = [ // Example path
  [-100, 40], [-99, 41], [-98, 42], [-97, 43],[-96, 44]
];
let currentStep = 0;

function updatePlaneAltitude(coordinates) {
    map.queryTerrainElevation(coordinates, {}, (err, elevation) => {
        if (err) {
            console.error("Error fetching elevation:", err);
            return;
        }

        const altitudeOffset = targetAltitude - elevation;
         map.setLayerPaintProperty(planeModelLayerId, 'translate', [0, 0, altitudeOffset]);
        });
}

map.on('render', function() {
 if (map.loaded() && planeModelLayerId) {
      if (currentStep < pathCoordinates.length) {
          updatePlaneAltitude(pathCoordinates[currentStep]);
          currentStep++;

      } else {
         currentStep = 0;
      }

   }
});

map.on('load', () => {
// ...load 3d model here setting the layer id
   planeModelLayerId = 'my-plane-layer-id'

});

```

In this last snippet, we are introducing a small path for the plane to follow, illustrating that the altitude pinning logic is general regardless of the location of the plane.

A few important points:

*   **Performance:** Be mindful that querying terrain elevation every frame can be costly, especially with a high number of objects. In our flight-tracking project, we found that implementing a less frequent update cadence (e.g., every few frames or based on significant location changes) resulted in a smoother experience without compromising accuracy. Consider leveraging requestAnimationFrame properly for controlling the pacing of updates.
*   **Coordinate System:** Always be aware that `queryTerrainElevation` expects coordinates in the same system that MapLibre is using.
*   **Interpolation:** For smooth movements, you would need to combine this with proper interpolation methods, perhaps by using `requestAnimationFrame` together with the coordinate update.

For further reading on this subject, consider delving into the MapLibre GL JS documentation specifically around terrain, layers, and transformations; they offer detailed explanations. Additionally, explore *Computer Graphics: Principles and Practice* by Foley, van Dam, Feiner, and Hughes for a deeper understanding of 3D transformations and coordinate systems, which forms the foundational knowledge behind these methods. Furthermore, papers detailing the performance optimization of terrain rendering in virtual globes would be worthwhile. These resources should offer a more robust theoretical foundation to these concepts. Remember to always refer to the official documentation as a primary source.

In conclusion, pinning a plane to a specific altitude in MapLibre GL JS involves understanding the terrain elevation, calculating the necessary offset, and applying that offset to the model's transformation. This combined with careful performance tuning leads to a robust solution that maintains the desired altitude regardless of changes in the underlying terrain.
