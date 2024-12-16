---
title: "How to pin a plane to specific altitude using MapLibre GL JS?"
date: "2024-12-16"
id: "how-to-pin-a-plane-to-specific-altitude-using-maplibre-gl-js"
---

Okay, let's tackle this. I recall working on a flight simulation project a few years back where we needed precisely this functionality – keeping a plane icon locked to a specific altitude within a MapLibre GL JS environment. It’s a bit more involved than simply setting the lat/long, as you're dealing with the 3D space implications.

The challenge stems from how MapLibre GL JS handles positioning of features, particularly when you introduce elevation data. By default, a feature's coordinates are treated as 2D unless explicitly told otherwise. To accurately represent the plane's altitude, we need to manipulate the `translate` and, potentially, the `pitch` of the layer. We will need to use the `transformRequest` callback effectively to intercept the loading of terrain tiles to figure out the terrain altitude at a specific point.

The core issue is this: MapLibre GL JS doesn't automatically extrapolate elevation from the terrain data for every feature. So, while the map might show mountains and valleys, your plane icon will remain fixed at the base elevation unless you actively adjust its vertical position. The primary concept is to get the elevation of the terrain at the plane's coordinates, compare that with the desired altitude, and then apply a translation to the plane icon to get it at the correct vertical position.

Here's how I’ve approached this in the past and how you can do it:

First, we need to ensure we're working with a suitable terrain source. MapLibre GL JS supports several formats for terrain, and you will need to make sure yours is set up properly. Assuming we have a working terrain layer, the process involves these main steps:

1.  **Fetching Terrain Height:** Implement the `transformRequest` callback on the map to intercept the loading of vector tiles and extract the terrain information at the plane's location.
2.  **Calculating Vertical Offset:** Determine the desired altitude of the plane relative to mean sea level and find the current altitude based on the terrain and calculate the vertical offset.
3.  **Applying the Translation:** Adjust the `translate` property of the plane's layer to position it correctly, using the calculated offset. If the viewing angle (pitch) has been altered by user interaction, this will require some vector math.

Let's walk through some code examples. Note that these examples assume you have a plane icon that is already loaded as an image into the sprite sheet.

**Example 1: Basic Altitude Adjustment without considering camera tilt**

This first snippet will keep the plane icon at a desired altitude, assuming the camera is looking at the map from straight above. The plane icon will move up or down as the terrain elevation changes, always at the same offset altitude from the terrain:

```javascript
  const map = new maplibregl.Map({
    container: 'map',
    style: 'your-map-style.json', // Replace with your style
    center: [longitude, latitude], // Replace with plane's initial coordinates
    zoom: 10,
  });

    const desiredAltitude = 1000; // Example altitude, in meters
    let currentTerrainAltitude = 0;
  
  map.on('load', () => {
    map.addSource('plane-source', {
        type: 'geojson',
        data: {
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [longitude, latitude], // Replace
            },
        },
    });

    map.addLayer({
        id: 'plane-layer',
        type: 'symbol',
        source: 'plane-source',
        layout: {
            'icon-image': 'your-plane-icon', // Replace with your icon name
            'icon-size': 0.1,
            'icon-allow-overlap': true,
        },
    });

     map.on('render', () => {
         const terrainData = map.queryTerrainElevation(
                [longitude, latitude]
            );
           if (terrainData) {
              currentTerrainAltitude = terrainData.altitude
               let verticalOffset = (desiredAltitude - currentTerrainAltitude);
                 map.setPaintProperty('plane-layer', 'icon-translate', [0, verticalOffset]);

            }
    });
  });
```

In this code, we are using `queryTerrainElevation` to find the elevation of the ground and use that to vertically offset the plane. This won't account for the view pitch, but it’s good start.

**Example 2: Adjusting the plane position with a tilted camera (complex)**

This snippet demonstrates a more sophisticated way to handle camera pitch. It involves calculating the translation vector based on the view angle:

```javascript
const map = new maplibregl.Map({
      container: 'map',
      style: 'your-map-style.json',
      center: [longitude, latitude],
      zoom: 10,
    });

    const desiredAltitude = 1000; // Example altitude, in meters
     let currentTerrainAltitude = 0;
     let cameraPitch = 0
     let cameraBearing = 0
     
     map.on('load', () => {
        map.addSource('plane-source', {
            type: 'geojson',
            data: {
              type: 'Feature',
              geometry: {
                  type: 'Point',
                  coordinates: [longitude, latitude],
                  },
              },
          });
         map.addLayer({
            id: 'plane-layer',
            type: 'symbol',
            source: 'plane-source',
            layout: {
              'icon-image': 'your-plane-icon',
              'icon-size': 0.1,
              'icon-allow-overlap': true,
             },
         });

        map.on('render', () => {
            const terrainData = map.queryTerrainElevation(
                [longitude, latitude]
             );
           if (terrainData) {
              currentTerrainAltitude = terrainData.altitude
                let verticalOffset = (desiredAltitude - currentTerrainAltitude);

                cameraPitch = map.getPitch()
                cameraBearing = map.getBearing()

                const pitchRadians = (cameraPitch * Math.PI) / 180
                const bearingRadians = (cameraBearing * Math.PI) / 180
             
              const xTranslate = verticalOffset * Math.sin(pitchRadians) * Math.sin(bearingRadians)
              const yTranslate = verticalOffset * Math.cos(pitchRadians)

            map.setPaintProperty('plane-layer', 'icon-translate', [xTranslate, yTranslate]);
           }
        });
     });
```

Here, we get both pitch and bearing and calculate horizontal and vertical offsets, to make the plane stay above ground even with a rotated camera. This is a more robust solution for cases where your view isn't always straight down. This approach, while more complex, is crucial for a realistic 3D presentation.

**Example 3: Updating altitude when the user moves the plane**

This code will demonstrate how to dynamically change the desired altitude and update the plane's position on the map when you change it with a slider:

```javascript
const map = new maplibregl.Map({
        container: 'map',
        style: 'your-map-style.json',
        center: [longitude, latitude],
        zoom: 10,
    });

    let desiredAltitude = 1000; // Initial altitude
    let currentTerrainAltitude = 0;
    let cameraPitch = 0;
    let cameraBearing = 0;
    let planeSource = null;

    map.on('load', () => {
        planeSource = map.addSource('plane-source', {
            type: 'geojson',
            data: {
                type: 'Feature',
                geometry: {
                    type: 'Point',
                    coordinates: [longitude, latitude],
                },
            },
        });
        
        map.addLayer({
            id: 'plane-layer',
            type: 'symbol',
            source: 'plane-source',
            layout: {
                'icon-image': 'your-plane-icon',
                'icon-size': 0.1,
                'icon-allow-overlap': true,
            },
        });
       map.on('render', () => {
            const terrainData = map.queryTerrainElevation(
                [longitude, latitude]
            );
           if (terrainData) {
               currentTerrainAltitude = terrainData.altitude;
               let verticalOffset = (desiredAltitude - currentTerrainAltitude);

                cameraPitch = map.getPitch();
                cameraBearing = map.getBearing();
                const pitchRadians = (cameraPitch * Math.PI) / 180;
                const bearingRadians = (cameraBearing * Math.PI) / 180;
            
                const xTranslate = verticalOffset * Math.sin(pitchRadians) * Math.sin(bearingRadians)
               const yTranslate = verticalOffset * Math.cos(pitchRadians)

               map.setPaintProperty('plane-layer', 'icon-translate', [xTranslate, yTranslate]);
            }
         });

    const altitudeSlider = document.getElementById('altitude-slider');

      altitudeSlider.addEventListener('input', (event) => {
        desiredAltitude = parseFloat(event.target.value);

        // Optionally update the position of the plane with planeSource.setData() if you plan on moving it in 2D
        });
    });

```

This snippet introduces the use of an HTML slider to dynamically control altitude. Ensure you have an HTML input element with `id="altitude-slider"` in your page to make this code work. This example shows you how to keep the plane at the desired altitude even when the altitude changes dynamically.

**Technical resources**

For deep dives into the underpinnings of MapLibre GL JS and its rendering pipeline, consult the official MapLibre GL JS documentation. Specifically, explore the documentation on terrain and layer rendering.

Further, "Computer Graphics: Principles and Practice" by Foley et al. provides a comprehensive look into the mathematical foundations of 3D transformations, which is essential for fully mastering the techniques here. Additionally, you may find material on camera transformations and projection particularly useful. In addition, “Fundamentals of Computer Graphics, Fourth Edition” by Steve Marschner and Peter Shirley offers another look into graphics concepts and the theory behind rendering graphics.

In my experience, debugging map issues can sometimes feel like chasing a phantom. The best strategy is always to break the problem into smaller, manageable pieces. Use the debugging tools in your browser liberally, and don’t hesitate to add extra logging to pinpoint exactly what’s going on at each stage of the calculation. Start simple, and incrementally add complexity as you progress. This will give you a solid foundation that you can extend and fine-tune according to the specifics of your use case. Good luck.
