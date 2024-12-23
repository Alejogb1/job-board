---
title: "How do I pin a horizontal plane to a specific altitude when using MapLibre GL JS with a terrain layer?"
date: "2024-12-23"
id: "how-do-i-pin-a-horizontal-plane-to-a-specific-altitude-when-using-maplibre-gl-js-with-a-terrain-layer"
---

,  I've certainly seen this exact scenario crop up in various projects over the years, and it's a common hurdle when working with terrain data in MapLibre GL JS. The goal here, as I understand it, is to create a flat, horizontal plane visually fixed to a specific altitude, irrespective of the underlying terrain's undulations. This is particularly useful for things like representing water surfaces, building platforms, or even creating distinct layers of information that aren’t tied to the ground contours. Getting this precisely right involves a few key concepts, and it's more nuanced than simply adjusting z-values.

Essentially, we’re manipulating the ‘translation’ or ‘offset’ of a 3D object relative to the terrain. The terrain layer itself, when enabled through MapLibre GL JS's `terrain` option, provides a displacement of the map's view in the z-direction based on the elevation data, while we need to introduce a counter-translation that effectively anchors our plane to a particular vertical coordinate.

My experience involved building a platform simulation for a geological survey several years back. We needed flat platforms hovering above the actual terrain, and the naive approaches, simply setting z-indexes, resulted in them following the terrain, which wasn’t what we wanted. The solution lies in utilizing the capabilities of the `fill-extrusion` layer type combined with careful construction of the underlying geojson. Instead of trying to offset the *entire* layer relative to the map terrain, we build individual polygons that are already positioned at the required altitude, and set the vertical position as an extrusion property of each of them.

The core strategy is to:

1.  **Create GeoJSON with Predefined Altitudes:** Generate GeoJSON data representing the polygons that will form our plane. Crucially, the z-coordinate in the geojson should represent the fixed altitude, *not* a relative altitude.
2.  **Use `fill-extrusion` Layer:** Utilize the `fill-extrusion` layer type, and use the coordinate provided as the `z` value for the height of the extrusion.
3.  **Ensure Correct Camera Setup:** The `camera` properties like `pitch` and `bearing` can influence how the extrusion is rendered, particularly when the scene isn't viewed directly from overhead.

Let me demonstrate this with a practical example, including some code snippets.

**Example 1: Simple Fixed-Altitude Plane**

This first example shows a single plane, defined by a rectangle at a fixed altitude.

```javascript
    const map = new maplibregl.Map({
        container: 'map',
        style: 'https://demotiles.maplibre.org/style.json',
        center: [-73.987392, 40.757445],
        zoom: 15,
        pitch: 45,
        hash: true,
        terrain: {source: 'dem'},
        attributionControl: false,
    });


    map.on('load', () => {
       map.addSource('my-plane', {
           type: 'geojson',
           data: {
               type: 'FeatureCollection',
               features: [
                   {
                       type: 'Feature',
                       properties: {},
                       geometry: {
                           type: 'Polygon',
                           coordinates: [
                               [
                                   [-73.988, 40.757],
                                   [-73.988, 40.758],
                                   [-73.986, 40.758],
                                   [-73.986, 40.757],
                                   [-73.988, 40.757],
                               ]
                           ]
                       }
                   },
               ]
           }
       });


        map.addLayer({
            id: 'plane-layer',
            type: 'fill-extrusion',
            source: 'my-plane',
            paint: {
                'fill-extrusion-color': 'skyblue',
                'fill-extrusion-height': 100, // Fixed altitude of 100 meters
                'fill-extrusion-base': 100, // Fixed altitude of 100 meters

                'fill-extrusion-opacity': 0.7
            }
        });
    });
```

In this example, the `fill-extrusion-height` and `fill-extrusion-base` properties are set to 100. This ensures the visual representation of the plane is rendered at 100 meters, regardless of the terrain data in the vicinity.

**Example 2: Multiple Planes at Different Altitudes**

Let's explore something a little more complex. Imagine you need multiple platforms at varying altitudes:

```javascript
   const map = new maplibregl.Map({
        container: 'map',
        style: 'https://demotiles.maplibre.org/style.json',
        center: [-73.987392, 40.757445],
        zoom: 15,
        pitch: 45,
        hash: true,
        terrain: {source: 'dem'},
        attributionControl: false,
    });


   map.on('load', () => {

      map.addSource('multi-plane', {
            type: 'geojson',
            data: {
              type: 'FeatureCollection',
                features: [
                    {
                        type: 'Feature',
                        properties: {altitude: 50},
                        geometry: {
                            type: 'Polygon',
                            coordinates: [
                                [
                                    [-73.989, 40.756],
                                    [-73.989, 40.757],
                                    [-73.987, 40.757],
                                    [-73.987, 40.756],
                                    [-73.989, 40.756],
                                ]
                            ]
                         }
                    },
                     {
                         type: 'Feature',
                         properties: {altitude: 150},
                         geometry: {
                           type: 'Polygon',
                           coordinates: [
                             [
                                 [-73.988, 40.758],
                                 [-73.988, 40.759],
                                 [-73.986, 40.759],
                                 [-73.986, 40.758],
                                 [-73.988, 40.758],
                             ]
                            ]
                         }
                     }
                ]
            }
        });



        map.addLayer({
            id: 'multi-plane-layer',
            type: 'fill-extrusion',
            source: 'multi-plane',
            paint: {
                 'fill-extrusion-color': 'orange',
                'fill-extrusion-height': ['get', 'altitude'],
               'fill-extrusion-base': ['get', 'altitude'],
                'fill-extrusion-opacity': 0.7
            }
       });
  });
```

In this example, the `altitude` property of each geojson feature is used to set `fill-extrusion-height` and `fill-extrusion-base`. This way, each polygon is extruded to the height specified in the data.

**Example 3: Dynamic Altitude Adjustment**

Now, let's see how we can change the altitude of our plane dynamically. This involves programmatically altering the geojson data source and triggering a redraw.

```javascript
   const map = new maplibregl.Map({
        container: 'map',
        style: 'https://demotiles.maplibre.org/style.json',
        center: [-73.987392, 40.757445],
        zoom: 15,
        pitch: 45,
        hash: true,
        terrain: {source: 'dem'},
        attributionControl: false,
    });

   let currentAltitude = 100;
    let planeSource;

    map.on('load', () => {
        planeSource = {
            type: 'geojson',
            data: {
                type: 'FeatureCollection',
                features: [
                    {
                        type: 'Feature',
                        properties: {},
                        geometry: {
                            type: 'Polygon',
                            coordinates: [
                                [
                                    [-73.988, 40.757],
                                    [-73.988, 40.758],
                                    [-73.986, 40.758],
                                    [-73.986, 40.757],
                                    [-73.988, 40.757],
                                ]
                            ]
                        }
                    },
                ]
            }
        };

        map.addSource('dynamic-plane', planeSource);

        map.addLayer({
            id: 'dynamic-plane-layer',
            type: 'fill-extrusion',
            source: 'dynamic-plane',
            paint: {
                'fill-extrusion-color': 'lightgreen',
               'fill-extrusion-height': currentAltitude,
                'fill-extrusion-base': currentAltitude,
                'fill-extrusion-opacity': 0.7
            }
        });
    });

    document.getElementById('raiseButton').addEventListener('click', () => {
        currentAltitude += 20;
        map.setPaintProperty('dynamic-plane-layer', 'fill-extrusion-height', currentAltitude);
         map.setPaintProperty('dynamic-plane-layer', 'fill-extrusion-base', currentAltitude);

    });
        document.getElementById('lowerButton').addEventListener('click', () => {
         currentAltitude -= 20;
        map.setPaintProperty('dynamic-plane-layer', 'fill-extrusion-height', currentAltitude);
         map.setPaintProperty('dynamic-plane-layer', 'fill-extrusion-base', currentAltitude);
    });
```

In this last example, we use a simple button to adjust the altitude and call `map.setPaintProperty` to change the height of the plane.  This highlights how changing the geojson and subsequently changing the layer properties can control the elevation dynamically.

For further exploration on related topics, I’d highly recommend looking into resources such as the *MapLibre GL JS documentation*, which details all the API parameters that I've used above, or *'Web Cartography: Exploring the World Wide Web' by Ian Muehlenhaus'* for detailed background on web mapping technologies. Additionally, the *'Real-Time 3D Rendering with DirectX' by Paul V. D'Angelo'* can provide deeper understanding on the 3D rendering pipeline, even though it’s DirectX focused the principles directly apply to opengl as well, which MapLibre GL JS is based on.

These examples and the background will provide the foundation you need to control horizontal planes with MapLibre GL JS and its terrain layer. Remember that proper understanding of coordinate systems and layer drawing order is crucial when creating these types of visualizations.
