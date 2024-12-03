---
title: "How are Gaussian splats used in rendering interactive 3D scenes?"
date: "2024-12-03"
id: "how-are-gaussian-splats-used-in-rendering-interactive-3d-scenes"
---

Hey so you want to know about Gaussian splats for 3D rendering right cool stuff  It's basically a way to render scenes super efficiently especially when you've got a ton of points or a crazy detailed mesh think millions of polygons we're talking  Instead of rendering each polygon individually which is slow as molasses in January you represent your geometry as a bunch of these things called splats  Think of them as little fuzzy blobs each representing a tiny piece of your 3D model


These splats aren't just any blobs though they're Gaussian blobs which means they have this nice smooth bell curve shape  That's where the "Gaussian" part comes in  This smooth shape is key because it lets you blend the splats together seamlessly creating a smooth surface even though you're only working with a bunch of individual blobs  It's like magic but it's math magic  


The way it works is you assign each splat a position and a bunch of other properties like size color normal vector even things like material properties if you want to get fancy  Then during rendering the GPU aka your graphics card works its magic blending these Gaussian splats together based on their properties and your viewpoint  It's all done in parallel so it's super fast  This parallel processing is a huge reason why this method shines


The cool thing is that you can control the level of detail  If you need a super detailed rendering you can use lots of small closely packed splats  If you only need a rough approximation you can use fewer larger splats  This adaptive level of detail is awesome for things like level of detail in games or interactive visualizations where you need performance without sacrificing too much visual fidelity  Think about flying over a city in a flight simulator you don't need every single brick texture rendered when you're miles away


One really cool application is rendering point clouds  You know those massive datasets of 3D points collected by lidar scanners or something  Rendering those directly as points is often messy and looks like a swarm of bees but with Gaussian splats you can turn that chaotic mess into a smooth surface  It's like a magical point cloud to mesh converter


So let's talk code  I'll give you some snippets to illustrate the concepts  Remember this is just a taste  Actual implementations can get really complex but these should give you a general idea


First let's look at a simple splat representation  This is in a C++-like pseudocode because I'm not gonna write a full renderer here


```c++
struct Splat {
  vec3 position; // position of the splat's center
  vec3 normal;  // normal vector at the splat's center
  float radius; // radius controlling the splat's size
  vec3 color;   // color of the splat
};
```

Pretty straightforward right  We have the position the direction it's facing its size and its color  You could add more properties like roughness or reflectivity based on your needs  That's the fundamental building block


Next we need a function to evaluate a Gaussian  This function takes a position in 3D space and returns a value representing the density of the Gaussian at that position


```c++
float gaussian(vec3 pos, float radius) {
  float distSq = dot(pos, pos); // distance squared from the splat's center
  return exp(-distSq / (2 * radius * radius));
}
```

Simple right This is just a basic implementation you'll find many variations but the key is that its a radial function that decreases with distance from the center. The radius parameter controls how spread out or concentrated the Gaussian is.


Finally here's a super simplified rendering loop again in pseudocode  In a real implementation this would be done on the GPU using shaders which are programs that run on your graphics card


```c++
for each pixel on screen
  for each splat
    vec3 splatToPixel = pixelPos - splat.position; // vector from splat to pixel
    float splatContribution = gaussian(splatToPixel, splat.radius) * splat.color; // splat's influence
    accumulatedColor += splatContribution; // add this splat's color
  end for
  finalPixelColor = accumulatedColor; // color of the pixel is the sum of all splats influence
end for
```


This loop iterates over all the pixels on the screen and for each pixel it iterates over all the splats  It calculates how much each splat contributes to the color of the pixel using the Gaussian function  Then it sums up the contributions from all the splats to get the final color of the pixel


Now keep in mind this is massively simplified  Real-world implementations use more sophisticated techniques like importance sampling hierarchical data structures optimized shading models all to improve performance and visual quality


To delve deeper I'd suggest looking up papers and books on  point-based rendering  and  GPU rendering techniques.  A good starting point might be searching for publications on  "Level of Detail for Point-Based Rendering" or "Efficient Gaussian Splatting on GPUs".  You can also find a lot of information in books on computer graphics  many cover advanced rendering techniques including  splatting  methods.  Look for resources covering  real-time rendering  as that's where Gaussian splats are usually employed.


This stuff is complex but it's also really cool  Experiment with different splat properties and you'll start to see how you can control the look and feel of your rendered scenes.  It's a powerful technique with a lot of potential  and  it’s a great way to learn more about the fascinating world of computer graphics and GPU programming.  Happy splatting
