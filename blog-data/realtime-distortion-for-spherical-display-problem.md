---
title: "realtime distortion for spherical display problem?"
date: "2024-12-13"
id: "realtime-distortion-for-spherical-display-problem"
---

 so you're talking about realtime distortion for spherical displays right I've been down that rabbit hole more times than I'd like to admit Back in the day when I was working on that immersive art project thing a giant half-dome display was our biggest nightmare we had this incredibly detailed 3D model and we wanted to project it onto the dome in real time without everything looking like a melted Dali painting and man it was not pretty initially We’re not talking about a simple flat screen projection here we're working with a curved surface so what works for a regular monitor is gonna look seriously messed up

The core issue here is that a flat image which is what our graphics cards naturally render needs to be warped or distorted in such a way that when it's projected onto a sphere it appears undistorted from the viewers perspective Think about projecting a rectangular photo onto a basketball it just doesn't work you need to pre-distort that image so it looks correct on the curve it's not simple its some serious matrix math manipulation

So initially when we tackled this we tried the basic fisheye lens type distortion it was straightforward yeah its simple enough to implement but the problem was the extreme warping towards the edges of the dome we got some serious stretching and loss of detail and the more we tried to compensate for the edges the more the center ended up looking squished it was a mess We even tried some basic texture mapping techniques but the mapping was like a flat plane projected onto the spherical surface we got pinching and that's not what we're after that was a total no go too much effort for garbage results

 lets get into the nuts and bolts of a couple of things we actually tried I mean the real deal here first up we played around with pre-rendered lookup tables this approach uses an offline process to compute the distortion for a whole grid of pixels Then during the projection each pixel on the display grabs its pre-calculated pixel from the table That was an idea to speed things up later on I'll put the code on the bottom first example was some precalc stuff but it was not real time enough so we needed a real solution later on

```python
import numpy as np

def generate_lookup_table(width, height, sphere_radius):
  """Generates a lookup table for spherical distortion.

  Args:
      width: The width of the output image.
      height: The height of the output image.
      sphere_radius: The radius of the sphere.

  Returns:
      A numpy array representing the lookup table.
  """
  lookup_table = np.zeros((height, width, 2), dtype=np.float32)
  for y in range(height):
    for x in range(width):
      nx = (x / width) * 2 - 1  # Normalize x to [-1, 1]
      ny = (y / height) * 2 - 1  # Normalize y to [-1, 1]
      r_squared = nx * nx + ny * ny
      if r_squared > 1:
            lookup_table[y,x] = np.array([0.5,0.5])
            continue  # Outside circle
      r = np.sqrt(r_squared)
      phi = np.arcsin(r)

      theta = np.arctan2(ny, nx)
      
      spherical_x = np.cos(theta) * sphere_radius * phi / r if r!=0 else 0
      spherical_y = np.sin(theta) * sphere_radius * phi / r if r!=0 else 0
    
      
      
      lookup_table[y,x] = np.array([(spherical_x / sphere_radius + 1)/2,(spherical_y/sphere_radius + 1)/2 ])
  return lookup_table
# Example usage
width = 512
height = 512
radius = 1  # Sphere radius
lookup_table = generate_lookup_table(width, height, radius)

print(lookup_table)
```
This python snippit just generates the lookup table for a specific radius and resolution you can use it in the shaders later on to get the texture position

We then moved onto a more robust approach using shaders directly this means we're doing the distortion calculations on the graphics card which is what you want for realtime results The idea is this each pixel on your projected flat image corresponds to a point on the sphere so in the vertex shader you transform each point in the source texture into spherical coordinates and then from those coordinates you calculate the position on the display surface the key element is to handle the normalization so it looks right when projected the first try on this was ok but we were far away from perfection

Let's talk shaders specifically for that initial vertex transformation You'd be using something like this the glsl shader code is more or less the same for most of the languages:

```glsl
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
uniform float sphereRadius;
void main()
{
    vec2 normCoords = aPos;
     float r = length(normCoords);
        if(r > 1.0)
            {
                gl_Position = vec4(0.0,0.0,0.0,1.0);
                TexCoord = vec2(0.5,0.5);
                 return;
             }
        float phi = asin(r);
        float theta = atan(normCoords.y, normCoords.x);
        float x = cos(theta) * sphereRadius * phi / r;
        float y = sin(theta) * sphereRadius * phi / r;

       gl_Position = vec4(x , y, 0.0, 1.0);
       TexCoord = aTexCoord;

}
```

Now that shader needs to run on a surface that you want to render your scene which in this case is like a flat quad mesh where the texture coordinates are used to map the texture into a 3D model the shader above is only the vertex shader here is the fragment shader example:
```glsl
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D ourTexture;
void main()
{
    FragColor = texture(ourTexture, TexCoord);
}
```
Now these two simple shaders do not include things like uv mapping or normal mapping or any other advance lighting techniques but they are the core for basic realtime distortion in spherical displays The vertex shader transform the vertexes using the spherical projection function to properly display the texture and the fragment shader will just copy the pixel color from texture to the display

Now a big problem we had early on was getting the parameters right the field of view of your virtual camera and the radius of your display have to be calibrated perfectly otherwise you end up with stretched or pinched images getting these values correct is crucial if you have a physical dome display you can measure the radius you can also use different values for x and y radii if you want to have a different curved display

Debugging this stuff in real time was a whole level of fun that is sarcasm by the way we're essentially distorting a distortion so when things didn't quite line up it was a challenge to trace the errors back to the source of them I mean we spent a solid week chasing down a calculation error in the shader that we though was an issue with our physical display setup turns out it was me and my bad coffee decision that day.

For resources I'd suggest looking into the classic graphics bible "Real-Time Rendering" by Tomas Akenine-Möller Eric Haines and Naty Hoffman it's got a very very solid math background for this and also take a look at "Advanced Graphics Programming Using OpenGL" by Tom McReynolds and David Blythe if you want to use the OpenGL side of the problem and have actual implementations This is some very serious math I mean that's why it works you need a strong geometric foundation to understand the transformations you are doing

There are also tons of research papers on projection systems like dome projections but honestly the ones I saw mostly focused on more advanced corrections like blending across projectors in a multi-projector setup or dealing with imperfect displays this simple spherical mapping is not too studied in papers

The key thing to remember here is that this isn't just about throwing some code together It’s about understanding the coordinate systems the math behind the spherical transformations and how they translate to a real-world display surface It can be a headache to setup but once you get it working it is a magical thing to see a perfectly distorted 3D world wrapping around your field of vision its pretty cool stuff actually and it's worth learning it if you're into this stuff

So I hope that kinda clears things up a bit if you have any questions about details about the code or implementation specific I'll try my best to answer it based on my very very long experience with this
