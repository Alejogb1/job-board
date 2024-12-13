---
title: "augmented reality lighting real world objects with virtual light?"
date: "2024-12-13"
id: "augmented-reality-lighting-real-world-objects-with-virtual-light"
---

Okay so you're asking about augmented reality lighting how to make virtual lights actually affect real world objects that’s a meaty one Ive spent way too much time wrestling with this stuff trust me Its not as simple as slapping a point light source in Unity or Unreal and calling it a day let me tell you My foray into this started back in the dark ages of mobile AR like 2014 when we were using clunky custom marker-based systems before ARKit or ARCore ever existed We were working on an early education app that involved a virtual dinosaur interacting with the users real desk and of course kids wanted to see the dino cast shadows and be lit by the room

The initial attempts oh boy They were laughable We had a single omnidirectional light source that tried to simulate general room lighting and it looked completely flat fake no depth at all like a sticker pasted on the scene The dino looked like it was floating in a vacuum We tried using ambient occlusion hoping it would help but again it looked off because it didnt actually use the information of the real world surrounding the dinosaur Its like applying a filter it looks nice but is useless in our scenario It just darkens the model without actually being aware of the real world This is where I had my first “aha” moment okay this is not going to be easy it wont work without real world understanding

So the challenge comes down to understanding the actual real world geometry and lighting which is no small feat How do you figure out the light conditions and the geometry of the room in real time using a phone camera This led me down a rabbit hole of research papers I can never unlearn it was brutal but worth it I really suggest you delve into papers on Simultaneous Localization and Mapping (SLAM) If you want to know why look up papers on structure from motion its a heavy read but a necessary one Another one to look into is research about reflectance models for example PBR or physically based rendering its not a AR specific subject but is very important for lighting rendering of any kind also check out research on environment mapping these will be very useful

After spending many weeks buried in literature and lots of experimentation we landed on a method that worked better We started by implementing a basic light estimation algorithm this is where we look at a section of the camera feed and try to guess the light color and intensity This involves analyzing the image pixel information which is essentially a collection of color values that are converted to numbers and processed by algorithms That was relatively simple It was a good start it was the first step toward actual real world lighting but was still far from what I wanted. Here is a basic example using pseudocode to show how it might work:

```
    function calculateLightIntensity(cameraFrame):
        // 1. Convert frame to grayscale
        grayFrame = convertToGrayscale(cameraFrame)

        // 2. Calculate average pixel intensity of the frame
        averageIntensity = calculateAveragePixelValue(grayFrame)

        // 3. Normalize to a range (0-1) representing light intensity
        lightIntensity = normalize(averageIntensity, minGrayValue, maxGrayValue)

        return lightIntensity


    function calculateLightColor(cameraFrame):
         // 1.  calculate average color values (Red Green Blue)
        averageColor = calculateAverageRGB(cameraFrame)
        // 2. Normalize to a range (0-1)
        normalizedColor = normalizeRGB(averageColor, minColorValue, maxColorValue)
        return normalizedColor


    //  example usage to combine
    function estimateEnvironmentLight(cameraFrame):
        intensity = calculateLightIntensity(cameraFrame)
        color = calculateLightColor(cameraFrame)
        return {color : color, intensity: intensity}

```

This gave us a global light that at least changed based on the scene but the shadows were still missing Also this method only captured an average of the scene so in practice the results are not accurate if there are multiple light sources in the real world

Then came the big challenge: proper shadows and actual occlusion To achieve that we needed to know more than just average light color we needed to know about the scene geometry which is a 3d representation of the real world In those days we had to resort to some clever trickery because mobile devices were not powerful enough for real-time SLAM It was mostly projecting pre-made shapes and trying to match them to the edges of objects on screen It was bad but it was all we had back then It worked sort of for simple things like tabletops but failed miserably with anything more complex

After the release of the ARkit framework we started exploring depth sensors and point clouds it was amazing to see how much better the framework handled the geometry of the real world It wasnt just projecting 2d images but it was creating a 3d representation of the space and the surfaces inside it This was huge it was a game changer The method involved capturing the point clouds of the scene and converting them to a surface mesh It was messy but gave us a working map of the real world The idea then was to cast ray traces from the virtual light source against the real world mesh This allows us to calculate shadows and occlusion more accurately based on the point cloud data. This is the core of the current real time solutions this type of method combined with other optimizations is how it is usually done today

Here is a simplified illustration in pseudocode of what we were doing at the time using a point cloud method and ray tracing to calculate shadows:

```
function castShadowRay(lightPosition, pointPosition, realWorldMesh):
   // 1. Calculate the direction vector from the point to the light source
  rayDirection = normalize(lightPosition - pointPosition)
   // 2. Ray trace to see if there's an object between the point and the light
  intersectionPoint = raytrace(pointPosition, rayDirection, realWorldMesh)

   // 3. If there's an intersection, then the point is in shadow
  if(intersectionPoint exists):
   return true // it is in shadow
  else:
   return false // it is not in shadow
end function

function calculateShadowedLighting(virtualObject, realWorldMesh, lightPosition):
    for each point in virtualObject.points:
        isInShadow = castShadowRay(lightPosition,point, realWorldMesh)
        if(isInShadow):
            // reduce intensity of the light for the point
            applyShadowEffect(point)
        end if
    end for
end function

```

Now this is also a simplified example of what happens behind the scenes there are complex optimizations that need to be done such as creating a BVH tree for ray tracing or using GPU for accelerating calculations but it shows the basics

Now of course the problem is not completely solved we still get cases where the shadow is a bit too jittery because of the uncertainty and small errors of the real world surface mesh And sometimes light estimations are not always correct This is due to how the sensor gathers the real world information the sensors are not perfect it always has some uncertainty which means that the geometry and the lighting will have some errors

Also there is the whole issue with materials which makes it more complex Its not just about casting shadows or how bright the light is it also about how materials react to light Some surfaces are shinier and will reflect more light others are matte and diffuse it. This makes it more complex because you also have to take into account the material properties to render light properly. Lets not forget about indirect illumination as well also known as global illumination or GI this is light that bounces from one surface to another and that adds another level of realism to the scene without it the scene looks flat and unreal

For the material part I suggest looking into the Cook-Torrance BRDF model its widely used in renderers it will give you a deeper understanding of how materials can affect light Also check out papers about global illumination algorithms like path tracing or photon mapping These things are complex but they are the secret sauce behind realistic lighting. There are several free books and online courses about this subject as well so its a good way to start

One funny thing I remember from this project is that when we had the first version working correctly with dynamic shadows it was so exciting that we spent an entire afternoon shining lights at various objects on the desk to see the dinosaur reacting in real-time We were like kids and you know what we have to keep the fun in programming Its a tough field that needs a bit of fun otherwise you burn out

So in short its not a single bullet solution this whole thing involves a combination of light estimation real world geometry understanding and proper rendering techniques Its a journey not a destination The key is to try to simulate the behaviour of real light as accurately as possible while optimizing it to run smoothly on mobile devices

Here is another example in pseudocode for material properties

```
function calculateSpecularHighlights(lightPosition, pointPosition, normalVector, viewDirection, materialProperties):

  // 1. Calculate the reflection vector
   reflectionVector = reflect(lightPosition,normalVector)

   // 2.  Calculate the specular highlight intensity based on material properties
  specularIntensity = materialProperties.specularIntensity *  pow(dot(reflectionVector, viewDirection), materialProperties.specularExponent)
   return specularIntensity

end function

function calculateDiffuseLight(lightPosition, pointPosition, normalVector, materialProperties):
    // 1. Calculate the direction vector from the point to the light
     lightDirection = normalize(lightPosition - pointPosition);

     // 2. Calculate the diffuse light intensity
     diffuseIntensity = materialProperties.diffuseIntensity * max(0, dot(normalVector, lightDirection));
     return diffuseIntensity;
end function
function calculateLightingForPoint(lightPosition, pointPosition, normalVector, viewDirection, materialProperties):

    // 1. Calculate diffuse contribution
      diffuseComponent = calculateDiffuseLight(lightPosition, pointPosition, normalVector, materialProperties);
    // 2. Calculate specular contribution
      specularComponent = calculateSpecularHighlights(lightPosition, pointPosition, normalVector, viewDirection, materialProperties);
    // 3.  Sum up all lights with ambient light
    finalColor = materialProperties.ambientColor + diffuseComponent + specularComponent;
    return finalColor
end function
```

This is an abstract simplification it shows how to combine the material properties with light color information to arrive at the final result but it shows how complex it can be To be real this response can fill another 1000 words if I went deeper on each one of the subjects Ive mentioned

Good luck with your AR adventures remember its about trial and error and continuous learning the field changes all the time
