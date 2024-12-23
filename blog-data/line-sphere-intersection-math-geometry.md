---
title: "line sphere intersection math geometry?"
date: "2024-12-13"
id: "line-sphere-intersection-math-geometry"
---

 so you're asking about line sphere intersection classic computer graphics problem I've wrestled with this one plenty back in the day when I was really getting into 3D rendering with my old Amiga 500 yeah I'm dating myself a bit there

The core problem is to figure out if a line and a sphere touch or intersect and if they do where those intersection points actually are  We need this for all sorts of things like collision detection ray tracing anything involving objects interacting in 3D space

Let's break down the math the line can be represented parametrically like this `L(t) = O + tD` where O is the origin of the line D is the direction of the line t is a scalar that controls how far along the line we go and the sphere is easier it's just given as a center C and radius r

First we calculate the vector from the lines origin to the sphere's center say `OC = O - C` then we calculate `a = D dot D` which is just the length of the direction vector squared and we do that because D is often normalized so its square is 1 then we calculate `b = 2 * (OC dot D)` which has this 2 because it's a term from the quadratic equation we will be working with and `c = (OC dot OC) - (r * r)` which is essentially the distance of line origin to sphere center minus sphere radius squared

things are getting interesting now The actual intersection part relies on solving a quadratic equation `at^2 + bt + c = 0` We do that with the famous quadratic formula
`t = (-b +- sqrt(b^2 - 4ac)) / (2a)`

The discriminant `delta = b^2 - 4ac` tells us everything about the intersections if `delta < 0` it means there are no real solutions no intersection if `delta = 0` then there's one solution a single tangent intersection where the line touches the sphere in one point if `delta > 0` then we have two solutions two intersections where the line goes through the sphere

 so here's some code in a pseudo c-ish style should be easy to grasp

```cpp
struct Vec3 {
  float x;
  float y;
  float z;
};

struct Sphere {
  Vec3 center;
  float radius;
};

struct Line {
  Vec3 origin;
  Vec3 direction;
};


struct Intersection {
    bool hit;
    Vec3 point;
    float t; // Distance along line
};


Intersection intersect_line_sphere(Line line, Sphere sphere) {
    Intersection result;
    result.hit = false; // Assume no hit initially
    Vec3 oc;
    oc.x = line.origin.x - sphere.center.x;
    oc.y = line.origin.y - sphere.center.y;
    oc.z = line.origin.z - sphere.center.z;

    float a = dot(line.direction, line.direction);
    float b = 2.0f * dot(oc, line.direction);
    float c = dot(oc, oc) - (sphere.radius * sphere.radius);

    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return result; // No intersection
    }


    float t1 = (-b - sqrt(discriminant)) / (2.0f * a);
    float t2 = (-b + sqrt(discriminant)) / (2.0f * a);

    if(t1 > t2) {
        float temp = t1;
        t1 = t2;
        t2 = temp;
    }



   if(t1 >= 0) { // check if it hits in the direction of the line
        result.hit = true;
        result.t = t1;
        result.point = line.origin;
        result.point.x = line.origin.x + line.direction.x * t1;
        result.point.y = line.origin.y + line.direction.y * t1;
        result.point.z = line.origin.z + line.direction.z * t1;

        return result;

    }

    if(t2 >=0) {
        result.hit = true;
        result.t = t2;
        result.point = line.origin;
        result.point.x = line.origin.x + line.direction.x * t2;
        result.point.y = line.origin.y + line.direction.y * t2;
        result.point.z = line.origin.z + line.direction.z * t2;

        return result;

    }


    return result;

}

float dot(Vec3 v1, Vec3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}


```

I’ve had a lot of headaches with numerical precision especially when `delta` is very close to zero you can get some weird behavior so you should keep an eye on that It took me a while to realize I should make a tolerance for this it's really subtle

Now for some optimization tips because I know you’re thinking about that for a large scene where you have to do this intersection test with a lot of spheres that can get quite expensive really quickly

First is to implement bounding spheres or boxes If we use bounding spheres check first if the line intersects the sphere that encompasses a collection of small spheres and that can reduce the amount of time spent on this test but the math can be simplified to avoid calculating the `sqrt()` when comparing distances for example to reduce the number of expensive operations you can try using an approximation for that too not super accurate but will speed things up a little bit or use lookup tables to avoid doing the square root calculation every single time for each line sphere check if its necessary

Second consider using spatial partitioning data structures like k-d trees or octrees if you're dealing with multiple spheres instead of iterating through each sphere in the scene you only need to test against a relatively small subset of the spheres which speeds up the whole process quite a lot

Here's a quick example of just comparing distances and avoiding the square root

```cpp
bool intersects_bounding_sphere_fast(Line line, Sphere sphere) {
    Vec3 oc;
    oc.x = line.origin.x - sphere.center.x;
    oc.y = line.origin.y - sphere.center.y;
    oc.z = line.origin.z - sphere.center.z;


    float a = dot(line.direction, line.direction);
    float b = 2.0f * dot(oc, line.direction);
    float c = dot(oc, oc) - (sphere.radius * sphere.radius);

    float discriminant = b * b - 4.0f * a * c;
    return discriminant >= 0; // fast check without sqrt
}

```

I remember one time I had a rendering bug where the raytracer was just rendering black screens only to realize that I wasn't handling the case where the origin of the ray was inside the sphere correctly it gave zero values for the time along the ray parameter so the t value should be greater than or equals to zero I've learned it the hard way you should avoid doing that believe me

And for reference I’d recommend “Real-Time Rendering” by Tomas Akenine-Moller et al Its my go to resource for anything graphics related and “Geometric Tools for Computer Graphics” by Philip Schneider and David Eberly is another solid choice these books should cover the math and give more examples that can help you understand the core concepts

One final optimization idea that I've used in the past is to use SIMD instructions this is more advanced but if you're doing a lot of calculations on many lines and spheres then you should definitely try it using for example SSE or AVX on x86 processors or NEON on ARM these can process multiple data points with a single instruction and can speed up things by an order of magnitude but it depends on the hardware you are working on. I once tried optimizing this routine and it became so fast that I swear I could feel the electrons moving in my cpu its a feeling I cant describe haha.

here's a small example that uses only one function to test all the distances at once it's not SIMD this one but its a good example of an abstraction that can help simplify your code:

```cpp

struct CollisionInfo {
    float min_t;
    bool hit;
};



CollisionInfo closest_intersection(Line line, Sphere * spheres, int numSpheres) {
  CollisionInfo best;
  best.hit = false;
  best.min_t = INFINITY;


  for (int i = 0; i < numSpheres; i++){
     Intersection result = intersect_line_sphere(line, spheres[i]);
      if (result.hit && result.t < best.min_t) {
          best.hit = true;
          best.min_t = result.t;
      }

  }
    return best;

}

```

So to recap line sphere intersection math can be tricky but if you take it step by step and are careful with the math it should work every single time and also consider using optimizations to avoid unnecessary calculations that should do the trick.
