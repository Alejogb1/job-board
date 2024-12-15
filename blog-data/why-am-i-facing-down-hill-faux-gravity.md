---
title: "Why am I facing down hill faux gravity?"
date: "2024-12-15"
id: "why-am-i-facing-down-hill-faux-gravity"
---

so, you're experiencing this weird "downhill faux gravity" thing, that's interesting. i've definitely been down that rabbit hole before, and it's usually a combination of a few suspects, rarely ever just one. let me explain where i'm coming from, maybe it will resonate.

early on in my game development journey, i had this racing game prototype. the cars felt like they were always sliding downhill, even on perfectly flat surfaces. it drove me nuts for a week because the physics engine i was using, a simple implementation of verlet integration, should have been behaving neutrally. i spent hours staring at the code, printing debug values to the console, basically living on coffee and frustration.

it turned out that the issue was multi-faceted: a tiny bias in my position update, compounded by a non-standard way i was handling collision response, and some rounding errors. each was so small on its own that it took me forever to actually see it happening. it was a classic case of a system with multiple interacting flaws creating a much bigger, more apparent overall problem.

so, breaking down what can usually cause this "downhill faux gravity" feeling, let's think of this as layers. we can go one by one.

first off, we need to look at your position updates. even the tiniest little bias here can accumulate over time and give that feeling of a constant downhill pull. i once messed up an update method and ended up with a drift where my objects would gradually descend even when I wasn't telling them to. it's embarrassing now but at the time it felt like a major design flaw. if you have code like this, well, i can relate:

```c++
   // bad example
    void updatePosition(float dt) {
        velocity.y += gravity * dt;
        position += velocity * dt;
    }
```

the problem here is that the velocity is being updated first, and then the position. it doesn’t look like much, but that slight offset creates a continuous “drift” in the direction of the gravity, and this is how it'll look if your gravity vector is not directly vertical. the correct method of doing this in a typical physics system would be something like this:

```c++
// better, more standard approach
void updatePosition(float dt) {
    Vector3 acceleration = Vector3(0, gravity, 0); // gravity applied as an acceleration, the Vector3 is a simple class with x y z floats
    Vector3 velocity_half = velocity + acceleration * dt * 0.5f;
    position = position + velocity_half * dt;
    velocity = velocity_half + acceleration * dt * 0.5f;
}
```

this integrates more accurately and reduces that unwanted bias, by calculating the velocity halfway through the frame, this is called a semi-implicit euler integration. i'd recommend looking into numerical integration methods like verlet or runge-kutta if you need more precision, "game physics engine development" by ian millington is pretty good for an in depth dive into it.

another area where issues arise are in collision handling. if your collision response isn't perfectly accurate, or if your collision detection is a bit off, you might end up with objects perpetually being "nudged" downwards. imagine a scenario where your character is slightly penetrating a surface. you might have a collision resolution that pushes it *almost* out of the surface, leaving a tiny bit of penetration every single frame. this can compound fast, it will feel like a slide. in the racing game i worked on that happened and the car kept falling into the ground because of this. this is an example of a flawed collision response:

```c++
   // flawed collision response
    void resolveCollision(Vector3 normal, float penetration) {
        position += normal * penetration; // bad idea!
    }
```

pushing the object directly out using the penetration as the magnitude can create this continuous "settling" effect where the object is never fully free of contact and slides a bit every frame. you typically want to push out along the normal and also adjust velocity accordingly to prevent this. a proper way of doing this should look like this:

```c++
   // better collision resolution
    void resolveCollision(Vector3 normal, float penetration, float restitution) {
       position += normal * penetration;
       velocity = velocity - (normal * dotProduct(velocity,normal) * (1 + restitution));
    }
```
where restitution is a float, usually between 0.0f and 1.0f defining how much velocity is retained upon impact. a great book on this is "real-time collision detection" by christer ericson. it goes into way more detail about these things than i can cover here. i spent a whole summer going through that book, it paid off though. it also explains things like contact manifold generation, which is another potential point of failure if you are dealing with multiple collisions at the same time.

finally, floating point numbers. these are the bane of any computational physicist. computers can’t store all the numbers with absolute precision, there are always small errors. these small errors can accumulate, and when you do a lot of calculations every frame the inaccuracies become noticeable, especially when dealing with physics. that small tiny numerical error, when multiplied over thousands of times per second creates a drift in your object position. if your world coordinates are huge, then the small error is amplified by the magnitude of the coordinate you are using and things will look like they are going down hill way more often.

there was a moment i used very large world coordinates, without thinking. my physics suddenly started acting like everything was on a sloped surface. it took me a while to pinpoint that the float precision was just running out. i ended up having to use a technique where everything was local and all the math was computed in a very localized space, but that story is for another time. it's funny, i sometimes feel like i should write a book on all the physics engine nightmares i've had.

as an example here is how a vector addition can lose precision when using floating point numbers:

```c++
   float a = 1000000.0f;
   float b = 0.000001f;
   float c = a + b;
   // c will most probably be just 1000000.0f because 0.000001 is not representable at that order of magnitude.
   // this example is exaggerated, but it happens nonetheless in every calculation.
```

so, to recap, check those three areas:

*   **position update:** use a method that’s more accurate, semi-implicit euler is a good start.
*   **collision handling:** ensure you are resolving collisions correctly, pushing objects along the normals and correcting the velocities.
*   **floating-point errors:** are your world coordinates too big? are you minimizing error accumulation?

if you went through all of this you will probably find your faux gravity issue. it’s usually some combination of these elements. it’s the classic "it’s always the last thing you check" kind of scenario. don't get disheartened, physics is hard. i hope this makes the issue a little less painful and that it makes sense. let me know how it goes.
