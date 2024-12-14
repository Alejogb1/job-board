---
title: "Why is AddForce accumulating on NavMeshAgent and Their Movement gets weird?"
date: "2024-12-14"
id: "why-is-addforce-accumulating-on-navmeshagent-and-their-movement-gets-weird"
---

alright, so you’re seeing some janky movement when applying `addforce` to a game object that also has a `navmeshagent` attached. i get it, i’ve been there, pulling my hair out at 3 am trying to figure out why my perfectly planned pathfinding is acting like a drunk teenager. this is a classic problem with a few interconnected causes, and honestly, it took me a couple of game jams to really nail down the nuances. i’ve probably wasted more hours than i care to think about on this issue, so let's break it down.

the core problem is this: `navmeshagent` is built to control movement based on pathfinding calculations. it sets velocities and positions based on the navigation mesh, aiming for a specific destination. then you come in with `addforce`, which doesn't care about the navmesh agent and applies forces directly to the object's rigidbody component. so, you've got two separate systems trying to control the same object's movement at the same time. it's like having two drivers on one steering wheel, each trying to go a different direction. not ideal.

the biggest culprit is that `addforce` is designed to push an object and give it velocity and momentum, while the `navmeshagent` is designed to move the object by directly setting its velocity or position to follow the planned navigation path. when you apply force, you're essentially adding a velocity vector to the current velocity vector the `navmeshagent` is already trying to implement. the two forces combine, and the navmesh agent tries to adjust, sometimes it does it very strangely and gives that odd movement effect, like it over-corrects or gets thrown out of course. the agent is trying to correct itself because it's no longer moving along its intended path.

sometimes you have that specific situation where you have a gameobject that has an active navmeshagent, and you, for example, shoot the gameobject with a projectile using addforce for the gameobject to react to the impact and then it starts behaving in a strange way after that. you would think the `navmeshagent` would recover but that is not always the case. depending on how much force you apply to the gameobject and the internal state of the `navmeshagent` it can produce erratic movement, especially if the force is against the navmeshagent path, because now you have two vectors opposing each other. this situation is a big factor for causing weird movement. this is also true if you are doing a physics simulation of an entity using addforce and then you are activating a `navmeshagent` for the same entity.

the `navmeshagent` tries to use its internal calculations to correct the new velocity but it is never a smooth transition since `addforce` is an external, uncontrolled force. that can lead to a buildup of error over multiple frames leading to unexpected jerking movements, and unpredictable behaviours.

another thing is the mass and drag of the rigidbody attached to the agent. if you are applying a relatively small force to a really massive object it might not even budge. if you are applying a huge force to a very light object it might go flying away from its intended path, and again the `navmeshagent` will try to correct but the results might be less than ideal. the issue might also stem from the fact that you might be applying force every frame, this will accumulate and it will cause erratic movement. the `navmeshagent` expects more deterministic movement to function properly so it is important to apply force only when needed.

now, i’m not saying you should never use `addforce` with a `navmeshagent`. sometimes, you *need* that physical reaction, especially for things like knockback or ragdoll effects. the key is to use them sparingly and with a clear strategy. also if you use it, understand how you should use it.

so here’s how i usually tackle this problem, based on my painful experiences, and some hard earned wisdom i got from reading through papers on robotics control (which are actually helpful even if it's a game!).

**1. disabling the agent when applying force:**

this is the most straightforward fix, if the situation permits. before applying `addforce`, disable the `navmeshagent`. let the physics system move the object then re-enable the agent after a short time, or when needed.

```csharp
using UnityEngine;
using UnityEngine.AI;

public class ForceAndNavMesh : MonoBehaviour
{
    public float pushForce = 5f;
    public float disableTime = 0.5f;
    private NavMeshAgent _navAgent;
    private Rigidbody _rb;
    private float _timer = 0f;
    private bool _navAgentWasDisabled = false;

    void Start()
    {
        _navAgent = GetComponent<NavMeshAgent>();
        _rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
      if (_navAgentWasDisabled) {
        _timer += Time.deltaTime;
          if(_timer > disableTime) {
              _navAgent.enabled = true;
              _navAgentWasDisabled = false;
              _timer = 0;
          }
      }

      if (Input.GetKeyDown(KeyCode.Space)) {
            ApplyPush();
      }
    }

    void ApplyPush()
    {
        if (_navAgent.enabled) {
            _navAgent.enabled = false;
            _navAgentWasDisabled = true;
            _rb.AddForce(transform.forward * pushForce, ForceMode.Impulse);
        }
    }
}
```
this is a simple solution that i use a lot when the interaction is simple and involves just a push from an external force that i want to simulate. the important part is that the `navmeshagent` is disabled right before applying the force and then reenabled, this gives you a window of control, you can tweak the `disableTime` variable if the agent is re-enabled too soon.

**2. using a 'velocity dampening' system:**

if you absolutely need to use `addforce` while the agent is active (for example, for continuous pushing), you’ll need to manage the forces carefully. you can add a system that takes into account the `navmeshagent` movement, and instead of directly adding the force, you dampen the final applied velocity so it is not completely overriding the agent direction.

```csharp
using UnityEngine;
using UnityEngine.AI;

public class ForceAndNavMesh : MonoBehaviour
{
    public float pushForce = 5f;
    public float forceDamping = 0.5f;
    private NavMeshAgent _navAgent;
    private Rigidbody _rb;

    void Start()
    {
        _navAgent = GetComponent<NavMeshAgent>();
        _rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
      if (Input.GetKeyDown(KeyCode.Space)) {
          ApplyPush();
      }
    }

    void ApplyPush()
    {
       Vector3 pushDirection = transform.forward;
       Vector3 agentVelocity = _navAgent.desiredVelocity;

       Vector3 finalVelocity = (pushDirection * pushForce) * forceDamping + agentVelocity * (1 - forceDamping);
       _rb.velocity = finalVelocity;
    }
}
```
in this code, i'm mixing the intended push direction and the direction the `navmeshagent` was going to achieve the final velocity. the `forceDamping` parameter is what allows you to mix those two vectors. a value closer to 0 will give more weight to the agent's desired velocity while a value closer to 1 will give more weight to the push direction. with this you will need to experiment to get the exact behaviour you desire.

**3. using a custom movement controller:**

sometimes you might not be able to use either of the above methods and a more custom approach is needed. it could be that you need more control over how the agent reacts to forces and you want more of a hybrid movement system, blending the best of pathfinding and physics. in this case, you could build a custom movement system that takes into account both the `navmeshagent` path and external forces.

```csharp
using UnityEngine;
using UnityEngine.AI;

public class CustomMovement : MonoBehaviour
{
    public float movementSpeed = 5f;
    public float acceleration = 10f;
    public float pushForce = 5f;
    private NavMeshAgent _navAgent;
    private Rigidbody _rb;
    private Vector3 _desiredVelocity;
    private bool _applyPush = false;

    void Start()
    {
        _navAgent = GetComponent<NavMeshAgent>();
        _navAgent.updatePosition = false;
        _navAgent.updateRotation = false;
        _rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space)) {
            _applyPush = true;
        }
    }

    void FixedUpdate()
    {
        _desiredVelocity = _navAgent.desiredVelocity;
        Vector3 targetVelocity = _desiredVelocity.normalized * movementSpeed;
        Vector3 velocityChange = targetVelocity - _rb.velocity;
        Vector3 accelerationVector = velocityChange * acceleration;

        if(_applyPush) {
          accelerationVector += transform.forward * pushForce;
           _applyPush = false;
        }

        _rb.AddForce(accelerationVector, ForceMode.Acceleration);

         //sync agent position after physics calculation.
        if(_navAgent.enabled)
        {
             _navAgent.nextPosition = _rb.position;
        }

    }
}
```

in this code, i am updating the rigidbody forces using the `navmeshagent` target velocity and also injecting a push force when requested. the key element here is that i disabled the agent’s position and rotation updates ( `_navAgent.updatePosition = false;` `_navAgent.updateRotation = false;`), and i’m updating the agent’s position manually after doing all the physics calculations ( `_navAgent.nextPosition = _rb.position;`).

a couple more things, to note is that i am moving the agent using forces and `forcemode.acceleration` since `forcemode.velocitychange` and `forcemode.impulse` also changes the mass of the object (something that you might or might not want), using acceleration is more deterministic, if you need other forces then you should calculate the force using f = m * a.

also, as a rule of thumb always use fixedupdate for physics, you will get a more deterministic result. also note that i am using the agent's `desiredvelocity` not its `velocity`, the `desiredvelocity` is the vector calculated by the pathfinding, so this way it will not be affected by the rigidbody.

i cannot stress this enough, this is not a silver bullet solution, you’ll likely need to adjust the parameters based on your specific game design. the custom system gives you a very fine-grain control of the movement and the forces in the gameobject, so it is more of a solution for more specific scenarios. using a combination of those approaches (and lots of debugging) is how you make the agent react how you want.

i have heard someone say that working with physics is just a matter of adding a few more parameters to your equations until it looks good. i don’t know if i agree but it's definitely fun.

if you want to do a deeper dive into this stuff, i can recommend some more general resources not specific to unity. look into papers on motion planning, these go very deep into the theory behind these kinds of movement systems, books like "probabilistic robotics" by sebastian thrun et al. give a great theoretical background on this. also, papers on reinforcement learning (specifically hierarchical and hybrid reinforcement learning) will provide you a broader understanding on how agents can have complex behaviors.

and a little tech joke for you: why do programmers prefer dark mode? because light attracts bugs!

hope this helps, and good luck getting your navmesh agents to behave!
