---
title: "How can CUDA call Thrust kernel functions on multiple object members?"
date: "2025-01-30"
id: "how-can-cuda-call-thrust-kernel-functions-on"
---
The challenge of applying Thrust kernels across multiple object members arises when attempting to parallelize operations on structured data, where each data instance is stored as part of a larger class or struct. Thrust, by design, operates on flat, contiguous memory regions. Directly accessing member data from within a Thrust kernel is not permitted, as it breaks the abstraction of CUDA device memory and host data structures. My experience in simulating large-scale fluid dynamics using custom particle systems brought this problem into sharp focus. The solution lies in effectively projecting the desired member data into temporary, contiguous arrays that Thrust can consume, performing the necessary operations, and then transferring the processed data back into the original object members. This methodology maintains Thrust's performance benefits while enabling complex operations on structured data.

The core concept involves creating temporary device vectors that store copies of the object members we intend to operate on. Thrust algorithms, then, work on these vectors. Crucially, after the processing, the results are copied back to their respective object members.  This approach sidesteps the limitations of direct member access from within Thrust kernels while maintaining correct data integrity and preserving encapsulation. Memory management is critical; allocation and deallocation of these temporary vectors must be meticulously handled to avoid resource leaks.

Let's illustrate this with a practical example using a hypothetical class `Particle` which we wish to manipulate. We assume `Particle` has a member, `velocity`, of type `float3` (a custom structure or a standard CUDA float3). Our task involves scaling each particle's velocity by a factor of, say, 2. The initial setup will involve preparing a vector of `Particle` objects on the host, and then transferring this to the device memory.

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

struct float3 { float x, y, z; }; // Simplified float3 definition

struct Particle {
    float3 position;
    float3 velocity;
};


// Host side code
void scaleParticleVelocities(std::vector<Particle>& particles) {
    int numParticles = particles.size();
    if (numParticles == 0) return;

    // 1. Transfer to device
    thrust::device_vector<Particle> d_particles = particles;

    // 2. Extract velocity data to a temporary device vector
    thrust::device_vector<float3> d_velocities(numParticles);
    thrust::transform(d_particles.begin(), d_particles.end(), d_velocities.begin(),
                    [](const Particle& p) { return p.velocity;});


    // 3. Scale the velocities with thrust using a lambda
    thrust::transform(d_velocities.begin(), d_velocities.end(), d_velocities.begin(),
                     thrust::multiplies<float3>( {2.0f, 2.0f, 2.0f} ));

    // 4. Copy the modified velocities back into the particles.
    thrust::transform(d_particles.begin(), d_particles.end(), d_velocities.begin(), d_particles.begin(),
        [](Particle& p, const float3& v) { p.velocity = v; return p; });

    //5. Transfer results back to the host.
    particles = d_particles;
}
```

In this first example, the lambda captures the essential operation of scaling.  The `transform` function then applies this to the temporary vector `d_velocities`. We then apply the results back to the device `d_particles` vector. Finally, the vector is transferred back to the host.  This implementation makes minimal assumption of any further functionality.

Now, consider a slightly more complex scenario, calculating the kinetic energy of each particle. The kinetic energy calculation involves a reduction operation which requires a different approach within thrust. Again, we operate using temporary vectors to store relevant data.

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric> // for std::inner_product

struct float3 { float x, y, z; };

struct Particle {
    float3 position;
    float3 velocity;
    float mass;
};


// Host side code
std::vector<float> calculateKineticEnergy(const std::vector<Particle>& particles){
    int numParticles = particles.size();
    if (numParticles == 0) return {};
    
    // 1. transfer to device
    thrust::device_vector<Particle> d_particles = particles;
    
    // 2. extract the mass and velocities 
    thrust::device_vector<float> d_masses(numParticles);
    thrust::device_vector<float3> d_velocities(numParticles);
    
    thrust::transform(d_particles.begin(), d_particles.end(), d_masses.begin(), [](const Particle& p){ return p.mass;});
    thrust::transform(d_particles.begin(), d_particles.end(), d_velocities.begin(), [](const Particle& p){ return p.velocity;});


    // 3. Calculate the magnitude of velocity
    thrust::device_vector<float> d_velocity_magnitudes(numParticles);
    thrust::transform(d_velocities.begin(), d_velocities.end(), d_velocity_magnitudes.begin(), 
        [](const float3& v){ return sqrt(std::inner_product(&v.x, &v.x + 3, &v.x, 0.0f)); });

    // 4. Calculate KE on the device
    thrust::device_vector<float> d_kinetic_energies(numParticles);
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_masses.begin(),d_velocity_magnitudes.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(d_masses.end(), d_velocity_magnitudes.end())),
                    d_kinetic_energies.begin(),
                    [](thrust::tuple<float, float> t){ return 0.5f * thrust::get<0>(t) * thrust::get<1>(t) * thrust::get<1>(t);});

    // 5. Copy the results back to host
    std::vector<float> kinetic_energies(numParticles);
    thrust::copy(d_kinetic_energies.begin(), d_kinetic_energies.end(), kinetic_energies.begin());

    return kinetic_energies;

}
```

Here, the use of `thrust::make_zip_iterator` allows the transformation to operate element-wise over both the mass and velocity magnitude, which we have first calculated. This is an important use case when combining several member data. We have also introduced the reduction operation of summing the inner product.

Finally, consider a case where we need to update a member based on a different member within the same object, using a custom calculation. Let's say we want to update the position of each particle based on a simple time-step integration of velocity: `newPosition = oldPosition + velocity * dt`. We'll assume a `float dt` is passed as a parameter.

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

struct float3 { float x, y, z; };

struct Particle {
    float3 position;
    float3 velocity;
};

// Host side code
void updateParticlePositions(std::vector<Particle>& particles, float dt) {
    int numParticles = particles.size();
    if (numParticles == 0) return;

    // 1. Transfer to device
    thrust::device_vector<Particle> d_particles = particles;

    // 2. Update particle positions on the device using a lambda
    thrust::transform(d_particles.begin(), d_particles.end(), d_particles.begin(),
        [dt](Particle& p) {
            p.position.x += p.velocity.x * dt;
            p.position.y += p.velocity.y * dt;
            p.position.z += p.velocity.z * dt;
            return p;
        });

    // 3. Transfer updated particles back to the host
    particles = d_particles;
}
```
In this case, we update in place, with the updated member directly replacing the old member. The `transform` method utilizes an in-place algorithm, directly affecting the `position` member based on the `velocity` and the scalar `dt`.

For further exploration, I recommend consulting the official NVIDIA Thrust documentation. Additionally, books and online courses focusing on CUDA parallel programming offer invaluable insights.  Texts focusing specifically on GPU-based data structures and algorithms can greatly improve one's understanding of best practices in memory management.  Furthermore, practicing with real-world simulations or physics engines can reinforce the practical applications of these techniques. The CUDA Toolkit documentation itself is an essential reference and should be considered the definitive resource for CUDA API details.  These resources provide a more detailed understanding of the underlying architecture and available libraries.
