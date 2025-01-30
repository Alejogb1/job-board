---
title: "How does Nvidia Thrust handle arbitrary transformations on a 3D grid?"
date: "2025-01-30"
id: "how-does-nvidia-thrust-handle-arbitrary-transformations-on"
---
Nvidia Thrust, a high-performance C++ template library for CUDA, does not inherently provide explicit functionality to perform *arbitrary* transformations directly on a 3D grid. Instead, it relies on a combination of its parallel primitives – like `transform`, `for_each`, `copy`, and sorting routines – and user-defined functors to achieve such transformations indirectly. My experiences optimising finite element simulations that heavily relied on Thrust for parallelisation have underscored this core principle; Thrust provides the building blocks, not a pre-packaged 3D transformation suite. The responsibility of correctly translating indices and applying the desired geometric changes rests squarely on the developer.

The fundamental mechanism for handling these transformations in Thrust involves defining a functor that, given a linear index representing a grid cell, computes the transformed coordinates or data associated with that cell. This approach is flexible, allowing for any complex transformation, not only affine ones like rotation, scaling, or translation. The key insight is that Thrust operates on sequences, and a 3D grid can be flattened into a 1D sequence for its parallel operations. Subsequently, the transformed data can either be used directly or copied back into a new data structure corresponding to the transformed grid, if necessary. No explicit 3D grid abstraction or transformation library exists within Thrust itself.

For instance, let's consider a simple case: applying a linear translation to a 3D grid. Given a 3D grid with dimensions `nx`, `ny`, and `nz`, and corresponding data `grid_data`, I might define a translation vector `tx`, `ty`, and `tz`. To apply this translation, I would create a custom functor that calculates the new coordinates. Assume `grid_data` is stored as a linear array with standard row-major ordering. The corresponding linear index `i` maps to 3D coordinates `(x, y, z)` via:

```c++
x = i % nx;
y = (i / nx) % ny;
z = i / (nx * ny);
```

Then, to apply a translation, I can define the functor to calculate the transformed x, y, and z (x', y', z') using:

```c++
x' = x + tx;
y' = y + ty;
z' = z + tz;
```

If the transformed coordinates fall outside the bounds of the original grid, handling that case is a design consideration for the functor (e.g. clipping, wrapping, or assigning a default value). Here is code demonstrating this:

```c++
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>

struct TranslateFunctor {
    int nx, ny, nz;
    float tx, ty, tz;
    float* grid_data;
    float* output_data;

    TranslateFunctor(int nx, int ny, int nz, float tx, float ty, float tz, float* grid_data, float* output_data)
        : nx(nx), ny(ny), nz(nz), tx(tx), ty(ty), tz(tz), grid_data(grid_data), output_data(output_data) {}

    __device__ void operator()(const int i) {
        int x = i % nx;
        int y = (i / nx) % ny;
        int z = i / (nx * ny);

        int x_transformed = x + tx;
        int y_transformed = y + ty;
        int z_transformed = z + tz;

        if (x_transformed >= 0 && x_transformed < nx &&
            y_transformed >= 0 && y_transformed < ny &&
            z_transformed >= 0 && z_transformed < nz){
            output_data[i] = grid_data[z_transformed * nx * ny + y_transformed * nx + x_transformed];
        } else {
            output_data[i] = 0.0f; // or some other default
        }
    }
};

void performTranslation(int nx, int ny, int nz, float tx, float ty, float tz,
                       thrust::device_vector<float>& grid_data, thrust::device_vector<float>& output_data) {
        int size = nx * ny * nz;
        TranslateFunctor functor(nx, ny, nz, tx, ty, tz, thrust::raw_pointer_cast(grid_data.data()), thrust::raw_pointer_cast(output_data.data()));
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(size),
                         functor);
}
```

In this first example, the `TranslateFunctor` computes the transformed coordinates and copies data from the original location to the translated location in the `output_data`. The out-of-bounds case is handled by simply writing 0.0 to the `output_data`. The core computation and the conditional bounds check execute on the GPU in parallel. The `performTranslation` function is responsible for allocating device memory, initializing the functor, and calling the `for_each` algorithm.

Now, consider applying a non-affine transformation, such as a radial distortion, to the same grid. We need to define a different functor. Assuming the center of the grid is approximately `(nx/2, ny/2, nz/2)`, I can define the following:

```c++
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <cmath>

struct RadialDistortFunctor {
    int nx, ny, nz;
    float distortion_factor;
    float* grid_data;
    float* output_data;

    RadialDistortFunctor(int nx, int ny, int nz, float distortion_factor, float* grid_data, float* output_data)
        : nx(nx), ny(ny), nz(nz), distortion_factor(distortion_factor), grid_data(grid_data), output_data(output_data) {}

    __device__ void operator()(const int i) {
      int x = i % nx;
      int y = (i / nx) % ny;
      int z = i / (nx * ny);

      float centerX = nx / 2.0f;
      float centerY = ny / 2.0f;
      float centerZ = nz / 2.0f;

      float dx = x - centerX;
      float dy = y - centerY;
      float dz = z - centerZ;
      float radius = sqrt(dx*dx + dy*dy + dz*dz);

      float distortion = 1.0f + distortion_factor * radius;

      int x_transformed =  static_cast<int>(centerX + dx * distortion);
      int y_transformed = static_cast<int>(centerY + dy * distortion);
      int z_transformed =  static_cast<int>(centerZ + dz * distortion);


        if (x_transformed >= 0 && x_transformed < nx &&
            y_transformed >= 0 && y_transformed < ny &&
            z_transformed >= 0 && z_transformed < nz){
            output_data[i] = grid_data[z_transformed * nx * ny + y_transformed * nx + x_transformed];
        } else {
            output_data[i] = 0.0f;
        }
    }
};


void performRadialDistort(int nx, int ny, int nz, float distortion_factor,
                       thrust::device_vector<float>& grid_data, thrust::device_vector<float>& output_data) {
        int size = nx * ny * nz;
        RadialDistortFunctor functor(nx, ny, nz, distortion_factor, thrust::raw_pointer_cast(grid_data.data()), thrust::raw_pointer_cast(output_data.data()));
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(size),
                         functor);
}

```

This functor, `RadialDistortFunctor`, calculates the radial distortion based on the distance from the grid center and applies it to each coordinate. Again, the transformed coordinates are checked before accessing `grid_data` to prevent out-of-bounds access. In both examples, data from the original grid `grid_data` is transformed and then copied into `output_data`. If you need to perform in-place transformations, you would need to overwrite the grid’s data in place and account for potential race conditions. This typically involves an intermediate copy or careful synchronisation, making it more complex.

A third, more complex, scenario is performing a non-uniform grid transformation. This requires each index to map to an entirely new coordinate. This involves an auxiliary data array to provide the destination indices. Consider a scenario where we’ve computed a distorted grid of the same dimensions, and stored the destination indices in a separate set of arrays: `x_dest`, `y_dest`, and `z_dest`. The functor would look like this:

```c++
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>

struct NonUniformTransformFunctor {
    int nx, ny, nz;
    int* x_dest;
    int* y_dest;
    int* z_dest;
    float* grid_data;
    float* output_data;


    NonUniformTransformFunctor(int nx, int ny, int nz, int* x_dest, int* y_dest, int* z_dest,  float* grid_data, float* output_data)
        : nx(nx), ny(ny), nz(nz), x_dest(x_dest), y_dest(y_dest), z_dest(z_dest), grid_data(grid_data), output_data(output_data) {}

    __device__ void operator()(const int i) {
        int x_transformed = x_dest[i];
        int y_transformed = y_dest[i];
        int z_transformed = z_dest[i];


        if (x_transformed >= 0 && x_transformed < nx &&
            y_transformed >= 0 && y_transformed < ny &&
            z_transformed >= 0 && z_transformed < nz) {
              output_data[z_transformed * nx * ny + y_transformed * nx + x_transformed] = grid_data[i];
        } else {
            // handle invalid destination index, perhaps zeroing output
              output_data[i] = 0.0f;
        }
    }
};


void performNonUniformTransform(int nx, int ny, int nz, thrust::device_vector<int>& x_dest, thrust::device_vector<int>& y_dest, thrust::device_vector<int>& z_dest,
                       thrust::device_vector<float>& grid_data, thrust::device_vector<float>& output_data) {
    int size = nx * ny * nz;
    NonUniformTransformFunctor functor(nx, ny, nz, thrust::raw_pointer_cast(x_dest.data()), thrust::raw_pointer_cast(y_dest.data()), thrust::raw_pointer_cast(z_dest.data()),  thrust::raw_pointer_cast(grid_data.data()), thrust::raw_pointer_cast(output_data.data()));
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(size),
                         functor);
}
```

In this example, the `NonUniformTransformFunctor` directly looks up the destination indices from `x_dest`, `y_dest` and `z_dest` arrays which were pre-computed outside of Thrust based on an arbitrary deformation. The transformed grid is in this case stored in `output_data`. Note, that this example demonstrates a data transfer from one grid to another but the underlying grid locations are arbitrary, and must be precomputed.

For further exploration, I recommend studying the Thrust documentation, focusing on the `transform`, `for_each`, and `reduce` algorithms. Also, familiarizing oneself with CUDA's execution model and memory management is critical for effective use.  Finally, understanding techniques for handling out-of-bounds conditions when performing grid transformations, as showcased in the provided examples, is vital to avoid potential crashes or unexpected results. These three elements are the pillars of effectively using Thrust for any form of grid manipulation.
