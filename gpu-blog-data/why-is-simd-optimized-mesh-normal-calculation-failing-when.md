---
title: "Why is SIMD-optimized mesh normal calculation failing when converting from C++?"
date: "2025-01-30"
id: "why-is-simd-optimized-mesh-normal-calculation-failing-when"
---
The observed failure in SIMD-optimized mesh normal calculation after a C++ conversion often stems from subtle differences in how memory is accessed and interpreted by the SIMD instructions compared to scalar code, especially with regard to data alignment and instruction selection. I've personally encountered this several times during the porting of rendering engines, and it's rarely a problem with the core math itself.

A primary culprit is data alignment. SIMD instructions operate most efficiently on data that is aligned to specific memory boundaries. For example, SSE instructions often require 128-bit (16-byte) alignment, AVX instructions 256-bit (32-byte), and AVX-512 instructions 512-bit (64-byte). Scalar C++ code, particularly when working with user-defined structures for vertices, often does not guarantee this alignment. When these structures are packed tightly without alignment considerations, attempting to load 4 or 8 floats using SIMD instructions results in memory access violations, unpredictable behavior, or incorrect calculations due to reading from or writing across incorrect memory locations.

Secondly, instruction selection can be a factor. The C++ compiler's auto-vectorization capabilities, even with optimization flags, don't always produce the most optimal SIMD instructions. What works correctly on one architecture may not translate well to another, even if the target architecture nominally supports the same SIMD extensions. Specifically, when porting existing C++ code that is not explicitly written with SIMD in mind, the compiler might choose scalar operations or inefficient SIMD combinations, inadvertently breaking the logical equivalence expected by your parallelized version.

The third significant factor relates to the handling of corner cases and edge vertices in a mesh. Scalar loops usually increment sequentially. SIMD, however, processes multiple elements at once. If special conditions are needed for edge elements (e.g., when performing neighbor lookups in triangle meshes), these special conditions might be inadvertently skipped by the parallel processing if not properly masked. This requires careful consideration of how the SIMD loops handle the boundaries of the data, potentially masking or splitting execution to avoid out-of-bounds reads or incorrect results.

Let’s illustrate this with concrete examples.

**Code Example 1: Unaligned Vertex Data**

Assume a simplified vertex structure in scalar code:

```c++
struct Vertex {
    float x, y, z;
};

void computeNormalsScalar(Vertex* vertices, int numVertices, int* indices, int numTriangles, float* normals) {
    for (int i = 0; i < numTriangles; ++i) {
        int idx0 = indices[i * 3];
        int idx1 = indices[i * 3 + 1];
        int idx2 = indices[i * 3 + 2];

        Vertex v0 = vertices[idx0];
        Vertex v1 = vertices[idx1];
        Vertex v2 = vertices[idx2];

        float vx1 = v1.x - v0.x;
        float vy1 = v1.y - v0.y;
        float vz1 = v1.z - v0.z;

        float vx2 = v2.x - v0.x;
        float vy2 = v2.y - v0.y;
        float vz2 = v2.z - v0.z;

        float nx = vy1 * vz2 - vz1 * vy2;
        float ny = vz1 * vx2 - vx1 * vz2;
        float nz = vx1 * vy2 - vy1 * vx2;

        float len = sqrt(nx * nx + ny * ny + nz * nz);
        normals[idx0 * 3] += nx / len;
        normals[idx0 * 3 + 1] += ny / len;
        normals[idx0 * 3 + 2] += nz / len;

        normals[idx1 * 3] += nx / len;
        normals[idx1 * 3 + 1] += ny / len;
        normals[idx1 * 3 + 2] += nz / len;

        normals[idx2 * 3] += nx / len;
        normals[idx2 * 3 + 1] += ny / len;
        normals[idx2 * 3 + 2] += nz / len;
    }
}
```

Here, the `Vertex` struct is likely not aligned. Converting this to SIMD might involve attempting to load vertex data directly with instructions like `_mm_loadu_ps` which works on unaligned data but is less efficient and could potentially cross page boundaries, depending on the mesh structure. However, using aligned instructions might result in crashes or incorrect data. The scalar code also increments the `normals` array by three, implicitly assuming that data will not cross vector register boundaries. However, each vertex might be used in multiple triangles, which is then compounded by being unaligned and possibly written out of bounds.

**Code Example 2: Basic SIMD Conversion with Alignment Issue**

A naive SIMD conversion, without explicit alignment considerations, could look like this (using intrinsics for illustrative purposes):

```c++
#include <immintrin.h> // Needed for Intel AVX intrinsics

void computeNormalsSIMD_Naive(Vertex* vertices, int numVertices, int* indices, int numTriangles, float* normals) {
    for (int i = 0; i < numTriangles; i++) {
        int idx0 = indices[i * 3];
        int idx1 = indices[i * 3 + 1];
        int idx2 = indices[i * 3 + 2];

        __m256 v0 = _mm256_loadu_ps((float*)&vertices[idx0]); // Load 3 floats (x,y,z) + 1 undefined
        __m256 v1 = _mm256_loadu_ps((float*)&vertices[idx1]);
        __m256 v2 = _mm256_loadu_ps((float*)&vertices[idx2]);

        __m256 vx1 = _mm256_sub_ps(v1, v0); // v1 - v0
        __m256 vx2 = _mm256_sub_ps(v2, v0); // v2 - v0


        __m256 nx = _mm256_sub_ps(_mm256_mul_ps(_mm256_permute_ps(vx1, 0b10010001), _mm256_permute_ps(vx2, 0b10000110)),
                                _mm256_mul_ps(_mm256_permute_ps(vx1, 0b10000110), _mm256_permute_ps(vx2, 0b10010001)));

		__m256 ny = _mm256_sub_ps(_mm256_mul_ps(_mm256_permute_ps(vx1, 0b01100010), _mm256_permute_ps(vx2, 0b00101001)),
								_mm256_mul_ps(_mm256_permute_ps(vx1, 0b00101001), _mm256_permute_ps(vx2, 0b01100010)));

		__m256 nz = _mm256_sub_ps(_mm256_mul_ps(_mm256_permute_ps(vx1, 0b00011001), _mm256_permute_ps(vx2, 0b00010110)),
								_mm256_mul_ps(_mm256_permute_ps(vx1, 0b00010110), _mm256_permute_ps(vx2, 0b00011001)));


		__m256 len = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)), _mm256_mul_ps(nz, nz)));

        nx = _mm256_div_ps(nx, len);
        ny = _mm256_div_ps(ny, len);
        nz = _mm256_div_ps(nz, len);

        _mm256_storeu_ps(&normals[idx0 * 3], _mm256_add_ps(_mm256_loadu_ps(&normals[idx0 * 3]), nx)); // Accumulate. Unaligned store.
        _mm256_storeu_ps(&normals[idx1 * 3], _mm256_add_ps(_mm256_loadu_ps(&normals[idx1 * 3]), ny)); // Accumulate. Unaligned store.
        _mm256_storeu_ps(&normals[idx2 * 3], _mm256_add_ps(_mm256_loadu_ps(&normals[idx2 * 3]), nz)); // Accumulate. Unaligned store.
    }
}
```

This example attempts to use `_mm256_loadu_ps`, the unaligned load instruction, which works, but performs poorly and as with the scalar code will write out of bounds. The accumulation to the `normals` buffer is done incorrectly as well. It attempts to load four floats at a time, add the newly calculated normal, and store it back into memory, while only three floats of the normal are relevant.

**Code Example 3: Corrected SIMD Conversion with Alignment**

A better implementation involves aligning the vertex data and carefully addressing the accumulation:

```c++
#include <immintrin.h> // Needed for Intel AVX intrinsics

struct AlignedVertex {
    float x, y, z, pad; // Add padding for 16-byte alignment
};

void computeNormalsSIMD_Aligned(AlignedVertex* vertices, int numVertices, int* indices, int numTriangles, float* normals) {
    for (int i = 0; i < numTriangles; ++i) {
        int idx0 = indices[i * 3];
        int idx1 = indices[i * 3 + 1];
        int idx2 = indices[i * 3 + 2];

        __m256 v0 = _mm256_load_ps((float*)&vertices[idx0]); //Aligned load!
        __m256 v1 = _mm256_load_ps((float*)&vertices[idx1]);
        __m256 v2 = _mm256_load_ps((float*)&vertices[idx2]);


        __m256 vx1 = _mm256_sub_ps(v1, v0);
        __m256 vx2 = _mm256_sub_ps(v2, v0);


        __m256 nx = _mm256_sub_ps(_mm256_mul_ps(_mm256_permute_ps(vx1, 0b10010001), _mm256_permute_ps(vx2, 0b10000110)),
                                _mm256_mul_ps(_mm256_permute_ps(vx1, 0b10000110), _mm256_permute_ps(vx2, 0b10010001)));

		__m256 ny = _mm256_sub_ps(_mm256_mul_ps(_mm256_permute_ps(vx1, 0b01100010), _mm256_permute_ps(vx2, 0b00101001)),
								_mm256_mul_ps(_mm256_permute_ps(vx1, 0b00101001), _mm256_permute_ps(vx2, 0b01100010)));

		__m256 nz = _mm256_sub_ps(_mm256_mul_ps(_mm256_permute_ps(vx1, 0b00011001), _mm256_permute_ps(vx2, 0b00010110)),
								_mm256_mul_ps(_mm256_permute_ps(vx1, 0b00010110), _mm256_permute_ps(vx2, 0b00011001)));


		__m256 len = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)), _mm256_mul_ps(nz, nz)));

        nx = _mm256_div_ps(nx, len);
        ny = _mm256_div_ps(ny, len);
        nz = _mm256_div_ps(nz, len);

		float tempNx[8];
		float tempNy[8];
		float tempNz[8];

		_mm256_store_ps(tempNx, nx);
        _mm256_store_ps(tempNy, ny);
        _mm256_store_ps(tempNz, nz);

		normals[idx0 * 3] += tempNx[0];
        normals[idx0 * 3 + 1] += tempNy[0];
        normals[idx0 * 3 + 2] += tempNz[0];
		normals[idx1 * 3] += tempNx[1];
        normals[idx1 * 3 + 1] += tempNy[1];
        normals[idx1 * 3 + 2] += tempNz[1];
		normals[idx2 * 3] += tempNx[2];
        normals[idx2 * 3 + 1] += tempNy[2];
        normals[idx2 * 3 + 2] += tempNz[2];
    }
}
```

This revised code uses `AlignedVertex` to ensure proper alignment and loads vertex data with `_mm256_load_ps` which requires aligned memory. This example also accumulates the normals by copying them from SIMD vectors and then incrementing the scalar result.

**Resource Recommendations**

To deepen understanding of this subject and to avoid the type of pitfalls I described, consider the following resources:

1. **Processor Specific Instruction Manuals**: Refer to the Intel or AMD architecture manuals. These documents provide precise details on SIMD instruction behavior, memory requirements, and performance considerations. These are invaluable for understanding subtle nuances.

2.  **Compiler Optimization Documentation**: Explore the documentation for your compiler (e.g., GCC, Clang, MSVC). Pay close attention to the sections covering auto-vectorization, optimization flags, and how to leverage compiler support for SIMD. These resources can help you understand what the compiler is doing under the hood.

3.  **Online forums**: Engage with community forums focused on performance optimization. These places can help solve practical issues. Keep in mind that these solutions often need further debugging, and may not work best for all use cases.

In summary, the difficulties in porting SIMD code for mesh normal calculations often result from memory alignment mismatches, suboptimal instruction selection, and naive handling of corner cases. Attention to data structure alignment, and testing of your implementations, are paramount. By addressing these considerations, you can reliably achieve the performance benefits of SIMD in your application.
