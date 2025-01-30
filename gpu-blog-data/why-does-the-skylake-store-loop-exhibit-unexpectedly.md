---
title: "Why does the Skylake store loop exhibit unexpectedly poor and bimodal performance?"
date: "2025-01-30"
id: "why-does-the-skylake-store-loop-exhibit-unexpectedly"
---
The Skylake microarchitecture's store loop performance can degrade significantly and exhibit bimodal behavior primarily due to interactions between the store buffer, the memory subsystem's caching hierarchy, and the prefetcher. This is something I’ve observed numerous times debugging high-performance numerical code on Intel platforms. A seemingly simple loop storing to memory can become unexpectedly slow, switching between "fast" and "slow" execution periods, often without an immediately obvious reason in the code itself.

**The Core Problem: Store Buffer Saturation and Cache Conflicts**

The fundamental issue lies in how Skylake (and similar Intel architectures) handle stores. They aren’t directly committed to the cache immediately; rather, they're placed into a relatively small, fixed-size store buffer. This buffer acts as a staging area, allowing the processor to continue executing instructions while the store operations are completed in the background. This is essential for achieving high throughput.

However, when this buffer fills, the processor must stall, waiting for space to become available. This stall results in a drop in performance. Specifically for loops that repeatedly write to memory, several factors can lead to store buffer saturation:

1.  **Write-back Cache Policies and Cache Misses:** When a write operation targets a cache line not currently present in the L1 cache (a write miss), the cache line must be fetched from a higher level in the cache hierarchy or main memory. This fetch introduces a latency and holds up the store operation, keeping the corresponding slot in the store buffer occupied longer. Subsequent store operations must then wait, contributing to store buffer fill. If the writes cause repeated L1 misses, the store buffer becomes the limiting factor.

2.  **Store-to-Load Forwarding Issues:** If a subsequent load operation targets an address in the store buffer before that store has committed to the cache, the CPU must perform store-to-load forwarding. This bypass mechanism ensures data consistency. However, store-to-load forwarding adds overhead, particularly if the forwarding isn’t straightforward. For example, the forwarding path might be longer if the data resides in multiple store buffer entries, leading to delays that increase store buffer occupancy.

3.  **Memory Ordering and Synchronization:** Explicit or implicit memory barriers (e.g., due to fences or lock acquisition/release) can force the store buffer to flush, which is also a costly operation if the store buffer is full. This effect could be unexpected because the loop itself may look free of ordering requirements, but library or runtime calls may introduce implicit synchronisation points.

4.  **Prefetcher Interactions:** The prefetcher attempts to bring data into the cache before the CPU requires it. However, its algorithms aren't perfect. Incorrect prefetching can invalidate useful cache lines, resulting in unnecessary L1 cache misses upon store operations, further impacting store buffer occupancy and overall loop performance.

The "bimodal" behavior arises because once the store buffer becomes full and stalls the processor, the system becomes locked into this "slow" mode until the backlog of stores clears. Then, when the store buffer empties sufficiently, the loop runs faster, only to slow again as the buffer fills. This cycle manifests as the bimodal oscillation of performance.

**Code Examples and Commentary**

Below, I'll provide several code examples that illustrate these issues along with explanations of why they might exhibit poor performance in a store-heavy loop:

**Example 1: Simple Array Write**

```c
void simple_write(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1.0f;
  }
}
```

*   **Explanation:** This straightforward loop writes a constant value to an array. In isolation, it may look harmless. However, if 'size' is large enough, this loop can easily saturate the store buffer, particularly if the data array isn't cached in the L1 cache initially, resulting in L1 cache misses on each write during the initial phase of execution. The writes would then fill the store buffer, causing stalls and the "slow" phase of the bimodal behaviour. Once the data array is brought to cache, it will be in the "fast" mode.

**Example 2: Stride Access**

```c
void stride_write(float* data, int size, int stride) {
  for (int i = 0; i < size; i+= stride) {
    data[i] = 1.0f;
  }
}
```

*   **Explanation:** This variant introduces a 'stride' that steps through the array. If the stride is large enough that it exceeds the L1 cache line size, each write operation will likely result in a cache miss. Each write now involves fetching the corresponding L1 cache line, potentially from L2 cache or memory, delaying the store and, again, contributing to buffer fill. Further, depending on the stride and the way the CPU maps memory addresses into cache sets (addressing bits), this stride may cause severe cache thrashing, where writes repeatedly overwrite the same L1 cache sets. This behaviour will increase L1 misses and exacerbate the store buffer performance degradation and bimodal behavior.

**Example 3: Store to Load Forwarding Issues**

```c
void store_load_forwarding(float* data, int size) {
  for (int i = 0; i < size - 1; i++){
    data[i] = 1.0f;
     float dummy = data[i];
     // Some other computations not related to memory
  }
}
```

*   **Explanation:** This example demonstrates potential issues with store-to-load forwarding. Within the loop, the program writes to an address in the array, then immediately reads it back. Because this read may happen before the write is fully committed to the L1 cache, the processor needs to forward data from the store buffer. The delay involved in store-to-load forwarding will negatively affect throughput in the loop and can cause the store buffer to fill faster, increasing the chance of stalling. The presence of the `dummy` variable doesn't change this issue of the forwarding from the store buffer. The important part is reading data from a location that was just written to.

**Mitigation Strategies**

While the described behavior is inherent in the microarchitecture, I have encountered several strategies that mitigate these performance issues:

1.  **Data Alignment and Layout:** Ensure that data structures are aligned to cache line boundaries. Proper data layout can improve cache locality and minimize the likelihood of cache misses. Arranging data to ensure multiple nearby writes fall within the same cache lines can effectively reduce misses.

2.  **Blocking/Tiling:** Decompose large computations into smaller "blocks" that fit entirely into the caches. Operate on the "blocks" sequentially. Once a block is written into memory, work on a new block to continue computation. This increases cache re-use and decreases overall memory traffic.

3.  **Prefetching (Carefully):** Implement manual prefetching using intrinsics, but do so with caution as improper prefetching can worsen performance. Make sure that prefetches are done only for locations that will be actually required.

4.  **Non-Temporal Stores:** Where appropriate, utilize non-temporal store instructions to write directly to memory without polluting caches. This should be done very carefully and only when the data is definitely not going to be reused. This reduces overall cache pressure.

5.  **Reduce Dependencies:** If possible, restructure loops to minimize dependencies, particularly between stores and loads on the same memory locations. The previous example illustrates the dangers of store-load dependency in a loop.

6.  **Software Transactional Memory:** In cases that permit, software transactional memory may help to reduce ordering constraints and memory barriers. However, this would require a very specific programming approach and runtime environment support.

**Resource Recommendations**

For in-depth information on Intel microarchitecture, I recommend consulting the Intel Software Developer Manuals, specifically the volume related to optimization. Performance analysis tools like Intel VTune Amplifier or similar profiling software provide critical insights into memory access patterns and bottlenecks. Furthermore, research papers and white papers related to cache behavior, memory hierarchy, and microarchitectural optimizations are invaluable for a deep understanding. The books "Computer Architecture: A Quantitative Approach" and "Modern Processor Design" provide a solid theoretical foundation for anyone interested in low-level computer architecture and performance analysis. These resources offer comprehensive knowledge crucial for deciphering complex behavior like the bimodal store loop performance on Skylake.
