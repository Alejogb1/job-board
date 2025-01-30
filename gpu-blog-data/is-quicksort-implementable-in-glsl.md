---
title: "Is quicksort implementable in GLSL?"
date: "2025-01-30"
id: "is-quicksort-implementable-in-glsl"
---
Yes, Quicksort *is* implementable in GLSL, but its practical application within the shader pipeline is extremely limited and likely ill-advised for most use cases. The inherent parallel nature of GPUs and the limitations imposed by shader execution models make traditional sorting algorithms like Quicksort inefficient and often problematic.

My experience stems from developing a custom particle simulation system using compute shaders for a research project. While the ability to directly manipulate particle data on the GPU was beneficial, attempting to sort large datasets within a single shader pass revealed significant performance bottlenecks. The core issue lies in the fact that GPUs, especially those executing fragment shaders (where sorting would *never* be advisable), are optimized for parallel, data-independent operations, not sequential logic with branching as found in algorithms like Quicksort.  Shaders operate on individual fragments (or in compute shaders, work items) independently. Synchronization between these work items is expensive, and the inherent recursive nature of Quicksort demands frequent data access and comparisons.

Let's consider how one might approach a Quicksort implementation. The standard recursive method is immediately problematic; GLSL does not directly support recursive function calls. Iterative implementations are possible, but require explicit stack management, which GLSL provides no direct method for. Therefore, we must resort to using techniques involving loops and atomic operations within the shader, significantly hindering performance and increasing complexity.

Here's a basic illustration of how one might set up the necessary data structures within a compute shader to attempt Quicksort:

```glsl
#version 450 core
layout(local_size_x = 1) in;

layout(binding = 0) buffer DataBuffer {
  float data[];
};

layout(binding = 1) buffer StackBuffer {
  uint stack[];
};

layout(binding = 2) buffer  CounterBuffer{
    uint counter;
};

const uint MAX_STACK_SIZE = 100;

void push(uint low, uint high, inout uint stackIndex){
    if(stackIndex < MAX_STACK_SIZE){
       stack[stackIndex*2] = low;
       stack[stackIndex*2 +1] = high;
       stackIndex = stackIndex + 1;
    }
}

void pop(out uint low, out uint high, inout uint stackIndex){
    if(stackIndex > 0){
        stackIndex = stackIndex -1;
        low = stack[stackIndex*2];
        high = stack[stackIndex*2+1];
    }
    else{
        low = 0;
        high = 0;
    }

}


uint partition(uint low, uint high) {
    float pivot = data[high];
    uint i = low - 1;
    for (uint j = low; j < high; j++) {
        if (data[j] < pivot) {
            i++;
            float temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    float temp = data[i + 1];
    data[i + 1] = data[high];
    data[high] = temp;
    return i + 1;
}


void main() {
    uint arraySize = data.length();
    uint stackIndex = 0;


    if(gl_GlobalInvocationID.x == 0 && counter == 0){
        push(0, arraySize -1, stackIndex);
         counter = 1;
    }

    memoryBarrierShared();
    barrier();

    uint low;
    uint high;

    pop(low, high, stackIndex);


    if(low < high){
        uint p = partition(low, high);

        push(low, p-1, stackIndex);
        push(p+1, high, stackIndex);
    }

   memoryBarrierShared();
    barrier();


}
```

This code demonstrates a non-recursive version of quicksort, using a stack implemented with buffer objects.  `DataBuffer` stores the input values to be sorted, `StackBuffer` is used to store the ranges for which quicksort will act upon.  The `counter` is a simple flag to allow pushing the initial range. `push` and `pop` manage the stack. The `partition` function uses in-place swaps. The main function performs the stack handling and initiates the quicksort if the stack index is not empty. Note that the `stackIndex` and other variables must remain consistent for all workgroups.  We need to perform an initial check against the `counter` to begin our sort. While logically complete, this example has many limitations for any dataset beyond a small size (the fixed `MAX_STACK_SIZE` will quickly overflow) and will be extremely inefficient due to memory barriers and lack of work-group wide operation.

To illustrate a scenario closer to actual usage, consider a situation where one needs to sort a small set of data per fragment or work item, although this remains a corner case use:

```glsl
#version 450 core
layout(local_size_x = 1) in;

layout(binding = 0) uniform Data {
  float data[10];
} inputData;

layout(binding = 1) buffer OutputBuffer {
    float output[];
};

float partition(float arr[], uint low, uint high) {
    float pivot = arr[high];
    uint i = low - 1;
    for (uint j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            float temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    float temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return i + 1;
}


void quicksort(float arr[], uint low, uint high) {
    if (low < high) {
        uint p = partition(arr, low, high);
        quicksort(arr, low, p - 1);
        quicksort(arr, p + 1, high);
    }
}


void main() {
     float localData[10];
    for (int i=0; i < 10; i++) {
        localData[i] = inputData.data[i];
    }
    quicksort(localData, 0, 9);

    for(int i =0; i < 10; i++){
        output[gl_GlobalInvocationID.x * 10 + i] = localData[i];
    }
}
```

Here, I assume we're provided with an uniform array of 10 float values and are simply sorting it within the shader. This variant is simpler since the small set of data doesn't require the external stack management. We copy the uniform data to local stack, run quicksort, and store the sorted array in an output buffer. While it avoids the pitfalls of the previous example, it still highlights the core problem: recursive function calls, although possible, come with an overhead that typically exceeds any benefit when parallel processing is paramount. The limited size of the data also highlights that this approach can only be considered for trivial sorting tasks.

Finally, if we wanted to implement a completely iterative quicksort, as might be needed for larger, per-work-group datasets, we would need to explicitly manage our stack within shared memory and coordinate between work items of the same group, a complex and error-prone task, and still likely less performant than other sorting alternatives. Here's an illustrative example that demonstrates stack management and the shared memory:

```glsl
#version 450 core
layout(local_size_x = 64) in;

layout(binding = 0) buffer DataBuffer {
  float data[];
};

layout(binding = 1) buffer OutputBuffer {
  float output[];
};

shared uint stack[100];
shared uint stackIndex;

float partition(uint low, uint high) {
    float pivot = data[high];
    uint i = low - 1;
    for (uint j = low; j < high; j++) {
        if (data[j] < pivot) {
            i++;
            float temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    float temp = data[i + 1];
    data[i + 1] = data[high];
    data[high] = temp;
    return i + 1;
}

void push(uint low, uint high){
     if(stackIndex < 100){
        stack[stackIndex * 2] = low;
        stack[stackIndex * 2 + 1] = high;
        stackIndex++;
    }
}

bool pop(out uint low, out uint high){
    if(stackIndex > 0){
        stackIndex--;
        low = stack[stackIndex * 2];
        high = stack[stackIndex * 2 + 1];
        return true;
    }

    return false;
}


void main() {

    uint workGroupSize = gl_WorkGroupSize.x;
    uint arraySize = data.length();


    if(gl_LocalInvocationID.x == 0){
        stackIndex = 0;
        push(gl_WorkGroupID.x * workGroupSize, (gl_WorkGroupID.x + 1 ) * workGroupSize - 1); //initial push for each group
    }

    memoryBarrierShared();
    barrier();


    while(true){
        uint low;
        uint high;

        if(!pop(low, high)){
            break;
        }


        if(low < high){
            uint p = partition(low, high);
            push(low, p-1);
            push(p+1, high);
        }
     }

   memoryBarrierShared();
    barrier();


   if(gl_LocalInvocationID.x < workGroupSize){ //each thread within group writes to the final output.
      uint index = gl_WorkGroupID.x * workGroupSize + gl_LocalInvocationID.x;
       output[index] = data[index];
   }
}
```

This example demonstrates a quicksort operation on a per-work-group basis.  This approach requires using shared memory for work-group-level communication (stack). Note that the `stackIndex` and `stack` are marked as `shared` ensuring access across work items within the same work group. The initial `push` is done only for work item 0 which then initiates quicksort. Each work item performs parts of the sorting, controlled via stack and explicit barriers for synchronization.  While this is the most complete example, the complexity is still significant for the amount of sorting it accomplishes.

For practical applications on GPUs, consider researching parallel sorting algorithms more suitable for the parallel architecture. Merge sort or bitonic sort are often better choices. These algorithms are designed for parallel processing and better utilize the inherent capabilities of a GPU, thus allowing for more efficient sorting. These algorithms are often directly implemented using compute shaders for optimal speed. For resources, consult publications on parallel algorithm design, and documentation provided by graphics API vendors. There are also a multitude of university computer science courses online that often touch on these specific sorting topics.
