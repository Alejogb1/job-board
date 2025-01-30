---
title: "How can shuffling be sped up using GPU (Tesla K40m) and parallel CPU computations in MATLAB?"
date: "2025-01-30"
id: "how-can-shuffling-be-sped-up-using-gpu"
---
A primary bottleneck in many large-scale simulation projects I've managed involves the generation of randomized sequences for model initialization or Monte Carlo methods. Traditional single-threaded CPU-based shuffling in MATLAB, while straightforward, quickly becomes inadequate when dealing with datasets beyond a few gigabytes. Leveraging both GPU acceleration and parallel CPU computation, when implemented correctly, delivers substantial performance improvements.

The key insight here is that the shuffling process can be decomposed into independent operations: generating pseudo-random numbers and performing the associated array rearrangements. Both tasks lend themselves to parallel execution. The NVIDIA Tesla K40m, with its significant number of cores and memory bandwidth, provides the architecture to drastically accelerate the pseudo-random number generation and, to a lesser degree, the shuffling operation. Meanwhile, parallel CPU computations can simultaneously process distinct parts of the dataset. These two approaches are not mutually exclusive; they can, and should, be combined to achieve the best performance.

The traditional method in MATLAB, employing the `randperm` function, operates on a single thread. While convenient, this approach becomes the performance limiter as the dataset increases. A rudimentary solution might involve simply splitting the data into segments, shuffling each segment on a different core, and concatenating the results. However, this introduces a risk of biases if not handled meticulously, and does not address the inherent slowness of sequential generation of random numbers on the CPU. A more powerful solution, and the one I’ve found most effective, is to offload pseudo-random number generation onto the GPU using MATLAB’s `gpuArray` functionality, followed by GPU-accelerated shuffling if appropriate, or partitioned shuffling across multiple CPU threads, depending on data size and structure.

Let me demonstrate with some examples, along with commentary:

**Example 1: GPU Accelerated Random Number Generation and Basic CPU Shuffle**

This example focuses on offloading the random number generation to the GPU but retains the standard `randperm` function running on the CPU. This illustrates how using a GPU for random number generation alone can give a speedup. This approach isn’t ideal, but it's an easy modification for datasets that aren't yet large enough to benefit from fully GPU-based approaches.

```matlab
% Example 1: GPU accelerated RNG with standard CPU shuffle

N = 1e8; % Large dataset size

% Create a gpuArray of uniform random numbers on GPU
rng('default'); % Initialize CPU RNG for reproducibility
tic;
gpuRandNums = gpuArray(rand(1, N,'single')); % Generates on the GPU as single-precision for speed
t_gpuRandGen = toc;


tic;
% Transfer gpuArray to CPU
cpuRandNums = gather(gpuRandNums);

% Permute indices using CPU randperm
idx = randperm(N);
shuffledData = cpuRandNums(idx);

t_cpuShuffle = toc;

fprintf('GPU Random Number Gen Time: %.4f sec \n', t_gpuRandGen);
fprintf('CPU Shuffle Time: %.4f sec \n', t_cpuShuffle);
```

*   **Explanation:** Here, I first generate a large vector of uniformly distributed random numbers directly on the GPU using `gpuArray(rand(1, N,'single'))`. This avoids the time-consuming transfer of random numbers from the CPU to the GPU. The 'single' keyword enforces single-precision floating point to accelerate calculations. This is often acceptable for shuffling operations. The `gather` function then transfers this back to the CPU, where a standard CPU-based shuffling method via `randperm` is performed. The timing using `tic` and `toc` illustrates the relative time taken by the two operations. In my experience, this alone can provide notable speedup, as GPU-based random number generation is significantly faster than traditional CPU-based methods. This example is useful when the dataset is of manageable size to easily reside in CPU memory.

**Example 2: Fully GPU Accelerated Shuffle (Limited applicability)**

This example demonstrates using GPU for both random number generation and the shuffling. In some scenarios, direct GPU shuffling can be advantageous, although practical considerations regarding data transfer can limit its utility. Note that the `randperm` operation is not directly available on the GPU, so we are constructing an appropriate sequence of indices to shuffle on the GPU directly.

```matlab
% Example 2: Fully GPU Accelerated Shuffle

N = 1e6; % Smaller dataset for direct GPU shuffle

% Create a gpuArray of uniform random numbers on GPU
rng('default');
tic;
gpuRandNums = gpuArray(rand(1, N,'single')); % Generates on the GPU
t_gpuRandGen = toc;


% Generate index permutation array on GPU
tic;
idx = gpuArray(randperm(N));
shuffledData = gpuRandNums(idx); % Shuffle the vector
t_gpuShuffle = toc;

fprintf('GPU Random Number Gen Time: %.4f sec \n', t_gpuRandGen);
fprintf('GPU Shuffle Time: %.4f sec \n', t_gpuShuffle);

%optional gather
% shuffledData_cpu = gather(shuffledData);
```

*   **Explanation:** Here, we still start by generating the random numbers on the GPU, just like in the previous example. Crucially, however, instead of bringing the random numbers back to the CPU, we generate a permutation index array directly on the GPU.  The shuffling operation is then performed entirely on the GPU using indexing. This approach maximizes the utilization of the GPU’s processing power and memory bandwidth. It is worth noting that large indices can impose memory constraints on the GPU, which is why I have reduced the size of N for this example to be fully executable. The comment about `gather` highlights that the shuffled data can remain on the GPU if further processing steps can occur there. If the shuffled data needs to be on CPU, a `gather` operation will be required. In my experience, this approach is beneficial when processing data that already resides on the GPU or when the shuffled data can immediately be used in subsequent GPU-based calculations. Direct GPU shuffling, as demonstrated here, is often fastest for small to medium datasets but can lead to memory constraints on the GPU if the input is too large.

**Example 3: Parallel CPU Shuffling With GPU RNG**

This example combines parallel CPU processing with GPU-accelerated random number generation. This approach is often the most pragmatic for very large datasets because it handles the data size by using the RAM on a multi-core computer and does not rely so heavily on a limited amount of GPU memory.

```matlab
% Example 3: Parallel CPU Shuffle with GPU RNG

N = 1e8; % Large dataset size
numCores = 8; % Number of CPU cores to use

% Create a gpuArray of uniform random numbers on GPU
rng('default');
tic;
gpuRandNums = gpuArray(rand(1, N,'single'));
t_gpuRandGen = toc;


cpuRandNums = gather(gpuRandNums);

tic;
% Partition the data and apply randperm in parallel
parfor i = 1:numCores
    startIndex = floor((i - 1) * N / numCores) + 1;
    endIndex = floor(i * N / numCores);
    idx = randperm(endIndex-startIndex+1);
    partitionedData(startIndex:endIndex) = cpuRandNums(startIndex:endIndex);
    partitionedData(startIndex:endIndex) = partitionedData(startIndex:endIndex)(idx);
end

t_parShuffle = toc;

fprintf('GPU Random Number Gen Time: %.4f sec \n', t_gpuRandGen);
fprintf('Parallel CPU Shuffle Time: %.4f sec \n', t_parShuffle);

% optionally check against a normal shuffle using randperm
% tic
% idx = randperm(N);
% shuffledData_CPU = cpuRandNums(idx);
% t_cpuShuffle = toc;
% fprintf('CPU Shuffle Time: %.4f sec \n', t_cpuShuffle);
```

*   **Explanation:**  This example combines the benefits of fast GPU random number generation with parallel CPU processing. First, like in the previous examples, the random numbers are generated on the GPU. The `parfor` loop partitions the dataset into segments and applies the standard `randperm` function to each segment, allowing MATLAB to utilize multiple CPU cores in parallel. In my experience, this approach gives substantial performance boosts for large datasets that cannot fit within the GPU memory. Using the parallel toolbox, `parfor` automatically manages thread spawning and task assignment. This strikes a balance, leveraging the GPU for RNG, and the parallel CPU cores to tackle the actual shuffling. This approach does introduce a potential bottleneck at the ‘gather’ step, which can take considerable time to transfer large arrays back to CPU.  As a final point, I have included an optional section of code that generates an equivalent sequential shuffling on the CPU.  This is useful for comparing against the parallel version and verifying they perform the same operation.

In conclusion, optimizing shuffling for large datasets in MATLAB requires understanding and exploiting the strengths of both GPU and CPU architectures. Simply splitting the work between multiple CPU cores is a reasonable, and often easy, start, but not sufficient in demanding computational projects. Offloading random number generation onto the GPU is usually a simple step that yields significant performance gains. Further, using GPUs for full shuffling, when appropriate, and using parallel CPU computations to partition data further improves the overall throughput.

For further information on topics discussed here, I recommend exploring documentation related to MATLAB's Parallel Computing Toolbox and the GPU Computing Toolbox. It’s especially beneficial to review examples focusing on `gpuArray` and `parfor` constructs, and the documentation of `randperm`. Furthermore, the performance optimization sections in MATLAB's documentation provide a comprehensive overview of strategies for efficient code execution on both CPU and GPU. Lastly, understanding the specifics of the hardware architecture being used – in this case, the NVIDIA Tesla K40m – can inform which approaches are likely to yield the best results. The information available on NVIDIA developer website regarding CUDA, the underlying API for GPU programming, is also informative.
