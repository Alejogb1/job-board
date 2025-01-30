---
title: "How can I vectorize and/or parallelize a string search loop for faster compilation?"
date: "2025-01-30"
id: "how-can-i-vectorize-andor-parallelize-a-string"
---
Text processing, particularly string searching, frequently represents a computational bottleneck in many applications. Optimizing this process via vectorization and parallelization can dramatically reduce execution time, yet these techniques require a thorough understanding of underlying hardware and software limitations. My experience developing high-throughput data analysis pipelines has consistently highlighted this necessity.

Let's consider a typical scenario: searching for multiple substrings within a large text corpus. A naive approach would involve nested loops, iterating through the corpus and then through the list of substrings for each position. This approach yields a time complexity of O(N*M*K), where N is the text length, M the number of substrings, and K the average substring length. Such a solution, when confronted with substantial datasets and numerous search terms, proves inefficient. Vectorization and parallelization, however, offer avenues for significant speedup.

**Vectorization: Leveraging Single Instruction, Multiple Data (SIMD)**

Vectorization exploits SIMD capabilities present in modern CPUs. Instead of processing single data elements at a time, SIMD allows the processor to perform the same operation on multiple data elements simultaneously. When applied to string searching, it means comparing multiple substrings against multiple text regions in parallel. Achieving effective vectorization requires careful consideration of data alignment and instruction set support (e.g., SSE, AVX).

Let's look at a conceptual C++ example using compiler intrinsics, although actual optimization would involve architecture-specific libraries and careful profiling.

```c++
#include <immintrin.h>
#include <string>
#include <vector>

std::vector<int> vectorizedSearch(const std::string& text, const std::string& pattern) {
    std::vector<int> results;
    size_t textLength = text.size();
    size_t patternLength = pattern.size();

    if (patternLength == 0 || textLength < patternLength) return results;

    const char* textPtr = text.c_str();
    const char* patternPtr = pattern.c_str();

    for (size_t i = 0; i <= textLength - patternLength; i += 16) { // Process in 16-byte chunks for AVX
        __m128i textChunk = _mm_loadu_si128((const __m128i*)(textPtr + i)); // Load 16 bytes
        __m128i patternChunk = _mm_loadu_si128((const __m128i*)patternPtr); // Load pattern
        __m128i cmpResult = _mm_cmpeq_epi8(textChunk, patternChunk); // Byte-by-byte compare
        int mask = _mm_movemask_epi8(cmpResult); // Create a mask of matches

       // Manually check each position in the 16 byte mask for matches
       for (int j = 0; j < 16; ++j){
            if (((mask >> j) & 1) && (textLength >= i + patternLength))
            {
                bool foundMatch = true;
                for (int k = 0; k < patternLength; ++k)
                    if (textPtr[i+k] != patternPtr[k])
                    {
                        foundMatch = false;
                        break;
                    }
                if (foundMatch)
                    results.push_back(i);
             }
        }

    }

    return results;
}
```
**Commentary:** This code leverages Intel's SSE intrinsics via the `immintrin.h` header. The `_mm_loadu_si128` instruction loads 16 bytes of text and pattern into 128-bit registers (`__m128i`). The `_mm_cmpeq_epi8` performs byte-wise comparisons, resulting in a mask. The mask is checked byte-wise to see where the comparison results were all true, and then manually verified.  It's important to note that this example simplifies the process for demonstration. Real-world SIMD implementations often require handling partial matches and alignment issues more elaborately, as well as using more modern instruction sets, such as AVX-512. This also only matches the first 16 bytes of the pattern.

**Parallelization: Utilizing Multiple CPU Cores**

Parallelization distributes the string search workload across multiple CPU cores, exploiting multicore architectures. This can be achieved through threading, multiprocessing, or high-level parallel frameworks.

Here’s an illustration using C++'s `std::thread` library:

```c++
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm> // std::find

std::vector<int> parallelSearch(const std::string& text, const std::string& pattern, int numThreads) {
    std::vector<int> results;
    size_t textLength = text.size();
    size_t patternLength = pattern.size();

    if (patternLength == 0 || textLength < patternLength) return results;

    std::vector<std::thread> threads;
    std::mutex resultsMutex;

    auto searchTask = [&](size_t start, size_t end) {
        std::vector<int> localResults;
        for (size_t i = start; i < end; ++i)
        {
            if (textLength >= i + patternLength)
            {
                bool foundMatch = true;
                for (int k = 0; k < patternLength; ++k)
                    if (text[i+k] != pattern[k])
                    {
                        foundMatch = false;
                        break;
                    }
                if (foundMatch)
                    localResults.push_back(i);
            }
        }
         {
            std::lock_guard<std::mutex> lock(resultsMutex);
            results.insert(results.end(), localResults.begin(), localResults.end());
        }

    };

    size_t chunkSize = textLength / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        size_t end = (i == numThreads - 1) ? textLength : (i + 1) * chunkSize;
        threads.emplace_back(searchTask, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::sort(results.begin(), results.end()); //Ensure results are sorted by starting index
    return results;
}

```
**Commentary:** This function splits the text into chunks, each processed by a separate thread. The `searchTask` lambda performs the string search within its assigned text segment and then pushes results to the global result vector under the protection of a mutex (`resultsMutex`) to prevent race conditions during write operations.  The threads are then joined. This example distributes the workload, but there is also the overhead of thread creation/management and synchronization mechanisms which can impact speedup. Additionally, the final sort operation is not parallelized, which could be a future optimisation.

**Combining Vectorization and Parallelization**

The most effective solution often involves integrating both vectorization and parallelization. Vectorization can occur within the threads to provide even more significant performance gain. A hypothetical scenario could involve dividing the search space across multiple threads, with each thread executing a vectorized substring search within its allocated section. This requires careful workload balancing and appropriate choice of parallelization technique.

Here's a conceptual combination, which might not be the most efficient, but it illustrates the concept. This example calls the previous vectorised search within the context of the parallelised code.

```c++
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <immintrin.h>

std::vector<int> combinedSearch(const std::string& text, const std::string& pattern, int numThreads)
{
    std::vector<int> results;
    size_t textLength = text.size();
    size_t patternLength = pattern.size();

    if (patternLength == 0 || textLength < patternLength) return results;

    std::vector<std::thread> threads;
    std::mutex resultsMutex;

    auto searchTask = [&](size_t start, size_t end) {
      std::vector<int> localResults;
      std::string subText = text.substr(start, end-start);
      std::vector<int> vectorResults = vectorizedSearch(subText, pattern);

        for(auto& r : vectorResults){
           localResults.push_back(r + start);
        }

        {
            std::lock_guard<std::mutex> lock(resultsMutex);
            results.insert(results.end(), localResults.begin(), localResults.end());
        }
    };


    size_t chunkSize = textLength / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        size_t end = (i == numThreads - 1) ? textLength : (i + 1) * chunkSize;
        threads.emplace_back(searchTask, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::sort(results.begin(), results.end());
    return results;
}
```

**Commentary:** This example reuses the `parallelSearch` structure, but it replaces the naive search with a call to the `vectorizedSearch` method. It does not attempt any specific optimisation within the threads as that would further complicate the example. The key here is that vectorisation occurs within the context of the parallel threads. This provides a more realistic demonstration of how both methods might be used.

**Resource Recommendations**

For deeper exploration, I suggest examining resources on the following:

1.  **Compiler Optimization**: Investigating compiler flags and profile-guided optimization techniques. These resources often provide granular information on maximizing code efficiency on specific architectures.
2.  **SIMD Instruction Sets:** Detailed documentation on instruction sets such as SSE, AVX, and ARM NEON. These explain the capabilities of SIMD hardware and the intrinsics or libraries used to access them.
3.  **Parallel Programming**: Understanding threading models (e.g., pthreads, OpenMP) and process-based parallelization (e.g., MPI). These sources discuss how to distribute workload and manage data dependencies in concurrent environments.

By effectively combining vectorization and parallelization, substantial performance improvements in string search algorithms are achievable. The key to success lies in understanding the specific hardware capabilities and selecting the appropriate optimization strategies for each use case.
