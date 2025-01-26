---
title: "How can I parallelize pure functions?"
date: "2025-01-26"
id: "how-can-i-parallelize-pure-functions"
---

The inherent characteristic of pure functions—their lack of side effects and consistent output for given inputs—makes them exceptionally well-suited for parallel execution. This capability stems directly from their referential transparency; because a pure function's evaluation depends solely on its arguments and it does not alter any external state, invocations with identical arguments can be executed independently and concurrently without causing race conditions or producing inconsistent results. My past experience optimizing numerical simulations involving intensive calculations on large datasets solidified this understanding.

Parallelizing pure functions effectively leverages multiple cores or processors to achieve faster execution times, particularly for computationally demanding tasks. The strategy involves distributing independent invocations of the pure function across available processing units and subsequently combining their results. The key is ensuring that the mechanism used for parallelization correctly manages the splitting of input data, launching of concurrent function calls, and collation of outputs without introducing any external state dependencies or violating the inherent purity of the computation.

I've found that several techniques facilitate the parallelization of pure functions, each with its trade-offs regarding complexity and efficiency. Broadly, they can be categorized into data parallelism and task parallelism, although frequently the chosen approach will integrate aspects of both.

**Data Parallelism**

Data parallelism is applicable when the same operation needs to be applied to numerous data elements, or when a large dataset can be partitioned into smaller subsets. Each subset is then processed by an independent execution of the pure function, and the results are combined. This model is ideal for computations like map or reduce operations on lists or arrays.

*   **Example 1: Parallel Summation of a List**

    The following Python code demonstrates how to use the `multiprocessing` module to sum a list of numbers in parallel. The `square` function, which is pure, is used to pre-process the numbers.

    ```python
    import multiprocessing

    def square(x):
        return x * x

    def parallel_sum(data):
        with multiprocessing.Pool() as pool:
            squared_data = pool.map(square, data)
            return sum(squared_data)


    if __name__ == '__main__':
        numbers = list(range(10000))
        total = parallel_sum(numbers)
        print(f"The sum of squares is: {total}")
    ```

    In this example, `square` is a pure function that squares its input. The `parallel_sum` function creates a `Pool` of worker processes to distribute the squaring operation across the input list. `pool.map` applies the `square` function to every element in the list in parallel and then returns the result as a new list, which is subsequently summed. The use of `multiprocessing` creates a separate process for each core, providing true parallelism. The `if __name__ == '__main__':` ensures that the multiprocessing code is only executed in the main process to prevent issues on platforms such as Windows. This approach utilizes data parallelism effectively: each call to `square` is independent and operates on a fraction of the overall data.

*   **Example 2: Parallel Processing Using Joblib**

    The `joblib` library offers an alternative way to achieve parallel processing, particularly well-suited for scientific and machine learning applications. In my prior projects, I frequently used `joblib` for efficiently processing large arrays.

    ```python
    from joblib import Parallel, delayed
    import numpy as np

    def scale(x, factor):
        return x * factor

    def parallel_scale_array(data, factor):
        scaled_data = Parallel(n_jobs=-1)(delayed(scale)(d, factor) for d in data)
        return np.array(scaled_data)

    if __name__ == '__main__':
        data_array = np.random.rand(10000)
        scaling_factor = 2.5
        scaled_array = parallel_scale_array(data_array, scaling_factor)
        print(f"First 5 scaled values: {scaled_array[:5]}")
    ```

     Here, `scale` is a pure function multiplying input `x` by a factor. The `parallel_scale_array` utilizes `joblib`'s `Parallel` function, which distributes the application of `scale` across multiple cores. The `delayed` decorator freezes the function calls until all data is prepared for the execution. `n_jobs=-1` specifies the usage of all available CPU cores. The `joblib` library often performs better in terms of efficiency and memory usage compared to Python's native multiprocessing, particularly for large data. Note, the return is an numpy array, as joblib preserves the type when it can, and returns a list otherwise.

**Task Parallelism**

Task parallelism is useful when different operations need to be performed concurrently, often on different sets of data. This model is often associated with workflows or pipelines where the output of one pure function serves as the input of another.

*   **Example 3: Parallel File Processing**

    This example demonstrates a very simplified process for reading and processing data in parallel using Python's `concurrent.futures` library. In my work with distributed systems, I employed similar task-based parallelism patterns extensively.

    ```python
    import concurrent.futures
    import time

    def read_file(filename):
      time.sleep(0.5)  # Simulate time taken to read file
      return f"Data from {filename}"

    def process_data(data):
        time.sleep(0.2) # Simulate processing of data
        return f"Processed: {data}"


    def parallel_file_processing(filenames):
      with concurrent.futures.ThreadPoolExecutor() as executor:
          read_futures = [executor.submit(read_file, filename) for filename in filenames]
          processed_futures = [executor.submit(process_data, future.result()) for future in read_futures]
          return [future.result() for future in processed_futures]

    if __name__ == '__main__':
        filenames = [f"file_{i}.txt" for i in range(5)]
        results = parallel_file_processing(filenames)
        for result in results:
            print(result)
    ```

    The example defines two pure functions: `read_file`, which simulates reading from a file, and `process_data`, which simulates data processing. The `parallel_file_processing` function uses a `ThreadPoolExecutor` to execute each task concurrently. First, it submits calls to `read_file` for all filenames. Then, it submits calls to `process_data`, each taking the output of a `read_file` call as its input, thus exhibiting a task-based pipeline. The use of `concurrent.futures` is more memory-efficient than using `multiprocessing` if the tasks are I/O bound. In this scenario, we achieve parallelism through distributing independent operations across threads, all without violating the purity of the underlying `read_file` and `process_data` functions.

**Resource Recommendations**

For deeper understanding and more advanced techniques concerning parallel computing, I would recommend the following texts and libraries:

1.  **Programming Massively Parallel Processors: A Hands-on Approach:** This book, by David B. Kirk and Wen-mei W. Hwu, provides in-depth coverage of GPU programming using CUDA, which is useful for highly parallelizable tasks.

2.  **High Performance Python:** By Micha Gorelick and Ian Ozsvald, this work focuses specifically on optimization strategies for Python programs, which includes various approaches to parallel computing such as using `multiprocessing`, `threading`, and `asyncio`.

3.  **The documentation for libraries like `multiprocessing`, `concurrent.futures`, and `joblib`**: These resources offer practical insight into the usage and capabilities of the respective libraries.

4. **The Official Python documentation**: The official documentation provides clear insights into how Python handles multiprocessing and the differences between processes and threads.

In conclusion, the nature of pure functions enables straightforward parallelization, which significantly accelerates computations in many use cases. When implementing parallel strategies, one should carefully consider the nature of the problem (data or task parallel), the available resources, and the overhead of the parallel execution itself. The selection of the most appropriate parallelization mechanism depends on the specific characteristics of the pure functions and the problem domain. The code examples provided demonstrate how one might begin to implement data and task parallelism using readily available Python libraries.
