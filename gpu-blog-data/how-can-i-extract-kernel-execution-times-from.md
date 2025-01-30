---
title: "How can I extract kernel execution times from Nsight Systems' SQLite3 output?"
date: "2025-01-30"
id: "how-can-i-extract-kernel-execution-times-from"
---
Nsight Systems' SQLite3 output, while comprehensive, doesn't directly expose kernel execution times in a readily accessible format.  My experience analyzing performance data from large-scale HPC simulations taught me that extracting this information necessitates a multi-stage approach involving data manipulation and careful interpretation of the profiling data.  Kernel execution times, in this context, refer to the duration individual kernel launches spend on the GPU, excluding data transfer overhead.  This distinction is crucial for accurate performance analysis.


The core challenge lies in the fact that Nsight Systems' primary focus is on high-level application profiling. While it captures GPU activity, the granularity of kernel execution time reporting depends on the chosen profiling level and instrumentation.  Simple sampling might not provide the precise timing needed, requiring a deeper dive into the underlying data.  Therefore, extracting this data requires careful SQL queries targeting relevant tables within the SQLite3 database and potentially subsequent data processing.


**1.  Data Extraction and Preprocessing:**

The first step involves querying the relevant tables within the Nsight Systems SQLite3 database.  The specific table names might vary slightly depending on the Nsight Systems version, but generally, you'll need to focus on tables containing GPU activity details.  Tables related to kernel launches and their corresponding timestamps are key.  I’ve found that directly joining these tables often yields the most useful results.


A crucial element often overlooked is the need for data filtering.  Many profiling sessions capture data far beyond the area of interest.  Filtering by process ID, thread ID, or even specific kernel names is essential for isolating the relevant kernel execution data.  This substantially reduces the dataset size and streamlines subsequent processing.


**2. Code Examples:**

The following examples assume a basic understanding of SQL and the Nsight Systems database schema. I’ll present three examples demonstrating progressive complexity:


**Example 1: Basic Kernel Launch Extraction (Assuming a table named `gpu_kernel_launches`)**

```sql
SELECT
    start_time,
    end_time,
    kernel_name,
    (JULIANDAY(end_time) - JULIANDAY(start_time)) * 86400 AS execution_time_seconds
FROM
    gpu_kernel_launches
WHERE
    process_id = 1234  -- Replace with your process ID
ORDER BY
    start_time;
```

This query selects the start and end timestamps, kernel name, and calculates the execution time in seconds using the `JULIANDAY` function for accurate timestamp subtraction. The `WHERE` clause filters results to a specific process ID, improving efficiency.  Note that the assumption of a table named `gpu_kernel_launches` might need adjustment according to your specific Nsight Systems version.


**Example 2: Incorporating CUDA Stream Information (Assuming tables `gpu_kernel_launches` and `cuda_streams`)**

```sql
SELECT
    k.start_time,
    k.end_time,
    k.kernel_name,
    (JULIANDAY(k.end_time) - JULIANDAY(k.start_time)) * 86400 AS execution_time_seconds,
    s.stream_id
FROM
    gpu_kernel_launches k
INNER JOIN
    cuda_streams s ON k.stream_id = s.stream_id
WHERE
    k.process_id = 1234 -- Replace with your process ID
ORDER BY
    k.start_time, s.stream_id;
```

This query builds on the previous example by joining the `gpu_kernel_launches` table with a hypothetical `cuda_streams` table, adding stream ID information. This is useful for understanding kernel execution within specific CUDA streams, offering finer-grained analysis.  The accuracy depends on the level of detail Nsight Systems captures for stream management.


**Example 3:  Advanced Filtering and Aggregation (Assuming tables `gpu_kernel_launches` and `kernel_metadata`)**

```sql
SELECT
    km.kernel_name,
    AVG((JULIANDAY(k.end_time) - JULIANDAY(k.start_time)) * 86400) AS average_execution_time
FROM
    gpu_kernel_launches k
INNER JOIN
    kernel_metadata km ON k.kernel_id = km.kernel_id
WHERE
    k.process_id = 1234 -- Replace with your process ID
    AND km.kernel_name LIKE '%my_kernel%' --Example filtering
GROUP BY
    km.kernel_name
ORDER BY
    average_execution_time DESC;
```

This advanced query demonstrates filtering based on kernel metadata (e.g., kernel name) and aggregation using `AVG` to compute the average execution time for specific kernels.  This approach is especially valuable when dealing with numerous kernel launches and helps identify performance bottlenecks.  The `kernel_metadata` table is hypothetical and reflects the need to potentially join with additional tables to access kernel-specific information.



**3.  Post-Processing and Interpretation:**

The SQL queries above generate data representing kernel execution times. However, these results often require further processing.  I've frequently used scripting languages like Python with libraries like Pandas to perform tasks such as data cleaning, statistical analysis (e.g., calculating percentiles, standard deviations), and visualization. This allows for a deeper understanding of kernel performance characteristics beyond simple average execution times.

Furthermore, it's essential to consider the potential impact of factors like GPU occupancy, memory bandwidth limitations, and shared memory usage.  These factors influence kernel execution times and should be considered during performance analysis.  Correlating the extracted kernel times with other Nsight Systems metrics related to these factors is crucial for a complete performance evaluation.


**4. Resource Recommendations:**

Consult the Nsight Systems documentation for detailed information on the database schema and available tables. Familiarize yourself with SQL query writing and optimization techniques.  Master a scripting language like Python with data manipulation libraries (e.g., Pandas, NumPy) for efficient data processing and analysis.  Understanding CUDA programming concepts enhances interpretation of the results, especially when dealing with stream management and memory access patterns.  Explore statistical analysis techniques to gain insights from aggregated data.


By following this multi-stage process, combining SQL queries with post-processing techniques, and considering the broader performance context, you can effectively extract and analyze kernel execution times from Nsight Systems' SQLite3 output for thorough performance optimization.  Remember that the specific table names and query structures might need adjustment to reflect the particular version of Nsight Systems you are using.  Always consult the relevant documentation to verify the schema.
