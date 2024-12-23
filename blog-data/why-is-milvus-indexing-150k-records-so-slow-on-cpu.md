---
title: "Why is Milvus indexing 150k records so slow on CPU?"
date: "2024-12-23"
id: "why-is-milvus-indexing-150k-records-so-slow-on-cpu"
---

Alright, let’s tackle this. I’ve seen this particular issue crop up more times than I care to remember, and it almost always boils down to a few core culprits, particularly when we’re talking about CPU-based indexing in Milvus. The seemingly simple task of indexing 150,000 records can become surprisingly slow, and understanding the bottlenecks is crucial for optimal performance. Let's unpack this, drawing from some past experiences I've had with similar setups.

First off, when Milvus indexes data, especially on CPU, it's essentially converting your high-dimensional vectors into a searchable structure. This conversion involves intricate calculations and data manipulations. These operations, while seemingly straightforward conceptually, are computationally expensive and become painfully slow when the computational resources are insufficient. When the system is bogged down with other tasks, or when the allocated CPU cores are not used optimally, the entire process can bog down.

One of the primary issues I often see with CPU-based indexing revolves around the *type of index* being used. Milvus offers a variety of indexes (like flat, ivf_flat, ivf_pq, etc.), each with their own trade-offs between indexing speed, search speed, and memory usage. If you’re indexing on the CPU and chose a less efficient algorithm for the situation, such as an ‘IVF’ based index with a large nlist, then indexing will be considerably slower. Remember, `flat` index is simple but has a linear time complexity in relation to the vector number. For larger datasets, that can lead to significant slowdowns. When I was working on a large-scale recommendation system a couple of years back, we were mistakenly using `flat` initially, and the indexing times were unacceptable. We shifted to an IVF-based index, and things dramatically improved.

Here's a simple example that illustrates how to specify the index type when creating a collection in Milvus using Python, which helps steer the processing in a more performant direction:

```python
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema, connections, utility

connections.connect(host='localhost', port='19530')

# Example dimensions
dim = 128

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields=fields, description="Example Collection")
collection_name = "example_collection"

if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)


# Create Index with appropriate parameters
index_param = {
   "index_type": "IVF_FLAT",
   "params": {"nlist": 1024},  # Adjust nlist based on your dataset
   "metric_type": "L2" # Or "IP" for inner product based distance
}

# Create Index
if not utility.has_index(collection_name):
    collection.create_index(field_name="vector", index_params=index_param)
```

In this snippet, we're creating an IVF_FLAT index which typically provides a better balance of speed and accuracy when compared to ‘flat’ for large datasets, especially on CPU-based systems. `nlist` controls the level of granularity for partitioning the data space. This setting needs to be tailored to the size and characteristics of the data for optimal performance. Too few lists might speed up the indexing but slow searches. Too many could bog indexing down without significantly boosting the search performance.

Beyond index selection, the *number of threads* used for the indexing process plays a vital role. Milvus relies on the underlying BLAS (Basic Linear Algebra Subprograms) library for vector operations, which is typically optimized for multi-core processors. However, if this isn't correctly configured, you could be limiting yourself to a single thread. There are environment variables that control the number of threads used by these libraries. I once ran into a problem where the `OMP_NUM_THREADS` was set incorrectly on a server running Milvus, causing everything to be processed using just one core, even though the server had many available. It's worth verifying your specific configurations.

Here’s a bit of a Python snippet that shows how to manipulate environment variables that influence the threading:

```python
import os

# Set OMP_NUM_THREADS to a reasonable value (adjust based on your system)
num_threads = os.cpu_count() // 2 # Using half the available cores is a common starting point, change this according to your specific needs.
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)


print(f"OMP_NUM_THREADS set to: {os.environ.get('OMP_NUM_THREADS')}")
print(f"OPENBLAS_NUM_THREADS set to: {os.environ.get('OPENBLAS_NUM_THREADS')}")
print(f"MKL_NUM_THREADS set to: {os.environ.get('MKL_NUM_THREADS')}")

# Continue with Milvus operations...
from pymilvus import Collection, connections
connections.connect(host='localhost', port='19530')
#....
```

This snippet shows explicitly setting the `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` environment variables before any Milvus operations. These variables influence how the underlying libraries manage threading, directly affecting indexing performance. Experiment with the value here, but starting with half of your cores is generally a solid starting point. The correct value usually requires some experimentation, often by monitoring cpu load to see if you are being bottlenecked by underutilizing or overutilizing cpu.

Finally, and this might seem obvious, but the *hardware resources* themselves can be the culprit. A server with an older CPU, insufficient ram, or slow storage will struggle to keep pace with indexing a large number of vectors efficiently. If a significant amount of swapping is occuring due to a lack of RAM, the indexing can be drastically slow. Similarly, if the vectors are read from a slow disk or network storage, the overall process will take a lot longer. The server should have reasonable processing power, and the storage should be fast (preferably SSD). We had to upgrade a development environment once where the underlying hardware just could not keep up. I’ve found that monitoring the CPU and memory usage during the indexing process can clearly demonstrate where the bottleneck exists.

And one more thing, be aware of vector *dimensionality* - very high-dimensional vectors are significantly harder to index. Lowering the dimension via dimensionality reduction techniques might help, but that is not always an option.

Below is a snippet showing how to do basic performance monitoring using Python:

```python
import time
import psutil

def monitor_cpu_mem(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        initial_cpu_percent = psutil.cpu_percent()
        initial_memory_usage = psutil.virtual_memory().percent
        result = function(*args, **kwargs)
        end_time = time.time()
        final_cpu_percent = psutil.cpu_percent()
        final_memory_usage = psutil.virtual_memory().percent
        
        cpu_change = final_cpu_percent - initial_cpu_percent
        mem_change = final_memory_usage - initial_memory_usage

        print(f"Function '{function.__name__}' took {end_time - start_time:.2f} seconds.")
        print(f"CPU usage change: {cpu_change:.2f}%")
        print(f"Memory usage change: {mem_change:.2f}%")
        return result
    return wrapper


# Applying the decorator to the insert function
from pymilvus import Collection, connections
connections.connect(host='localhost', port='19530')

collection = Collection(name="example_collection")

@monitor_cpu_mem
def insert_data(collection, vectors, ids):
    collection.insert(data=[ids,vectors])

# Assume vectors and ids are previously defined and data is ready to be indexed
# The insert function will now have performance information printed
# insert_data(collection, vectors, ids)
```

This snippet illustrates using the `psutil` library to monitor CPU and memory usage before and after a function, giving an idea of the system resources used by that function (in this case, it's the insert operation for the purposes of demonstration).

In summary, slow CPU-based indexing in Milvus usually comes down to sub-optimal index selection, improperly configured threading, insufficient hardware resources or too large dimensionality. Before blaming Milvus, it's best to check these fundamentals.

For further reading, I would recommend reviewing the Milvus documentation thoroughly, especially the sections on index types and performance tuning. The “Milvus Operations Guide” is usually the best place to start. Additionally, getting a deeper understanding of BLAS libraries and how threading can be leveraged for numerical computation would also be beneficial. OpenBLAS and Intel MKL have lots of documentation online about threading controls. Lastly, understanding the performance characteristics of various algorithms in vector databases is critical. A book like “Mining of Massive Datasets” by Jure Leskovec, Anand Rajaraman, and Jeff Ullman could offer insights into this.
