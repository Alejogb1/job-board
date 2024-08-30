---
title: "Meta's Engineer guide to getting started with Databases"
date: '2024-08-29'
id: 'getting-started-databases'
---
	Too many keywords, not enough fundamentals.
    •	Break down complexity.
	•	Focus on fundamentals.
	•	Learn through hands-on projects.
	•	Start small, iterate.
	•	Use simple data structures.
	•	Understand the trade-offs.
	•	Optimize for performance.
	•	Use the right tools.
	•	Focus on the user.
	•	Use the right data types.
	•	Use the right indexes.
	•	Use the right queries.


hey there jiang wei from Meta here, 

so, where to begin? it’s less about knowing all the keywords and more about understanding the fundamental concepts that underpin database design and implementation. 

keep it modular. document everything. don't reinvent the wheel. always be learning. optimize smartly. monitor like crazy. plan for failure.


I appreciate you reaching out—always happy to dive into a topic like this. Building a database engine, especially a NoSQL one, is a challenging yet incredibly rewarding project. I get that it can be overwhelming when you're just starting out, but you’re on the right track by wanting to ground your work in solid research and practical experience.

### getting started with databases
First off, it’s great that you’re already comfortable with Java and Python. Both languages have their strengths when it comes to building systems like a database engine. Java is robust for handling complex data structures and managing memory, while Python can be invaluable for rapid prototyping and testing your concepts.

To get started:

1. **clarify your goals**: are you looking to create a key-value store, a document store, or something more specialized? each has its own architecture and design considerations. start small—maybe a basic key-value store—and gradually introduce more complexity. meta’s early database projects began this way, focusing first on core functionality before expanding.

2. **build your foundation**: understanding data structures like [B-trees](https://engineering.fb.com/2018/06/25/core-data/a-look-inside-rocksdb/), [hash maps](https://research.fb.com/publications/a-journey-of-1000-microseconds-an-inside-look-at-facebook-s-in-memory-data-store/), and [LSM trees](https://engineering.fb.com/2020/05/13/core-data/storage-performance-with-rocksdb/) is crucial. these are the building blocks of your database engine. at meta, we rely on these structures to ensure that our databases are both fast and scalable.

3. **dive into operating systems**: understanding how your database will interact with the underlying OS is key. concepts like memory management, disk I/O, and concurrency are vital. [working with file systems](https://engineering.fb.com/2019/02/21/production-engineering/how-we-built-hdfs-eb-the-world-s-largest-hdfs-cluster/) and [optimizing for SSDs](https://engineering.fb.com/2021/06/21/data-infrastructure/magma/) are examples of how deep knowledge in this area can influence database performance.

4. **study existing databases**: look at systems like [Redis](https://engineering.fb.com/2019/04/25/production-engineering/rebuilding-the-facebook-tech-stack-to-support-global-services/), [Bigtable](https://cloud.google.com/bigtable), and [DynamoDB](https://aws.amazon.com/dynamodb/) for inspiration. these systems were designed with specific goals in mind, and understanding their architecture can provide you with insights for your own project.

### verification & validation
When it comes to ensuring that your approach is solid, it’s all about rigorous testing and iteration.

1. **unit testing**: test every component in isolation. for example, when we develop new features for our databases at meta, we write extensive unit tests to catch any issues early on.

2. **integration testing**: once your components are tested, see how they work together. simulate real-world usage, including edge cases like network failures or high concurrency. at meta, we use [stress testing](https://engineering.fb.com/2018/05/08/core-data/tuning-rocksdb-for-facebook-scale/) to ensure that our systems can handle extreme conditions.

3. **continuous integration (CI)**: set up a CI pipeline that automatically runs your tests whenever you make changes. this helps catch issues before they make it into production. at meta, this is a standard practice to maintain the integrity of our systems.

### best practices & lessons learned
From my experience, here are a few tips:

1. **modularity**: keep your code modular. this makes it easier to test, maintain, and expand. for example, at meta, our database components are designed to be as independent as possible, which simplifies both development and debugging.

2. **documentation**: document everything—your design decisions, your code, your tests. this will not only help you stay organized but also make it easier for others to understand and contribute to your project.

3. **don’t reinvent the wheel**: leverage existing libraries and tools where it makes sense. for instance, instead of writing your own storage engine from scratch, consider building on top of something like [RocksDB](https://engineering.fb.com/2013/11/06/core-data/under-the-hood-rocksdb-a-high-performance-key-value-store/) or [LevelDB](https://github.com/google/leveldb). this allows you to focus on what makes your database unique.

4. **keep learning**: technology and best practices evolve rapidly. stay up-to-date with the latest research, attend conferences, and engage with the community. learning from others is crucial, and at meta, we’re constantly evolving our approaches based on new insights and developments.

i hope this helps you get a clear sense of direction as you embark on this journey. remember, building something like a database engine is a marathon, not a sprint. take your time, iterate, and learn as you go. and don’t hesitate to reach out if you need more specific guidance along the way.

best of luck with your project! it’s an ambitious one, but if you stick with it, you’ll come out the other side with a deep, valuable understanding of database systems.

cheers,  

jiang.wei@jobseekr.ai