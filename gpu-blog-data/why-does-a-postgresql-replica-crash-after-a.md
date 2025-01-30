---
title: "Why does a PostgreSQL replica crash after a successful failover?"
date: "2025-01-30"
id: "why-does-a-postgresql-replica-crash-after-a"
---
PostgreSQL replica crashes following a successful failover often stem from inconsistencies between the promoted replica and its prior state as a standby.  My experience troubleshooting this across numerous high-availability deployments points to several key causes, primarily related to WAL (Write-Ahead Log) replay, shared memory configuration, and potential data corruption.  Let's examine these in detail.


**1. Incomplete WAL Replay:**

The most frequent culprit is incomplete Write-Ahead Log replay during the promotion process.  When a primary server fails, the replica is promoted, inheriting the role of primary. However, if the failover mechanism didn't ensure complete WAL replay up to the point of the primary's failure, the new primary might possess an inconsistent data state. This inconsistency manifests as data loss or corruption, ultimately triggering a crash.  The critical factor here is the synchronization point between the primary and the standby. If the standby was significantly behind, even by a few transactions, promoting it directly can lead to instability.  Furthermore, network interruptions during the failover process can exacerbate the problem, hindering the complete download and application of necessary WAL segments.

**2. Shared Memory Configuration Mismatch:**

PostgreSQL's performance heavily relies on shared memory segments for efficient inter-process communication. During promotion, the replica inherits configuration settings designed for a standby role, often with reduced shared memory allocations. If these settings aren't dynamically adjusted upon promotion to reflect the increased workload of a primary server, resource contention can occur, leading to instability and crashes. This is particularly noticeable under heavy write loads, as the increased demand exceeds the available shared memory resources, causing the database processes to struggle and eventually fail.  I've seen this issue several times, especially in deployments lacking automatic configuration adjustments during the failover sequence.

**3. Underlying Data Corruption:**

While less common, pre-existing data corruption on the replica itself can be triggered during the promotion process. This corruption might have been present before the failover but remained latent until the increased workload of a primary server exposed it.  Such corruption can manifest in various forms, from subtle inconsistencies in index structures to outright data page corruption.  The stress of handling transactions and concurrently maintaining data integrity often amplifies the effects of latent corruption, resulting in a crash.  Thorough database integrity checks on the replica before deploying it into a high-availability setup are crucial in mitigating this risk.  I recall a specific case where a hardware failure on the storage device hosting the replica subtly corrupted a few data pages, only revealing the damage after promotion.


**Code Examples and Commentary:**

Below are three illustrative code examples showcasing potential scenarios and debugging approaches, all employing `psql` and assuming basic familiarity with SQL and PostgreSQL system tables.  Note these are simplified for illustrative purposes.


**Example 1: Checking WAL Replay Status:**

```sql
-- Check the last replayed WAL segment on the promoted replica
SELECT pg_last_wal_receive_lsn();
-- Compare this LSN with the primary's WAL position at the time of failover.
-- Significant discrepancy indicates incomplete WAL replay.  You would obtain the
-- primary's last WAL position from monitoring tools or logs.
```

This query retrieves the last WAL segment replayed on the promoted replica. Comparing this with the primary server's position at the failover moment is essential. A substantial difference points to incomplete replication, a major contributor to post-failover crashes.  The specifics of obtaining the primary's last WAL position depend on your monitoring setup (e.g., pg_stat_replication, dedicated monitoring tools).


**Example 2: Investigating Shared Memory Settings:**

```sql
-- Check shared memory settings on the promoted replica.
SHOW shared_buffers;
SHOW work_mem;
-- Compare these with recommended settings for a server of its size and workload.
-- Insufficient values can lead to performance issues and crashes under load.
```

This shows the current shared memory settings.  These values should be adequately sized for the anticipated workload.  Underestimation often manifests as performance degradation and eventually crashes. You should compare these values with best-practice guidelines based on the server's RAM and typical transaction volume.  Insufficient `shared_buffers` and `work_mem` can be particularly problematic.


**Example 3: Basic Data Integrity Check (Partial):**

```sql
-- Simple check for index inconsistencies –  adapt to your specific tables.
SELECT COUNT(*) FROM your_table;
SELECT COUNT(*) FROM your_table WHERE your_index_column IS NOT NULL;  --Check against index

-- A significant difference indicates potential index corruption.
-- More comprehensive checks involve using pg_checksums extensions and
-- pg_repack.  This is just a starting point.  This should be done with backups
-- readily available.
```

This demonstrates a rudimentary check for index inconsistencies.  While a simple check, a substantial difference between the row counts indicates a possible corruption. More rigorous checks, such as utilizing `pg_checksums` (to check data page checksums) and `pg_repack` (to rebuild potentially corrupt indexes) should be conducted, but only after creating backups.  This example only provides a minimal illustration and shouldn't replace comprehensive data integrity validation.


**Resource Recommendations:**

Consult the official PostgreSQL documentation.  Explore the documentation related to high-availability configurations, particularly regarding WAL replication and failover procedures.  Review best-practice guides for configuring shared memory settings. Understand the different recovery mechanisms available in PostgreSQL. Investigate methods for performing comprehensive database integrity checks.  Familiarize yourself with tools that monitor PostgreSQL performance and replication. Finally, learn about various failover strategies and their associated limitations.  These resources will significantly aid in understanding and resolving the issue of replica crashes post-failover.
