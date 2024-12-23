---
title: "Is optimistic locking functionally equivalent to SELECT FOR UPDATE?"
date: "2024-12-23"
id: "is-optimistic-locking-functionally-equivalent-to-select-for-update"
---

Let’s delve into the nuances of optimistic locking and `select for update`. I've spent more than a few late nights troubleshooting concurrency issues, and this is a topic that's always worth revisiting. The short answer is: no, they are not functionally equivalent, though they address similar problems. The crucial difference lies in *how* they handle concurrent access to data, and choosing one over the other depends heavily on your specific use case.

When we talk about `select for update`, we're dealing with *pessimistic locking*. The database, specifically, is responsible for managing concurrent access. If one transaction executes a `select for update` statement on a row, that row is locked for other transactions attempting to do the same. They will be forced to wait until the lock is released, typically by the transaction completing (either commit or rollback). This approach guarantees data consistency by preventing concurrent modifications but introduces the risk of deadlocks and performance bottlenecks if locks are held for extended periods. Imagine a system processing financial transactions; using `select for update` here would ensure that, for example, an account balance isn't updated concurrently by multiple withdrawals. It's a very solid approach in such high-stakes scenarios.

Optimistic locking, on the other hand, takes a more, shall we say, "relaxed" approach initially. It doesn't immediately lock rows. Instead, each row is given a version number (or a timestamp, depending on the implementation). When a transaction retrieves a row for modification, it also reads the current version. Later, when the transaction attempts to update the row, it includes the version number from its initial read in the `where` clause. The update succeeds only if the current version in the database still matches the version the transaction had when it started. If the version is different, it implies another transaction has modified the row, and the update is rejected. This prevents "lost updates," where one transaction overwrites another's changes. We then deal with the failure, perhaps by retrying the transaction with fresh data or informing the user about the conflict.

The key difference is that pessimistic locking blocks at the read stage, waiting for access, whereas optimistic locking detects changes at the write stage. It is not about actively *preventing* modifications while work is in progress; it is about *detecting* them upon submission of the data. This can significantly improve concurrency and avoid blocking for reads in many situations.

Let’s illustrate with some code.

**Example 1: Pessimistic Locking with `select for update`** (assuming PostgreSQL syntax)

```sql
-- transaction 1:
begin transaction;
select balance from accounts where account_id = 123 for update;
-- let's say the balance retrieved is 100
-- do some processing or calculations...
update accounts set balance = 90 where account_id = 123;
commit;

-- transaction 2 (executes after transaction 1's select for update):
begin transaction;
select balance from accounts where account_id = 123 for update;
-- transaction 2 is now blocked, waiting for transaction 1 to commit or rollback
-- it only proceeds once transaction 1 finishes and the lock on the row is released
```
In this scenario, transaction 2 is blocked, ensuring no overlapping updates. It’s straightforward and handles race conditions well, but it does come with blocking behavior that can impact concurrency if used extensively.

**Example 2: Optimistic Locking** (using a simple version number column)

```sql
-- transaction 1:
select balance, version from accounts where account_id = 123;
-- let's say balance is 100, version is 1
-- do some processing/calculations
-- attempted update:
update accounts set balance = 90, version = 2 where account_id = 123 and version = 1;

-- transaction 2 (executes roughly at the same time as transaction 1):
select balance, version from accounts where account_id = 123;
-- gets balance 100, version 1
-- does its own calculations
-- attempted update:
update accounts set balance = 110, version = 2 where account_id = 123 and version = 1;
```

Here’s where the difference shows. If transaction 1 succeeds in its update, the version will change to 2, and the balance will be updated to 90. Transaction 2's update will subsequently fail because its `where version = 1` will not match the current version in the database which is now 2. The application then needs to be aware of this error, so it will likely retry (perhaps after getting the latest balance and version). This highlights that optimistic locking requires more logic in the application than the simple use of `select for update`.
It's the application's responsibility to detect and handle the conflicts rather than relying on the database to prevent them via blocking.

**Example 3: Handling the Optimistic Lock Failure** (in application code - pseudo-code example)

```pseudocode
function updateBalanceOptimistic(accountID, newBalance) {
  try {
    var attempt = 0
    while (attempt < maxRetries){
      var currentData = database.query("select balance, version from accounts where account_id = ?", accountID);
      var currentBalance = currentData.balance
      var currentVersion = currentData.version;
      var updatedVersion = currentVersion + 1; //or perhaps use a timestamp

      //calculate new value, this is where any application specific processing should occur
      var calculatedBalance = calculateNewBalance(currentBalance, newBalance);

      var updateResult = database.execute(
         "update accounts set balance = ?, version = ? where account_id = ? and version = ?",
          calculatedBalance, updatedVersion, accountID, currentVersion);

       if (updateResult.rowsAffected > 0){
           //update successful, we can exit the loop
           return "Update successful";
       } else {
            attempt++
            //the update failed due to version mismatch, retry from start, should probably backoff if retrying often
        }
      }
      //if we reach max retries, return an error
      return "Update Failed after multiple retries";

    } catch (error) {
     //handle database or other errors
     return "Error occurred during database operation: " + error;
   }
}
```

This example illustrates how we handle the scenario where optimistic locking update fails. It involves retrying the operation while keeping a track of retries. This is critical since optimistic locking requires that the application has a specific retry strategy in place.
It is crucial to include error handling to gracefully respond to failures and possibly inform users.

To summarize: `select for update` uses a pessimistic lock to prevent concurrent access at the database level by blocking operations, ensuring strong consistency at the cost of potential concurrency limitations. Optimistic locking, on the other hand, uses a version check on the database during the update phase and lets the application handle conflict detection and resolution. Therefore, it doesn’t block, offering higher concurrency in many use cases, but puts more responsibility on the application for correct error handling.

For deeper reading, consider the following resources. Specifically, for more theoretical perspectives on transaction management in database systems, the book "Database System Concepts" by Abraham Silberschatz, Henry F. Korth, and S. Sudarshan is incredibly valuable. For a more application focused perspective, “Patterns of Enterprise Application Architecture” by Martin Fowler is a very useful resource with various strategies for dealing with database operations and concurrency. Furthermore, any good documentation for databases such as PostgreSQL, MySQL, or Oracle will have sections detailing the implementation of optimistic and pessimistic locking mechanisms.

In my experience, the choice between these isn't always straightforward. If you’re dealing with extremely high contention for writes, and very strict consistency requirements, then `select for update` can work well. But if the write contention is low to moderate, and you need high performance and scalability with a higher number of reads, then optimistic locking is usually the better fit. It all comes down to understanding the tradeoffs and selecting the right tool for the job.
