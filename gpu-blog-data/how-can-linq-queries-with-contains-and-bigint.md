---
title: "How can LINQ queries with `Contains` and `bigint` values be optimized for performance?"
date: "2025-01-30"
id: "how-can-linq-queries-with-contains-and-bigint"
---
The performance of LINQ queries utilizing `Contains` with `bigint` values can degrade significantly, particularly when dealing with large datasets and long lists of `bigint` values. The root cause lies in the manner `Contains` is internally implemented against a non-indexed collection – it often results in a sequential scan. I have repeatedly observed this performance bottleneck while working with large-scale data analysis pipelines where primary keys are represented as `bigint` and extracted to filter other entities. Optimizing this interaction demands a focused approach that minimizes scan operations and maximizes indexing capabilities.

The straightforward use of `Contains` on a collection of `bigint` against a database context using LINQ to Entities (or similar providers) translates to an inefficient WHERE IN clause in SQL. For example, if we have a `User` entity with a `UserId` property of type `bigint`, and we have a large list of `bigint` values representing IDs we want to retrieve, directly using `Contains` results in a SQL query that iterates through every provided `bigint` ID, causing performance to drop linearly with respect to the number of elements in the list. The database has to sequentially check each entry in the `User` table against each value in the provided list, which can lead to substantial delays.

The preferred optimization centers on moving data filtering as much as possible into the database layer, leveraging indexes. Instead of passing a long list to `Contains`, a temporary table with the `bigint` values is more performant. This provides the database engine a set of values against which it can use optimized lookup methods, potentially taking advantage of indexes if they are defined on the table hosting the `bigint` value in the database.

Let's examine three practical scenarios with code examples, demonstrating both the problem and a viable optimization.

**Example 1: The Inefficient `Contains` Approach**

This first example represents the problematic code that should be avoided.

```csharp
public List<User> GetUsersByIdsInefficient(List<long> userIds)
{
    using (var context = new MyDbContext())
    {
      return context.Users
                    .Where(u => userIds.Contains(u.UserId))
                    .ToList();
    }
}
```

This method takes a list of `long` (equivalent to `bigint`) values representing `UserId` and retrieves corresponding `User` entities. The `Contains` operation here, when translated to SQL, leads to a `WHERE UserId IN (id1, id2, ..., idN)` clause. For a long list of `userIds`, the database might perform poorly due to the lack of indexing on such a direct query with a long sequence of provided values.

**Example 2: Using a Temporary Table for Optimized Filtering**

This second example employs a temporary table to enhance performance, a strategy I've successfully deployed in similar contexts with positive results.

```csharp
public List<User> GetUsersByIdsOptimized(List<long> userIds)
{
    using (var context = new MyDbContext())
    {
        // Create a temporary table type (if not already present) and insert the provided IDs
        var tempTable = context.Database.SqlQuery<long>(
            "CREATE TABLE #tempUserIds (UserId BIGINT PRIMARY KEY); " +
            string.Join("", userIds.Select(id => $"INSERT INTO #tempUserIds VALUES ({id});")));

        // Retrieve users, joining with the temp table
        var users = context.Users
            .Join(tempTable,
                user => user.UserId,
                tempId => tempId,
                (user, tempId) => user)
            .ToList();

        // Drop the temporary table after use
         context.Database.ExecuteSqlCommand("DROP TABLE #tempUserIds;");

        return users;
    }
}
```

In this method, we create a temporary table named `#tempUserIds` and insert the `userIds` into it. Subsequently, we use an `INNER JOIN` to select `User` entities where the `UserId` exists in the temporary table.  This allows the SQL engine to utilize its indexing strategies on the `UserId` column during the join, significantly improving the query's execution time. The temporary table is then dropped after the operation, maintaining data integrity. It is also important to ensure `tempTable` is enumerated (e.g., using `.ToList()`) as it otherwise might result in an unnecessary double execution of the create table statement.  In production I would parameterize insert statements to prevent SQL injection if the `userIds` are untrusted input.

**Example 3: Using Database-Specific Table-Valued Parameters**

This third approach delves into database-specific features to further refine our strategy when working with SQL Server, an environment where I have personally seen major performance improvements.

```csharp
public List<User> GetUsersByIdsSqlTableValuedParameters(List<long> userIds)
{
  using (var context = new MyDbContext())
  {
     var tableParam = new SqlParameter("@userIds", SqlDbType.Structured)
     {
          TypeName = "dbo.BigIntList",
          Value = CreateDataTableFromLongList(userIds)
     };


     var sql = "SELECT u.* FROM Users u INNER JOIN @userIds t ON u.UserId = t.Value";
     return context.Database.SqlQuery<User>(sql, tableParam).ToList();

  }
}

private DataTable CreateDataTableFromLongList(List<long> userIds)
{
    var table = new DataTable();
    table.Columns.Add("Value", typeof(long));
    foreach (var id in userIds)
    {
        table.Rows.Add(id);
    }
    return table;
}
```

This method employs SQL Server's Table-Valued Parameters (TVPs) functionality. Before calling the method, I assume a user-defined table type named `BigIntList` exists in the database, having a single column of type `bigint`, which represents the IDs. This can be created with a script such as: `CREATE TYPE BigIntList AS TABLE ( Value BIGINT );`.  The `CreateDataTableFromLongList` method converts the `List<long>` to a `DataTable` which is then passed as an `@userIds` parameter of SQL parameter type `SqlDbType.Structured` along with the type name of the user defined type. This enables highly optimized joins within the database without incurring the overhead of constructing insert statements.

**Resource Recommendations**

For a deeper understanding of optimizing LINQ with SQL Server, researching articles and documentation related to: SQL Server table-valued parameters; performance tuning in SQL Server, particularly concerning indexed columns in JOIN clauses;  and the specifics of how your chosen LINQ provider (e.g., Entity Framework Core) translates LINQ operations to SQL, specifically focusing on its behavior when using `Contains`.
These areas of study, focusing on database-specific performance characteristics, have proved immensely valuable in similar optimization tasks I have undertaken. Further, exploring the execution plans generated by your database can offer immediate insights into inefficient queries.

By implementing these strategies, the performance impact of filtering entities based on a list of `bigint` values can be greatly minimized, leading to a more efficient application. The critical takeaway is to shift filtering operations into the database layer and make effective use of indexes wherever possible.
