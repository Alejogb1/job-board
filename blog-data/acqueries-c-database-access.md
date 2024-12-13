---
title: "acqueries c# database access?"
date: "2024-12-13"
id: "acqueries-c-database-access"
---

Okay so you're asking about accessing databases in C# right Been there done that more times than I care to admit Let's break this down like we're debugging some ancient spaghetti code

First thing's first you've got options for how you wanna connect to your database The big boys are ADO NET and Entity Framework ADO NET is kinda like the old school manual transmission way of doing things You're handling the connection opening closing and command execution yourself It's flexible and you have very precise control but its more work upfront Entity Framework is the automatic transmission its an ORM Object-Relational Mapper that does a lot of heavy lifting It takes your c# objects and maps them to database tables its convenient and quicker for standard CRUD operations but can be less performant if you don't know what your doing So its kinda of a trade of more flexibility for more work and more performance for less control

I remember vividly way back when I was building this inventory management system for a friend's electronics shop I went all in on ADO NET at first thinking I was hot stuff and could handle everything myself Turns out writing all that SQL and managing connection objects became a nightmare really fast I spent almost a day debugging a simple insert statement because I forgot to close the connection properly It was a rookie move but a good learning experience

Let's see some code snippets alright lets look at an ADO NET example that goes against my experiences and you know what not recommended for production use but still an example alright here we go lets suppose you are using SQL Server

```csharp
using System.Data.SqlClient;
using System.Data;

public void GetProducts()
{
  string connectionString = "Server=myServerAddress;Database=myDatabase;User Id=myUsername;Password=myPassword;";

  using(SqlConnection connection = new SqlConnection(connectionString))
  {
    try
    {
        connection.Open();
        string sql = "SELECT ProductID, ProductName, Price FROM Products";
        SqlCommand command = new SqlCommand(sql, connection);
        SqlDataReader reader = command.ExecuteReader();

      while(reader.Read())
      {
        Console.WriteLine($"Product ID: {reader["ProductID"]} Name: {reader["ProductName"]} Price: {reader["Price"]}");
      }
       reader.Close();

    }

   catch (Exception ex)
   {
       Console.WriteLine("Error "+ ex.Message);
   }
   }
}
```

Ok so what are we doing here Using statements here are crucial for automatically closing the connections and disposing of the objects like the connection and command in this example and so on This code opens a connection to the database executes a sql query and reads the data from the `SqlDataReader` object Its classic ADO NET the kind of thing I would have written years ago after being so confident it would be easier than what I was going to use in the next example

Ok and what would the entity framework solution look like Lets go back to the future and remember my time when I have to deal with more complex operations in the system and it was going to start scaling so this is much more suitable for that now

First you'd usually install the nuget package `Microsoft.EntityFrameworkCore.SqlServer` (or the correct one for your database provider) You also need to define the model objects corresponding to your tables and the DbContext that manages the database connection So lets assume your product looks like this

```csharp
public class Product
{
  public int ProductId {get; set;}
  public string ProductName {get; set;}
  public decimal Price {get; set;}
}
```
And a db context that we will use

```csharp
using Microsoft.EntityFrameworkCore;
public class AppDbContext : DbContext
{
  public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) {}
  public DbSet<Product> Products {get; set;}
}
```
Then the main method would be something like this
```csharp
using Microsoft.EntityFrameworkCore;
public void GetProductsEF()
{

 var options = new DbContextOptionsBuilder<AppDbContext>().UseSqlServer("Server=myServerAddress;Database=myDatabase;User Id=myUsername;Password=myPassword;").Options;
 using (var context = new AppDbContext(options))
 {
    var products = context.Products.ToList();
    foreach(var product in products)
    {
         Console.WriteLine($"Product ID: {product.ProductId} Name: {product.ProductName} Price: {product.Price}");
    }
 }
}
```
See how much cleaner that looks? Entity Framework handles the connection opening closing and data mapping for you automatically All you need to do is configure the db context to use your connection string which is done in the options variable and access data via the `Products` property and query it.

Now if you need to perform more complex queries or data manipulation things start to get more involved regardless of what you do But at this point it's a question of "do I need the extra control and performance" or "I want to write this very fast" and also the size of the system you want to make you can go with either solution

And here is a more advanced example of EF with filtering and a query

```csharp
using Microsoft.EntityFrameworkCore;
using System.Linq;
public void GetCheapProductsEF()
{
  var options = new DbContextOptionsBuilder<AppDbContext>().UseSqlServer("Server=myServerAddress;Database=myDatabase;User Id=myUsername;Password=myPassword;").Options;
  using(var context = new AppDbContext(options))
  {
    decimal maxPrice = 20;
    var cheapProducts = context.Products.Where(p=>p.Price <= maxPrice).OrderBy(p=>p.Price).ToList();
    foreach (var product in cheapProducts)
    {
         Console.WriteLine($"Product ID: {product.ProductId} Name: {product.ProductName} Price: {product.Price}");
    }
  }

}

```

See how I can use Linq here to filter and order products It is so easy to read and modify now and also it can scale to more complex solutions much easier.

A common problem I see often is folks forgetting to dispose of database connections This leads to connection leaks and performance issues. The `using` keyword as I showed you before is your best friend when working with ADO.NET and also the db context since they implement IDisposable. Also you can see I had to include an exception handler since things can and will fail at some point in time so it needs to be handled accordingly so the app doesn't crash and also it can be logged for future analysis

Another thing I've seen is people trying to write incredibly complex SQL queries directly in the code especially with ADO.NET While SQL is powerful sometimes it's better to let Entity Framework handle some of the heavy lifting or break those queries down into a more structured approach.

When thinking about performance make sure to index your columns properly in the database. No matter how good your C# code is if the database queries aren’t efficient you will still have performance problems.

You can learn more on this subject by the classic "ADO NET in a nutshell" by O'Reilly but for modern usage and practices consider the "Programming Entity Framework" series or similar but make sure it is up to date. These cover best practices and design patterns when working with databases in C#.

You've also got to think about things like connection pooling which ADO.NET manages automatically but its something you should be aware of. Also understand how the database engine works with query plans and execution strategies. Its good to learn the theory behind database management so you have a solid base and understand all these concepts.

Oh and before I forget the worst thing that can happen when connecting to a database is that one day its down you don't know how to connect to it and suddenly you have no idea of where is the data and start a panic attack It's not pretty trust me it happened to a friend once... It was pretty sad day

Anyway I hope this has helped you understand a bit better how to work with databases in C# It's a broad subject so keep practicing and refining your skills and remember to stay calm and not get a panic attack when it goes down it happens.
