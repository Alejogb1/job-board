---
title: "How can the Composite pattern be mapped using EF Core?"
date: "2024-12-16"
id: "how-can-the-composite-pattern-be-mapped-using-ef-core"
---

Alright, let's talk about the composite pattern and how it plays with entity framework core (ef core). I've tangled with this particular challenge a few times, most memorably on a project involving a complex product catalog where items could be grouped into sets and sub-sets. The naive approach of separate entities for 'product' and 'set' quickly turned into a relational nightmare.

The composite pattern, for those needing a quick refresh, is a structural design pattern that lets you compose objects into tree structures and then treat individual objects and compositions uniformly. This is incredibly useful when you have a hierarchy of objects where some objects are simple (leaves) and others are composed of other objects (branches). The typical example is a file system; you have individual files (leaves) and directories (branches) that can contain both files and other directories.

Now, mapping this to ef core can initially seem tricky because ef core, at its heart, wants relational data. Hierarchical data, by definition, tends to buck that straightforward tabular structure. However, it's not insurmountable. We can effectively represent a composite using a single table with some smart conventions and configuration. The primary strategy revolves around using self-referencing relationships and potentially a discriminator column for cases where the leaf nodes and branch nodes have different properties.

My experience has shown that the most effective approach involves having a single entity represent both leaf and composite nodes. Consider a scenario similar to the product catalog I mentioned earlier, let’s call this entity `CatalogItem`. This entity would have an `Id`, a `Name`, and a foreign key, `ParentId`, which relates back to the `CatalogItem` table itself. Critically, this `ParentId` is nullable, as root items (the top of the hierarchy) will have no parent.

Here's how you'd model this in C# using ef core:

```csharp
public class CatalogItem
{
    public int Id { get; set; }
    public string Name { get; set; }
    public int? ParentId { get; set; }
    public CatalogItem Parent { get; set; }
    public ICollection<CatalogItem> Children { get; set; }
}

public class CatalogContext : DbContext
{
    public CatalogContext(DbContextOptions<CatalogContext> options) : base(options)
    {
    }

    public DbSet<CatalogItem> CatalogItems { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<CatalogItem>()
                    .HasOne(c => c.Parent)
                    .WithMany(c => c.Children)
                    .HasForeignKey(c => c.ParentId)
                    .OnDelete(DeleteBehavior.Restrict); // Avoid accidental cascading delete

        base.OnModelCreating(modelBuilder);
    }
}
```

In this code, `CatalogItem` has a self-referencing relationship. The `Parent` property represents the parent `CatalogItem` and the `Children` property represents the child `CatalogItem` instances. The `OnDelete(DeleteBehavior.Restrict)` is crucial to prevent inadvertent cascading deletes, and you might need to refine this depending on your application's requirements.

Now, one question that often arises is: what if leaf nodes need properties that composite nodes don't, and vice-versa? For instance, what if `product` (a leaf) has a `price`, but a `set` (composite) needs a `discountPercentage`. This is where a discriminator column can be invaluable. Let's expand on our example:

```csharp
public abstract class CatalogItem
{
    public int Id { get; set; }
    public string Name { get; set; }
    public int? ParentId { get; set; }
    public CatalogItem Parent { get; set; }
    public ICollection<CatalogItem> Children { get; set; }
}

public class Product : CatalogItem
{
   public decimal Price { get; set; }
}

public class Set : CatalogItem
{
    public decimal DiscountPercentage { get; set; }
}

public class CatalogContext : DbContext
{
    public CatalogContext(DbContextOptions<CatalogContext> options) : base(options)
    {
    }

    public DbSet<CatalogItem> CatalogItems { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<CatalogItem>()
                    .HasOne(c => c.Parent)
                    .WithMany(c => c.Children)
                    .HasForeignKey(c => c.ParentId)
                    .OnDelete(DeleteBehavior.Restrict);

         modelBuilder.Entity<CatalogItem>()
             .UseTphMappingStrategy() // Table-Per-Hierarchy
             .HasDiscriminator<string>("ItemType")
             .HasValue<Product>("product")
             .HasValue<Set>("set");

        base.OnModelCreating(modelBuilder);
    }
}
```

Here, I've made `CatalogItem` an abstract base class. We now have concrete classes `Product` and `Set` which inherit from it, each with their unique properties. The `UseTphMappingStrategy` and subsequent configurations tell ef core to use a single table with a discriminator column named `ItemType`. This allows ef core to correctly instantiate the correct concrete class when fetching records from the `CatalogItems` table.

Finally, let's explore the common scenario of wanting to query up or down the hierarchy. Fetching all items under a given set might seem daunting at first, but it's actually quite manageable with ef core's powerful querying capabilities:

```csharp
public async Task<List<CatalogItem>> GetDescendants(int parentId)
{
    var descendants = new List<CatalogItem>();
    var root = await _context.CatalogItems.FindAsync(parentId);
    if (root == null) return descendants;

     Queue<CatalogItem> queue = new Queue<CatalogItem>();
     queue.Enqueue(root);

     while (queue.Count > 0)
     {
       CatalogItem current = queue.Dequeue();
       descendants.Add(current);

       if(current.Children != null)
       {
            foreach(var child in current.Children)
                queue.Enqueue(child);
       }
     }

    return descendants;
}
```

This example utilizes a breadth-first search (bfs) to recursively retrieve all descendants of a given parent node. It's also possible to achieve similar results using recursive common table expressions (ctes) within sql queries generated by ef core, especially in sql server. However, depending on the depth of your hierarchy, the bfs approach in memory may prove more efficient. Remember to configure lazy or eager loading properly to suit your application's performance needs.

In practice, dealing with the composite pattern in ef core isn’t so much about fighting the relational nature of the database, but more about leveraging ef core’s features to model the hierarchy effectively. Choosing between table-per-hierarchy, single table inheritance, or separate tables depends on your specific needs and how distinct the properties of your composite and leaf nodes are. Careful planning and a thorough understanding of ef core's capabilities are key.

For further reading, I'd recommend exploring Martin Fowler's "Patterns of Enterprise Application Architecture," particularly the sections on hierarchy patterns and object-relational mapping, as well as the official Microsoft documentation for ef core. Additionally, the book "Entity Framework Core in Action" by Jon P. Smith is a fantastic practical guide that goes into greater detail on various mapping strategies. These resources will provide a strong theoretical and practical foundation for handling the composite pattern and other relational mapping scenarios.
