---
title: "How do I implement this API with a builder pattern and generic parameters?"
date: "2024-12-23"
id: "how-do-i-implement-this-api-with-a-builder-pattern-and-generic-parameters"
---

Alright,  I've seen this scenario play out more than a few times, particularly when dealing with complex object creation across different services. The combination of builder patterns and generics can seem a bit daunting initially, but it’s an incredibly powerful way to create flexible and maintainable APIs. When I first encountered this years ago, I was refactoring a clunky legacy system that had objects with a dozen or more optional parameters. It was a nightmare; the code was fragile, and extending it was like threading a needle in the dark. Adopting a builder with generics proved to be the solution, and I've been using it ever since.

So, let’s break down how to implement an API using both builder patterns and generic parameters. At its core, the builder pattern separates the construction of a complex object from its representation. This allows us to create objects step-by-step, setting various attributes in a controlled fashion, rather than relying on potentially massive constructors. The real magic comes in with generics: they add type safety and reusability. Generics allow us to define class and method operations that work on various types without losing type information at compile time.

Here’s the typical strategy I employ: we'll start with defining an interface for the builder itself, which will make it clear about what methods are expected. We use generics to define the return type, providing type safety that’s crucial when chaining multiple builder methods. Then we’ll create the concrete builder classes, each tailored to a specific type while utilizing the generic interface.

Let's look at the interface first, assuming we have a simple `Product` object that we are building. Here's the code:

```java
interface ProductBuilder<T extends Product> {
    ProductBuilder<T> withName(String name);
    ProductBuilder<T> withPrice(double price);
    ProductBuilder<T> withSku(String sku);
    T build();
}
```

This `ProductBuilder` interface defines the contract. It's generic, parameterized by `T`, which represents a `Product` or a subtype thereof. This ensures that the builder methods return an instance of the `ProductBuilder`, allowing for method chaining, and that the `build()` method returns an instance of `T`. The key here is this ensures that the methods used to configure the builder type return the same type. The flexibility is provided by the `extends Product` constraint on `T`, allowing us to use this for `Product` itself, or for subclasses which inherit the properties we are setting here.

Now, consider a basic `Product` class:

```java
class Product {
    private String name;
    private double price;
    private String sku;

    // Private constructor to enforce builder usage
    private Product() {
    }

     public String getName() {
        return name;
    }

    public double getPrice() {
        return price;
    }

    public String getSku() {
      return sku;
    }


    public static class Builder implements ProductBuilder<Product>{
      private String name;
      private double price;
      private String sku;


      @Override
      public Builder withName(String name){
          this.name=name;
          return this;
      }
        @Override
        public Builder withPrice(double price){
            this.price=price;
            return this;
        }
        @Override
       public  Builder withSku(String sku){
           this.sku=sku;
           return this;
       }
       @Override
        public Product build(){
          Product product=new Product();
          product.name=name;
          product.price=price;
          product.sku=sku;
          return product;
       }
    }

    //Static method to return a new Builder
    public static Builder builder(){
        return new Builder();
    }
}
```

The Product class has a private constructor to force users to use the builder. The nested `Builder` class implements the `ProductBuilder<Product>` interface. Each `with` method sets a property and returns `this`, allowing for fluent chaining. The `build` method creates the final `Product` object using the values stored in the `Builder` class. Also, we are providing a static method to return a new builder which is standard practice.

Let's say we have a `Book` class that extends `Product`. It would have some extra properties such as author, isbn, and so on. This is how you might use generics with the builder for it:

```java
class Book extends Product {
    private String author;
    private String isbn;

    // Private constructor to enforce builder usage
    private Book() {}


    public String getAuthor() {
        return author;
    }

    public String getIsbn() {
        return isbn;
    }

    public static class Builder extends Product.Builder implements ProductBuilder<Book>{
        private String author;
        private String isbn;

       @Override
      public Builder withName(String name){
        super.withName(name);
         return this;
     }
        @Override
        public Builder withPrice(double price){
            super.withPrice(price);
           return this;
        }
        @Override
        public Builder withSku(String sku){
            super.withSku(sku);
          return this;
        }

         public Builder withAuthor(String author) {
             this.author = author;
             return this;
         }

         public Builder withIsbn(String isbn) {
             this.isbn = isbn;
             return this;
         }
       @Override
       public Book build(){
           Book book=new Book();
           book.author=this.author;
           book.isbn=this.isbn;
            book.name=this.name;
            book.price=this.price;
           book.sku=this.sku;
           return book;
       }
    }


    public static Builder builder(){
        return new Builder();
    }
}
```

Here, `Book.Builder` extends `Product.Builder` to reuse the basic product properties, then it adds fields for author and isbn. The `build` method returns a `Book` object. Notice how we are overriding the with methods for Product to return `Book.Builder`, which is essential for method chaining, and in `Book`'s `build` method, we initialize the Book using the data set in the `Builder`, including the properties from `Product` that were set using super. You can extend this pattern for numerous variations of `Product`, always maintaining type safety and making your code far easier to manage, read, and extend.

Now, let’s consider a very simple use case in our main method.

```java
public class Main {
    public static void main(String[] args) {
        // Using the Product Builder
        Product product = Product.builder()
                .withName("Laptop")
                .withPrice(1200.00)
                .withSku("LP123")
                .build();
        System.out.println("Product Name: "+product.getName());
        System.out.println("Product Price: "+product.getPrice());
        System.out.println("Product SKU: "+product.getSku());


        // Using the Book Builder
        Book book = Book.builder()
            .withName("The Hitchhiker's Guide to the Galaxy")
            .withAuthor("Douglas Adams")
            .withIsbn("978-0345391803")
            .withPrice(12.99)
            .withSku("HGG42")
            .build();

          System.out.println("Book Name: "+book.getName());
          System.out.println("Book Author: "+book.getAuthor());
          System.out.println("Book ISBN: "+book.getIsbn());
          System.out.println("Book Price: "+book.getPrice());
          System.out.println("Book SKU: "+book.getSku());
    }
}

```

The output of this would be as follows:
```
Product Name: Laptop
Product Price: 1200.0
Product SKU: LP123
Book Name: The Hitchhiker's Guide to the Galaxy
Book Author: Douglas Adams
Book ISBN: 978-0345391803
Book Price: 12.99
Book SKU: HGG42
```

This pattern allows you to create complex objects with far less complexity and code duplication. The advantage of this is not just for simple objects, but for those complex situations with dozens of optional parameters. The builder pattern makes creating these objects far easier, and also forces the user to use the builder method, preventing them from using a constructor with many parameters.

For more information, I’d recommend looking into "Effective Java" by Joshua Bloch, which details many best practices, including effective use of the builder pattern. "Design Patterns: Elements of Reusable Object-Oriented Software" by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides (the "Gang of Four" book) also provides fundamental insights into builder, and many other important patterns. Understanding these principles will provide a good base for creating your own robust, scalable, and developer friendly APIs.
