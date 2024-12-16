---
title: "How does a DDD value object composite identifier look like?"
date: "2024-12-16"
id: "how-does-a-ddd-value-object-composite-identifier-look-like"
---

Alright, let’s tackle this. It's a question I've seen crop up numerous times in the trenches, especially when trying to hammer out a robust domain model using Domain-Driven Design (DDD). Specifically, the composite identifier for a value object often throws people for a loop. Let's break it down.

First, it's essential to revisit what a value object *is* within DDD. Unlike entities, which possess identity and are tracked over time, value objects are immutable and are identified purely by their attributes. Think of it like this: two `Address` objects are considered equal if all their properties (street, city, zip) match, regardless of when or where they were created. They don’t have an intrinsic identity beyond their properties. Now, the 'composite' part kicks in when we need a combination of these properties to uniquely identify that value object within a specific context, particularly when embedding value objects within an entity.

In my experience, I've frequently seen this pattern when modelling geographical data. Imagine an `Order` entity which has a `ShippingAddress` value object. Now, suppose that addresses themselves, in the real world, can have a variety of data points: street address, optional unit number, city, postal code, etc. If we simply hashed all these properties together as a composite identifier, we might not capture meaningful equivalency. We might consider two addresses as 'different' simply because of extra white space characters. Thus a composite identifier needs to combine only the relevant parts of an object that are considered equal in its domain.

The key issue is that a composite identifier isn’t some special magical construct; it’s just a way to represent the uniqueness of a value object based on a combination of its inherent properties. It typically involves overriding the `equals` and `hashCode` methods to include all those relevant fields that make the value object unique. Remember, a poorly constructed composite key can lead to subtle bugs and data discrepancies downstream. I learned this the hard way once debugging an e-commerce system that was incorrectly categorizing the same addresses as distinct and causing inventory miscounts.

Here's a basic example in Java, illustrating a `GeoLocation` value object with a composite identifier:

```java
import java.util.Objects;

public final class GeoLocation {

    private final double latitude;
    private final double longitude;
    private final String geohash;

    public GeoLocation(double latitude, double longitude, String geohash) {
        this.latitude = latitude;
        this.longitude = longitude;
        this.geohash = geohash;
    }

    public double getLatitude() {
        return latitude;
    }

    public double getLongitude() {
        return longitude;
    }
    public String getGeohash(){
        return geohash;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GeoLocation that = (GeoLocation) o;
        return Double.compare(that.latitude, latitude) == 0 &&
               Double.compare(that.longitude, longitude) == 0 &&
               Objects.equals(geohash, that.geohash);
    }

    @Override
    public int hashCode() {
        return Objects.hash(latitude, longitude, geohash);
    }


    @Override
    public String toString() {
      return "GeoLocation{" +
                "latitude=" + latitude +
                ", longitude=" + longitude +
                ", geohash='" + geohash + '\'' +
                '}';
    }
}
```

Notice in the above example, `equals` and `hashCode` use all properties, the latitude, longitude, and geohash fields. They are all relevant for our domain in determining unique instances of `GeoLocation`. When these objects are embedded within a larger entity (e.g., as part of a `DeliveryRoute`), this ensures that two routes are equivalent if, and only if, all of their relevant location objects are equal as well.

Now, consider a situation where the domain dictates that only latitude and longitude determine equality for practical purposes. Maybe in this new application, the geohash is only an extra lookup tool that does not alter uniqueness. Here’s how that change would be reflected:

```java
import java.util.Objects;

public final class GeoLocationSimplified {

    private final double latitude;
    private final double longitude;
     private final String geohash;

    public GeoLocationSimplified(double latitude, double longitude, String geohash) {
        this.latitude = latitude;
        this.longitude = longitude;
         this.geohash = geohash;
    }

    public double getLatitude() {
        return latitude;
    }

    public double getLongitude() {
        return longitude;
    }
     public String getGeohash(){
         return geohash;
     }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GeoLocationSimplified that = (GeoLocationSimplified) o;
        return Double.compare(that.latitude, latitude) == 0 &&
               Double.compare(that.longitude, longitude) == 0;
    }

    @Override
    public int hashCode() {
      return Objects.hash(latitude, longitude);
    }
    
       @Override
    public String toString() {
      return "GeoLocation{" +
                "latitude=" + latitude +
                ", longitude=" + longitude +
                ", geohash='" + geohash + '\'' +
                '}';
    }
}
```
In this modified class `GeoLocationSimplified` the `equals` method only looks at latitude and longitude. So even if two `GeoLocationSimplified` objects differ in their `geohash` they are considered the same instance if their latitude and longitude are identical.

Here’s a more complex example using a `Product` value object with several attributes that contribute to a composite identifier:

```java
import java.util.Objects;

public final class Product {

  private final String productId;
  private final String productName;
  private final String color;
  private final String size;


  public Product(String productId, String productName, String color, String size) {
    this.productId = productId;
    this.productName = productName;
    this.color = color;
    this.size = size;
  }
   public String getProductId() {
        return productId;
    }

    public String getProductName() {
        return productName;
    }

    public String getColor() {
        return color;
    }
     public String getSize(){
         return size;
     }


  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Product product = (Product) o;
    return Objects.equals(productId, product.productId) &&
           Objects.equals(productName, product.productName) &&
           Objects.equals(color, product.color) &&
           Objects.equals(size, product.size);

  }


    @Override
  public int hashCode() {
    return Objects.hash(productId, productName, color, size);
  }

    @Override
    public String toString() {
        return "Product{" +
                "productId='" + productId + '\'' +
                ", productName='" + productName + '\'' +
                ", color='" + color + '\'' +
                ", size='" + size + '\'' +
                '}';
    }
}
```

In this example, a product is considered unique not only by its id but also by its name, color, and size. This shows that your composite identifier is highly dependent on your domain context. If you decide that only product id is relevant to identify a product you can change it to reflect that logic.

For further study on DDD, I’d highly recommend Eric Evans’ "Domain-Driven Design: Tackling Complexity in the Heart of Software." It’s foundational to understand the core principles. Also, Vaughn Vernon’s "Implementing Domain-Driven Design" offers pragmatic guidance on how to apply those concepts in practice. These are classic texts and should be considered essential reading for anyone working with DDD.

Lastly, always consider how your value objects will be used within aggregates and entities. A well-designed composite identifier makes value object comparison robust and less prone to errors. Remember it’s about encapsulating domain logic. It’s not just an exercise in coding, but it's also about accurately representing domain concepts in code.
