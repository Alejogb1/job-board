---
title: "How is a DDD value object's composite identifier constructed?"
date: "2024-12-23"
id: "how-is-a-ddd-value-objects-composite-identifier-constructed"
---

Let's unpack this, shall we? The notion of a value object in domain-driven design (ddd) is fundamental, and when that value object needs a composite identifier, things get interesting—and require careful consideration. I recall a particularly tricky implementation while working on a distributed logistics system. We were tracking packages, and each package had a unique identifier, but that identifier was not a single string or number. It was, in fact, a composite built from the origin warehouse id, the destination warehouse id, and a sequential batch number. That experience really hammered home the importance of handling these composite keys correctly.

At its core, a ddd value object is immutable. It represents a conceptual whole based on its attributes rather than having an independent identity. This characteristic dictates how we must handle identifiers. We're not looking at a traditional database primary key that auto-increments; instead, the "identity" of a value object arises from the combined values of its components. A composite identifier, therefore, is not a property *of* a value object, it *is* the value object, or at least, a critical part of its immutable data defining uniqueness. It’s critical to differentiate this from an entity, where identity is crucial and distinct from its properties.

The construction of this identifier should ideally happen within the value object's constructor or a dedicated factory method. This encapsulation ensures that the construction logic is self-contained and consistent. This is precisely the principle of value objects: they are self-contained units that maintain their internal state integrity. Moreover, the process should ideally be deterministic; given the same constituent parts, we should always arrive at the same identifier representation.

Let's consider an example in python (though the concept is language-agnostic). Imagine we're dealing with product codes: a manufacturer identifier and a sequential serial number form our composite identifier.

```python
class ProductCode:
    def __init__(self, manufacturer_id: str, serial_number: int):
        if not isinstance(manufacturer_id, str) or not manufacturer_id:
            raise ValueError("Manufacturer ID must be a non-empty string.")
        if not isinstance(serial_number, int) or serial_number <= 0:
            raise ValueError("Serial number must be a positive integer.")

        self.manufacturer_id = manufacturer_id
        self.serial_number = serial_number
        self._identifier = f"{manufacturer_id}-{serial_number:08}"  # Zero-padded for consistency

    @property
    def identifier(self):
        return self._identifier

    def __eq__(self, other):
        if not isinstance(other, ProductCode):
            return False
        return (self.manufacturer_id, self.serial_number) == (other.manufacturer_id, other.serial_number)

    def __hash__(self):
        return hash((self.manufacturer_id, self.serial_number))

    def __str__(self):
      return self._identifier
```

Here, we've encapsulated the logic to create the identifier within the `__init__` method. We're using f-strings for formatting, ensuring the serial number is zero-padded. This provides consistency when you need to, say, use these keys in a map or compare them across multiple instances. The `__eq__` and `__hash__` methods are implemented to ensure value equality, not object identity. This is crucial for value objects; we care if two instances have the same internal state, not if they are the same object in memory. The `__str__` allows for easily rendering our key as a string when needed.

Now, let's shift to a java example, modeling a geographic coordinate. Here, we’ll use latitude and longitude as components of our composite identifier.

```java
import java.util.Objects;

public final class GeoCoordinate {
    private final double latitude;
    private final double longitude;
    private final String identifier;

    public GeoCoordinate(double latitude, double longitude) {
        if (latitude < -90 || latitude > 90) {
            throw new IllegalArgumentException("Latitude must be between -90 and 90.");
        }
        if (longitude < -180 || longitude > 180) {
            throw new IllegalArgumentException("Longitude must be between -180 and 180.");
        }

        this.latitude = latitude;
        this.longitude = longitude;
        this.identifier = String.format("%.6f,%.6f", latitude, longitude); // Format to 6 decimal places
    }

   public String getIdentifier(){
        return this.identifier;
   }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GeoCoordinate that = (GeoCoordinate) o;
        return Double.compare(that.latitude, latitude) == 0 && Double.compare(that.longitude, longitude) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(latitude, longitude);
    }

    @Override
    public String toString() { return this.identifier; }
}
```

In this example, we use `String.format` to create the identifier ensuring consistent formatting. Again, `equals` and `hashCode` are overridden for proper equality comparison. Note that we are using the `final` keyword here; in java, making the class final helps enforce immutability, a core characteristic of value objects.

Finally, let's see a c# example, simulating a currency and amount combination for a transaction.

```csharp
public class Money
{
  public string Currency {get; init;}
  public decimal Amount {get; init;}
  public string Identifier {get;}


  public Money(string currency, decimal amount){
    if(string.IsNullOrEmpty(currency)) {
      throw new ArgumentException("Currency cannot be null or empty.", nameof(currency));
    }

    if(amount <= 0){
        throw new ArgumentOutOfRangeException(nameof(amount), "Amount must be a positive value.");
    }

    Currency = currency;
    Amount = amount;
    Identifier = $"{currency}-{amount:N2}";
  }

   public override bool Equals(object? obj)
    {
        if (obj == null || GetType() != obj.GetType())
        {
            return false;
        }

        var other = (Money)obj;
        return Currency == other.Currency && Amount == other.Amount;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(Currency, Amount);
    }

    public override string ToString()
    {
        return Identifier;
    }
}
```

Here, we are utilizing the format string capabilities of C# combined with a null-check and an out-of-range check for the `amount`. The concept remains the same across all three examples. The identifier is deterministically created based on the input parameters and is encapsulated within the value object. Immutability is enforced by using properties with only init accessors. The equality and hash code overrides ensure our value object behaves as expected.

From a pragmatic perspective, when choosing how to represent your composite identifiers as strings, be mindful of storage and query optimization. If these identifiers are part of database queries or used extensively in distributed environments, consider their size and readability. JSON serialization and deserialization needs to be considered, and choosing a consistently parsable format for the string is crucial. This is why I favored a consistent delimiter and padding where it would make sense to do so in all three examples. If using a hash as the composite key, while performant and small, you should still persist the components elsewhere for traceability, as the hash can only be relied upon for equality, not reconstruction of the components.

For further reading on the theoretical underpinnings of domain-driven design, I recommend Eric Evans' "Domain-Driven Design: Tackling Complexity in the Heart of Software." This is foundational for understanding value objects and their role. For more practical application, Vaughn Vernon's "Implementing Domain-Driven Design" offers concrete examples and patterns for implementation. Martin Fowler's "Patterns of Enterprise Application Architecture" provides a wider view on software architecture, which is helpful when considering these design patterns within the context of a whole system. Furthermore, any book discussing object-oriented programming, especially the concepts of immutability and value objects, would be beneficial to consult.

Ultimately, the construction of a composite identifier for a ddd value object is about maintaining its integrity, immutability, and equality semantics. It is not an arbitrary process; it must be deterministic and encapsulated within the value object itself, reflecting a fundamental understanding of value object’s role in a domain model. My experience has shown that taking this approach prevents the proliferation of incorrect identifier handling throughout your application.
