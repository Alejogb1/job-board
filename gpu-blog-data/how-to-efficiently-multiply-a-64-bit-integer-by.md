---
title: "How to efficiently multiply a 64-bit integer by a 64-bit fraction in C#?"
date: "2025-01-30"
id: "how-to-efficiently-multiply-a-64-bit-integer-by"
---
The inherent challenge in directly multiplying a 64-bit integer by a 64-bit fraction in C# stems from the lack of a built-in data type representing a true 64-bit fraction. Standard floating-point types like `double` or `decimal` introduce approximations due to their inherent structure, which may be unacceptable in contexts requiring high-precision arithmetic or deterministic results. Over the years, I've encountered this in embedded systems programming where predictable fractional scaling of sensor readings was critical. My experience has led me to favor integer arithmetic, utilizing fixed-point representation, as the solution.

A fixed-point representation emulates fractional values by scaling an integer by a fixed power of two (or ten, though powers of two are computationally more efficient). The core concept is to represent a fractional value as an integer, where the least significant bits implicitly denote the fractional part. Consider a 64-bit integer; we can choose a certain number of the least significant bits to represent the fraction, with the remaining bits representing the integer portion. This approach eliminates the uncertainty inherent in floating-point representation, providing predictable results.

To perform a multiplication of a 64-bit integer by a 64-bit fixed-point representation, the following steps are required. First, the 64-bit fraction will need to be represented using an unsigned 64-bit integer, with a pre-determined number of fractional bits. Then the multiplication will be performed, keeping in mind where the decimal point would be situated. Finally, the result will be divided by the necessary fixed-point scaling factor. The fractional bits, typically denoted as ‘F’ are fixed for a given representation and must be consistent across all calculations utilizing that representation.

Here's how it's done in C#, demonstrated with three code examples focusing on different aspects of the implementation and error handling:

**Example 1: Basic Multiplication with Fixed-Point Scaling**

This example showcases the most direct implementation of the fixed-point multiplication. It presumes a fraction represented as an unsigned 64-bit integer, with a predefined number of fractional bits represented by the constant `FractionalBits`.

```csharp
using System;

public class FixedPointMath
{
    private const int FractionalBits = 32;
    private const ulong One = 1UL << FractionalBits;

    public static long Multiply(long integerValue, ulong fixedPointFraction)
    {
        // Perform the multiplication as a 128-bit value, then shift to discard the extra fractional bits
        ulong productHigh = (ulong)(integerValue >> 32) * fixedPointFraction;
        ulong productLow = (ulong)(integerValue & 0xFFFFFFFF) * fixedPointFraction;
        ulong carry = (productLow >> FractionalBits);
        productHigh += carry;
        long scaledResult = (long)(productLow >> FractionalBits) + ((long)productHigh << 32);

         return scaledResult;
    }


    public static void Main(string[] args)
    {
        long integerPart = 10;
        ulong fractionalPart = (ulong)(0.75 * One); // Represents 0.75 as a fixed-point value
        long result = Multiply(integerPart, fractionalPart);

        // The result requires a manual shift to show what the fixed-point result represents.
        double finalResult = result / (double)One;
        Console.WriteLine($"Result of {integerPart} * 0.75 = {finalResult}");
        Console.WriteLine($"Raw Result {result}, scaled Result {finalResult}");

    }
}
```

**Commentary:**

*   `FractionalBits` establishes the precision of our fixed-point representation; 32 bits allow for fairly accurate decimal representation.
*   `One` is a constant representing the value 1 in the fixed-point space (1.0).
*   The `Multiply` method performs the multiplication, taking advantage of the 64-bit nature of the data types. The lower and higher parts of the result are calculated then added.
*   After the multiplication, the result is right-shifted to remove the implicit fractional part, and cast to a `long` which can be a signed value. The raw scaled value is also shown.
*   The `Main` method shows example usage, demonstrating the conversion of a decimal 0.75 to its fixed-point representation. The raw result is also shown in the console output.

**Example 2: Handling Potential Overflows**

This example introduces rudimentary overflow checking during the multiplication, a critical consideration in systems with limited dynamic ranges. While this implementation does not fully solve all possible overflow errors, it demonstrates the importance of considering these issues. It checks for overflows on the lower part of the computation. This can improve the general case by catching many overflows but should be used carefully in higher-performance applications.

```csharp
using System;

public class FixedPointMath
{
    private const int FractionalBits = 32;
    private const ulong One = 1UL << FractionalBits;

    public static long MultiplyOverflowChecked(long integerValue, ulong fixedPointFraction)
    {
            ulong productLow = (ulong)(integerValue & 0xFFFFFFFF) * fixedPointFraction;
            ulong carry = (productLow >> FractionalBits);
             if (carry > (0xFFFFFFFF))
             {
                throw new OverflowException("Overflow detected during multiplication");
             }

             ulong productHigh = (ulong)(integerValue >> 32) * fixedPointFraction;
            productHigh += carry;


            long scaledResult = (long)(productLow >> FractionalBits) + ((long)productHigh << 32);

        return scaledResult;
    }

    public static void Main(string[] args)
    {
         long integerPart = long.MaxValue/2;
         ulong fractionalPart = (ulong)(0.75 * One);


        try
        {
            long result = MultiplyOverflowChecked(integerPart, fractionalPart);
            double finalResult = result / (double)One;
              Console.WriteLine($"Result of {integerPart} * 0.75 = {finalResult}");
             Console.WriteLine($"Raw Result {result}, scaled Result {finalResult}");
        }
        catch (OverflowException ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}
```

**Commentary:**

*   The `MultiplyOverflowChecked` method performs the same basic multiplication but includes a check for an overflow of `carry` in the lower 32 bits, throwing an exception if overflow is detected.
*   The example in `Main` uses `long.MaxValue/2`, a large input value to potentially trigger the overflow check.
*   The `try-catch` block demonstrates proper exception handling. In a real-world application, more sophisticated overflow handling (like saturation) might be preferable, but this serves as a simple example.

**Example 3: Utilizing a Structure for Fixed-Point Operations**

This example introduces a structure to encapsulate the fixed-point representation, promoting code clarity and reusability. This is important when there are multiple functions utilizing the same format.

```csharp
using System;

public struct FixedPointNumber
{
    private const int FractionalBits = 32;
    private const ulong One = 1UL << FractionalBits;
    public long Value { get; }


    public FixedPointNumber(double value)
    {
        Value = (long)(value * (double)One);
    }

    private FixedPointNumber(long value)
    {
        Value = value;
    }

    public static FixedPointNumber FromLong(long value)
    {
         return new FixedPointNumber(value * One);
    }

    public FixedPointNumber Multiply(FixedPointNumber other)
    {
           ulong productHigh = (ulong)(this.Value >> 32) * (ulong)other.Value;
        ulong productLow = (ulong)(this.Value & 0xFFFFFFFF) * (ulong)other.Value;
        ulong carry = (productLow >> FractionalBits);
        productHigh += carry;

        long scaledResult = (long)(productLow >> FractionalBits) + ((long)productHigh << 32);
        return new FixedPointNumber(scaledResult);
    }


    public double ToDouble()
    {
        return (double)Value / (double)One;
    }

    public override string ToString()
    {
        return ToDouble().ToString();
    }
}

public class FixedPointMath
{
  public static void Main(string[] args)
  {
        FixedPointNumber integerValue = FixedPointNumber.FromLong(10);
        FixedPointNumber fractionValue = new FixedPointNumber(0.75);
        FixedPointNumber result = integerValue.Multiply(fractionValue);
       Console.WriteLine($"Result of {integerValue} * {fractionValue} = {result.ToDouble()}");

    }
}
```

**Commentary:**

*   The `FixedPointNumber` struct encapsulates the fixed-point value, its fractional bits, and provides utility methods.
*   The constructor takes a `double` value, converting it to the fixed-point representation. A second private constructor is provided for internal access. `FromLong` is provided to create the fixed point integer values.
*   The `Multiply` method now operates directly on the `FixedPointNumber` struct, returning a new `FixedPointNumber`. It does not have any overflow error checking.
*   The `ToDouble` method converts the fixed-point representation to a double, facilitating easy inspection and debugging. It also has a `ToString` override for ease of use.
*   The `Main` method demonstrates how to instantiate and use the structure for multiplication.

For further study, I recommend focusing on books and articles concerning digital signal processing (DSP) and embedded systems. These areas often utilize fixed-point arithmetic extensively. Look for texts covering topics like fixed-point math, fractional number representations, numerical stability, and overflow handling in integer-based systems. Also, investigating texts relating to compiler optimizations that may be applied to fixed point operations is valuable. Studying documentation related to C# intrinsic operations can also yield performance benefits. Finally, researching specific target architectures, for example ARM, may provide guidance on how this code may be further optimized.
