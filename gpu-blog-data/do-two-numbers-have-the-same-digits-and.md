---
title: "Do two numbers have the same digits and length?"
date: "2025-01-30"
id: "do-two-numbers-have-the-same-digits-and"
---
Given two integer values, determining if they possess identical digits irrespective of order and have the same length necessitates a meticulous approach beyond simple numerical comparison. This task, frequently encountered in data validation and algorithmic challenges, hinges on analyzing the constituent digits rather than the numerical value itself.

Fundamentally, the core requirement involves extracting individual digits from each integer, then comparing their frequencies. A direct numerical comparison proves insufficient; for example, the integers 123 and 321 are unequal numerically, yet fulfill the digit-matching criteria. Similarly, 123 and 1234 fail due to differing lengths, while 123 and 213 are valid. Therefore, the process requires two distinct steps: length verification and digit frequency analysis.

I've often found a hash map, or its equivalent structure in different languages, to be the most efficient tool for this task. This data structure enables efficient counting and retrieval of digit frequencies. Consider a scenario involving data validation for unique identification numbers – it became vital to ensure that a newly entered number hadn't been previously used, even if the digit order was different, which highlights the practicality of this problem.

Initially, we determine the length of both integer inputs. If the lengths are not equal, the numbers cannot satisfy the condition, and the process terminates immediately, thus reducing unnecessary computations. This initial check reduces time complexity in the cases where the integers differ considerably in their number of digits. Then, for each number, we extract the individual digits and update a frequency counter. If the frequencies are identical between the two counters then the numbers have the same digits and length.

The primary challenge resides in extracting and maintaining counts of digits efficiently. Direct string conversions, while simple, often introduce performance bottlenecks in large-scale operations. I prefer to use the modulo operator and integer division to extract digits, which leads to more optimized operations in many environments. Another area requiring caution is the handling of negative integers. We can either choose to analyze the absolute values or decide that numbers with differing signs can not fulfil our criteria from the beginning. For this explanation, I've assumed that both integers are non-negative.

Below are code examples in three different programming languages illustrating the solution:

**Example 1: Python**

```python
def same_digits_and_length(num1, num2):
    """
    Checks if two non-negative integers have the same digits and length.
    """
    str_num1 = str(num1)
    str_num2 = str(num2)

    if len(str_num1) != len(str_num2):
        return False

    freq1 = {}
    freq2 = {}
    for digit in str_num1:
        freq1[digit] = freq1.get(digit, 0) + 1
    for digit in str_num2:
        freq2[digit] = freq2.get(digit, 0) + 1

    return freq1 == freq2


#Example usages
print(same_digits_and_length(123, 321)) #Output: True
print(same_digits_and_length(123, 1234)) #Output: False
print(same_digits_and_length(123, 124)) #Output: False
print(same_digits_and_length(122, 212)) #Output: True
```

This Python implementation converts the integers to strings initially. It avoids the modulo and division method for extracting digits. However, the frequency counting is similar to the approach using maps. The function first performs length comparison. Afterward, it constructs a frequency map for each integer, then compares the equality of these maps.

**Example 2: Java**

```java
import java.util.HashMap;
import java.util.Map;

public class SameDigitsAndLength {

    public static boolean sameDigitsAndLength(int num1, int num2) {
         String strNum1 = Integer.toString(num1);
        String strNum2 = Integer.toString(num2);


        if (strNum1.length() != strNum2.length()) {
            return false;
        }

        Map<Character, Integer> freq1 = new HashMap<>();
        Map<Character, Integer> freq2 = new HashMap<>();


        for (char digit : strNum1.toCharArray()) {
             freq1.put(digit, freq1.getOrDefault(digit, 0) + 1);
        }
        for (char digit : strNum2.toCharArray()) {
            freq2.put(digit, freq2.getOrDefault(digit, 0) + 1);
        }
        return freq1.equals(freq2);
    }
    public static void main(String[] args) {
        System.out.println(sameDigitsAndLength(123, 321)); // Output: true
        System.out.println(sameDigitsAndLength(123, 1234)); // Output: false
        System.out.println(sameDigitsAndLength(123, 124)); // Output: false
        System.out.println(sameDigitsAndLength(122, 212)); // Output: true
    }
}
```

The Java code also uses the String conversion for simplicity. It utilizes `HashMap` to track the digit frequencies. This version performs the same core logic, first checking the length then comparing the frequency maps, as in the Python implementation. Using `toCharArray()` provides a convenient way to iterate through the characters.

**Example 3: JavaScript**

```javascript
function sameDigitsAndLength(num1, num2) {
    const strNum1 = String(num1);
    const strNum2 = String(num2);

    if (strNum1.length !== strNum2.length) {
        return false;
    }

    const freq1 = {};
    const freq2 = {};

    for (const digit of strNum1) {
        freq1[digit] = (freq1[digit] || 0) + 1;
    }
     for (const digit of strNum2) {
        freq2[digit] = (freq2[digit] || 0) + 1;
    }

    return JSON.stringify(freq1) === JSON.stringify(freq2);
}
// Example usages
console.log(sameDigitsAndLength(123, 321)); // Output: true
console.log(sameDigitsAndLength(123, 1234)); // Output: false
console.log(sameDigitsAndLength(123, 124));  // Output: false
console.log(sameDigitsAndLength(122, 212));  // Output: true
```

In JavaScript, similar to the Python and Java implementations, string conversion is performed initially. Objects are used as frequency counters, which is idiomatic to JavaScript. Stringifying the objects allows us to directly compare them for equality, as JavaScript object equality checks for identity, not structural equality. The core logic of length check and frequency comparison remains the same.

For further learning and improvement on this type of problem, I recommend focusing on discrete mathematics resources emphasizing permutations, combinations, and hashing algorithms, alongside textbooks covering algorithms and data structures. Understanding the underlying time complexity of your code is crucial, especially if you will be handling large datasets, and also exploring different data structures based on different language. Resources focused on the specific data structures used in each language are also beneficial. The key is to practice writing the algorithms using different approaches to gain deeper insight into their runtime characteristics.
