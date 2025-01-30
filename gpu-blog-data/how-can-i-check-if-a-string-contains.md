---
title: "How can I check if a string contains a digit between two other digits?"
date: "2025-01-30"
id: "how-can-i-check-if-a-string-contains"
---
A frequent challenge in data validation involves confirming the presence of a numeric character sandwiched between other numeric characters within a string. This isn't a straightforward string containment check, but requires a pattern-matching approach. I've encountered this repeatedly when parsing user-input fields that mandate a specific format, such as an identifier with embedded sequence numbers.

Fundamentally, the task necessitates evaluating the string's character sequence for a pattern where a digit is preceded and followed by another digit. Simple string methods like `indexOf` or `contains` are insufficient, as they only confirm the existence of single characters or subsequences, not specific arrangements. Regular expressions (regex) provide the requisite power for this kind of contextual matching. The approach I employ leverages regex syntax to define this three-digit pattern precisely.

The core of the solution hinges on constructing a regular expression that specifies: a digit, followed by another digit, followed by another digit. The common regex syntax for a digit is `\d`, and we need three of these in succession. Thus, the regex pattern `\d\d\d` is a good starting point. However, the given problem asks for *a* digit between *two* other digits; the use of three consecutive digits does not directly enforce this constraint. `\d\d\d` would, therefore, match "123", but not "a12b" or "a1b".

To match any character *preceding* a digit and *following* it, we can use the concept of lookbehind and lookahead assertions. However, these are not universally supported or can impact performance. Therefore, a practical approach avoids lookarounds. We can modify the regex to match an arbitrary character, followed by a digit, followed by another digit, followed by a digit and an arbitrary character `.\d\d\d.`. To be more precise and avoid cases like "1234", we need to ensure that the adjacent characters are not digits. Thus we'd need to specify that preceding and trailing characters are not digits using the negative character set `\D`. This gives us the pattern `\D\d\d\d\D` which now accurately describes the context of having a digit in between two digits.

Here's how this plays out in various programming languages:

**Example 1: JavaScript**

```javascript
function hasDigitBetweenDigits(inputString) {
  const regex = /\D\d\d\d\D/;
  return regex.test(inputString);
}

console.log(hasDigitBetweenDigits("abc123def"));  // true
console.log(hasDigitBetweenDigits("123"));      // false
console.log(hasDigitBetweenDigits("a12b"));     // false
console.log(hasDigitBetweenDigits("a123b"));    // true
console.log(hasDigitBetweenDigits("1abc2"));    // false
console.log(hasDigitBetweenDigits("12ab"));      //false
console.log(hasDigitBetweenDigits("ab12"));      //false
console.log(hasDigitBetweenDigits("a1b2c"));    //false
console.log(hasDigitBetweenDigits("a1a2a"));    //false
console.log(hasDigitBetweenDigits("1234"));     //false
console.log(hasDigitBetweenDigits("12abc"));    //false
console.log(hasDigitBetweenDigits("abc12"));    //false
```

In this JavaScript example, the `hasDigitBetweenDigits` function takes a string as input. It constructs a regex using the pattern `/D\d\d\d\D/`, and then employs the `test` method to check if the string matches this pattern. The function returns `true` if a match is found, and `false` otherwise. Crucially, only when the entire pattern (non-digit, digit, digit, digit, non-digit) is present does it return true. The output demonstrates different edge cases to explain the behavior.

**Example 2: Python**

```python
import re

def has_digit_between_digits(input_string):
    regex = re.compile(r"\D\d\d\d\D")
    return bool(regex.search(input_string))

print(has_digit_between_digits("abc123def"))  # True
print(has_digit_between_digits("123"))      # False
print(has_digit_between_digits("a12b"))      # False
print(has_digit_between_digits("a123b"))      # True
print(has_digit_between_digits("1abc2"))     # False
print(has_digit_between_digits("12ab"))     # False
print(has_digit_between_digits("ab12"))      # False
print(has_digit_between_digits("a1b2c"))    # False
print(has_digit_between_digits("a1a2a"))    # False
print(has_digit_between_digits("1234"))     # False
print(has_digit_between_digits("12abc"))    # False
print(has_digit_between_digits("abc12"))    # False
```

In the Python example, I utilize the `re` module for regular expressions. The `has_digit_between_digits` function takes the input string, compiles the regex with the same pattern `r"\D\d\d\d\D"`, and then employs the `search` method to find a match anywhere within the string. The `bool()` casting ensures that `search()` output (which will return a match object or none) translates into `True` or `False` return value. This approach performs the exact same validation with same outputs as the JavaScript variant.

**Example 3: Java**

```java
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DigitBetweenDigits {
    public static boolean hasDigitBetweenDigits(String inputString) {
        Pattern pattern = Pattern.compile("\\D\\d\\d\\d\\D");
        Matcher matcher = pattern.matcher(inputString);
        return matcher.find();
    }

    public static void main(String[] args) {
        System.out.println(hasDigitBetweenDigits("abc123def"));  // true
        System.out.println(hasDigitBetweenDigits("123"));      // false
        System.out.println(hasDigitBetweenDigits("a12b"));     // false
        System.out.println(hasDigitBetweenDigits("a123b"));    // true
        System.out.println(hasDigitBetweenDigits("1abc2"));    // false
        System.out.println(hasDigitBetweenDigits("12ab"));      //false
        System.out.println(hasDigitBetweenDigits("ab12"));      //false
        System.out.println(hasDigitBetweenDigits("a1b2c"));    //false
        System.out.println(hasDigitBetweenDigits("a1a2a"));    //false
        System.out.println(hasDigitBetweenDigits("1234"));     //false
        System.out.println(hasDigitBetweenDigits("12abc"));    //false
        System.out.println(hasDigitBetweenDigits("abc12"));    //false
    }
}
```

The Java implementation utilizes the `java.util.regex` package. Similar to the previous examples, `hasDigitBetweenDigits` constructs a `Pattern` object from the regex pattern `"\\D\\d\\d\\d\\D"` (note that we need double backslashes since backslash is also used as an escape character in Java strings). Then a `Matcher` is obtained from the input String. The `find()` method is then used to check for any match. As before, the same validation and output is achieved.

These code examples demonstrate how this pattern can be implemented across several languages. The core logic remains consistent: defining a suitable regex pattern using non-digit and digit character classes, and then applying that regex to determine string containment.

For further study, consider delving into these areas. First, familiarize yourself with the specific regex syntax of your language of choice; slight differences might exist, especially in escape character handling. Second, explore the performance implications of different regex constructs. While `\D` and `\d` are typically very efficient, more complex assertions might introduce performance overhead. Finally, practical application is key. Develop multiple test cases that cover the spectrum of possible valid and invalid inputs to ensure the regex is robust in production.

In conclusion, validating the presence of a digit between two other digits requires a regex solution. Using the pattern `\D\d\d\d\D` and the approaches presented here, one can reliably assess strings for this specific pattern in code. This method, while concise, provides the necessary rigor for data validation scenarios.
