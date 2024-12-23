---
title: "5.14 lab convert to reverse binary zybooks?"
date: "2024-12-13"
id: "514-lab-convert-to-reverse-binary-zybooks"
---

 so you're wrestling with that Zybooks 514 lab the one where you gotta flip an integer into its reverse binary representation I feel you dude I’ve been there done that got the t-shirt and probably a few obscure compiler error messages tattooed on my soul This is like classic CS101 but with a Zybooks twist I remember the first time I saw this kind of problem back in my undergrad days I spent a good two hours debugging a single line of code I was using some weird bit manipulation trick I thought I was being clever spoiler alert I wasn't I ended up doing it the simple way in the end lesson learned right

Anyway let's break this down You basically need to take an integer like say 25 which is 11001 in binary and turn it into 10011 which in decimal is 19 Now the trick is not to do it in decimal conversion and decimal reversal we are going binary all the way down This means you'll be dealing with bits and bitwise operations which are your best friends here  here's a C++ implementation I'd go with if I had to redo it today this is pretty straightforward and avoids those fancy bit tricks I was trying before

```cpp
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>


std::string toBinary(int n) {
    if (n == 0) return "0";
    std::string binary = "";
    while (n > 0) {
        binary = (n % 2 == 0 ? "0" : "1") + binary;
        n /= 2;
    }
    return binary;
}

int binaryToDecimal(std::string binary) {
    int decimal = 0;
    int power = 0;
    for (int i = binary.size() - 1; i >= 0; --i) {
        if (binary[i] == '1') {
            decimal += pow(2, power);
        }
        power++;
    }
    return decimal;
}


int reverseBinary(int n) {
    std::string binary = toBinary(n);
    std::reverse(binary.begin(), binary.end());
    return binaryToDecimal(binary);
}

int main() {
  int number;
  std::cin >> number;
    std::cout << reverseBinary(number) << std::endl;
    return 0;
}
```
Let's explain this hunk of code I think the toBinary method is pretty self explanatory it just takes the integer and then turns it to the corresponding binary string the binaryToDecimal also is very straightforward it receives the string and turns it into the decimal representation of the binary number and lastly the reverseBinary method is the one doing the heavy lifting It first converts the number to binary then uses the standard library to reverse the string representation and finally converts the string back to decimal that's all there is to it this works because string manipulation is easier than trying to reverse bits directly in an int and it is very readable and maintainable I recommend to avoid the overly clever bit shifting and masking that might work if you really want to get the lowest levels of performance but not for a simple lab assignment and believe me I have seen those kinds of issues in my career not good.

Now let's say you are a Python kind of person which is perfectly fine Python rocks too here's the Pythonic version of the same logic

```python
def to_binary(n):
    if n == 0:
        return "0"
    binary = ""
    while n > 0:
        binary = ("0" if n % 2 == 0 else "1") + binary
        n //= 2
    return binary

def binary_to_decimal(binary):
    decimal = 0
    power = 0
    for bit in reversed(binary):
        if bit == '1':
            decimal += 2 ** power
        power += 1
    return decimal


def reverse_binary(n):
    binary = to_binary(n)
    reversed_binary = binary[::-1]
    return binary_to_decimal(reversed_binary)

if __name__ == "__main__":
    number = int(input())
    print(reverse_binary(number))
```

The logic here is essentially the same as in the C++ code just expressed in Python syntax the string reversal is done using slicing a very common idiom in Python and the rest should be self explanatory but the real power of python is the expressiveness and the speed in doing the coding but also python can be easily extended so if I really needed to go low level I could use C/C++ to perform the bit operations and then import that into python for more performance this is what makes python great

Finally just to show this works with another language I'll throw in a Javascript version cause why not

```javascript
function toBinary(n) {
    if (n === 0) return "0";
    let binary = "";
    while (n > 0) {
        binary = (n % 2 === 0 ? "0" : "1") + binary;
        n = Math.floor(n / 2);
    }
    return binary;
}

function binaryToDecimal(binary) {
    let decimal = 0;
    let power = 0;
    for (let i = binary.length - 1; i >= 0; i--) {
        if (binary[i] === '1') {
            decimal += Math.pow(2, power);
        }
        power++;
    }
    return decimal;
}

function reverseBinary(n) {
    let binary = toBinary(n);
    let reversedBinary = binary.split("").reverse().join("");
    return binaryToDecimal(reversedBinary);
}


const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  readline.question('', number => {
    console.log(reverseBinary(parseInt(number)));
    readline.close();
  });
```

This javascript version is almost the exact same code as the python version there's just minor syntactic variations that should be easy to follow now if we are talking of performance of course the C++ version would be better because it is compiled directly into machine code however this is not something that you need to consider for this kind of assignment. I remember one time a colleague tried to use a very complex lookup table to solve this kind of problem it turned out to be faster but in a very small amount of time and the code was completely unreadable don't do that simple is better and more important readable is also better this I learned the hard way

And just because we have to have a joke in this thing how do you comfort a JavaScript bug you console it ha ha anyway sorry I will go now

 now for resources if you really want to dig deep I recommend the following first "Hacker's Delight" by Henry S Warren Jr this book is like the bible for bit manipulation you'll find everything you need to know about bitwise operations and more second "The Art of Computer Programming Volume 1" by Donald Knuth yeah yeah I know its huge but the first volume covers a lot of fundamentals including binary representation and basic algorithms if you want to learn how a computer actually works this is the place and third "Structure and Interpretation of Computer Programs" by Abelson and Sussman it's a classic for a reason it teaches you programming from a very fundamental perspective this should be enough to tackle any problem like this one

Anyways remember the key here is to be systematic and break down the problem into smaller manageable pieces and keep it simple at all times avoid getting fancy unless it is strictly necessary because there is nothing like debugging a single line of overly clever code for hours I hope that helps and happy coding and stay simple.
