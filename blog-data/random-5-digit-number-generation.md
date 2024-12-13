---
title: "random 5 digit number generation?"
date: "2024-12-13"
id: "random-5-digit-number-generation"
---

Okay I've been wrestling with random number generation for what feels like a lifetime honestly it's one of those deceptively simple things that gets complicated fast I’ve seen my share of janky implementations over the years from undergraduate projects to real-world production code where bugs lurk like gremlins waiting to strike

So let’s talk about 5-digit random numbers specifically yeah that’s what you asked for straightforward enough right Well not quite The first thing everyone thinks of is just slapping something together like `rand() % 90000 + 10000` in C or whatever similar in your language of choice And I'm guilty of doing exactly that on a particularly sleep-deprived all-nighter trying to debug a segmentation fault related to some memory buffer over flow related to my university assignment where random numbers were required but I completely glossed over the quality of them. I remember waking up the next day thinking this is not gonna work.

But here's the deal `rand()` isn’t exactly known for its cryptographic security or even its uniform distribution especially the older implementations can have some weird patterns that can ruin stuff in your code. It’s basically pseudo-random number generation or PRNG that’s important a computer can’t really generate true random numbers it uses algorithms that appear random. You want something that at least passes some statistical tests not something that always outputs numbers within the same small range every 1000 calls right

So what's the alternative Well it really depends on your requirements If you are coding up something simple and aren't too worried about the quality for testing purposes or just getting a quick prototype done and dusted sure you can go with `rand()` it might be enough but if you are building anything that matters like for instance a simulation of say an auction or something where those numbers would be important don't even think of using the `rand()` function.

Let me show you some code and you can see for yourself how to get good pseudo-random numbers

```cpp
#include <iostream>
#include <random>

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(10000, 99999);

    for (int i = 0; i < 5; ++i) {
        std::cout << distrib(gen) << std::endl;
    }
    return 0;
}
```

This C++ example uses the `<random>` header which is awesome. It's the modern way to do PRNG in C++. We use `std::random_device` to get a seed from the system which is usually good then `std::mt19937` that’s the Mersenne Twister engine a popular PRNG algorithm and it’s pretty reliable finally `std::uniform_int_distribution` makes sure the random numbers are evenly distributed between 10000 and 99999

I had this project a couple of years ago where I had to generate random order IDs for an e-commerce system. The first version of the code did something simple like the `rand()` method you know the old sin But this caused a few issues It was noticed that a lot of the generated IDs had patterns it would output a sequence of similar IDs for some period of time then it would output another completely different sequence and so on. It was quite problematic. This made debugging and tracking issues a nightmare. So I had to completely refactor the whole ID generation system using the approach like the above and that problem never came up again. It made me realize the devil is truly in the details of random number generation

Moving on here's how to do similar stuff in Python:

```python
import random

def generate_random_5_digit():
    return random.randint(10000, 99999)

if __name__ == "__main__":
    for _ in range(5):
        print(generate_random_5_digit())
```

Python’s `random` module is fairly solid. `random.randint` gives us the nice uniform distribution we want. Python handles seeding behind the scenes so generally we can just go straight to `randint`. I've used this to create some fake data in simulations and it worked great. No patterns just good random numbers so far. The old `random()` way always had some issues with generating good unique random numbers especially in parallel I saw some bugs that were really hard to trace once. My old boss used to say "randomness is not a get out of jail free card". I completely agree with him.

Now you might be thinking is there anything else I can do I mean we are using good PRNGs right? Well you can but it's not usually necessary for what we are doing here. But let me show you something using Java.

```java
import java.security.SecureRandom;
import java.util.Random;

public class RandomNumberGenerator {

    public static void main(String[] args) {
        SecureRandom secureRandom = new SecureRandom();
        Random random = new Random();
        
        for(int i=0; i<5; i++){
           int randomNumber = 10000 + secureRandom.nextInt(90000);
           System.out.println(randomNumber);

            
            int randomNumberFromRandom = 10000 + random.nextInt(90000);
            System.out.println(randomNumberFromRandom);
        }
       
    }
}
```

This Java example introduces `SecureRandom` which is a cryptographically strong random number generator. In this case I've also included `Random` which is the default Java PRNG. As you can see when using the `SecureRandom` in some cases this can take a bit more time to generate that's because it's doing some more complex math under the hood. It uses a different algorithm to generate the numbers and it's more suitable for situations where you need to avoid any predictability. I had a colleague who was working with lottery numbers in his spare time ( don't ask ) who insisted we use this approach. He kept saying *I'm not losing on the lottery with the garbage random numbers I learned in school*. It was kinda funny to be honest. Anyways, it depends on what you're trying to do. Do not use this for video game character generation it will be slow.

So the key here is understanding that not all random number generators are created equal. If you just need something to throw together in a quick prototype or some dummy data then using `rand()` or `random()` will be fine, but if you need something more robust and reliable especially for any kind of security related stuff or even a decent statistical analysis use the more powerful approaches like the examples above.

There are plenty of resources to explore further the world of random numbers and its nuances if you want to dig deeper. I recommend “Numerical Recipes” by Press et al. it’s a classic for a reason. It covers tons of different PRNG algorithms and statistical tests. For a more practical approach “Programming Pearls” by Jon Bentley has a very useful chapter on random number generation. If you are interested in more mathematical side then “The Art of Computer Programming Vol 2” by Knuth will give you all the details you will ever need.

So yeah that's my spiel on 5-digit random numbers. Don’t fall for the simple `rand()` trap and actually think about what type of randomness you need. That’s my two cents. Good luck with your coding!
