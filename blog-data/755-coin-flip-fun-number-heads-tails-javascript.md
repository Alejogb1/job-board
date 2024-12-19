---
title: "7.5.5 coin flip fun number heads tails javascript?"
date: "2024-12-13"
id: "755-coin-flip-fun-number-heads-tails-javascript"
---

Okay so a coin flip in javascript right Been there done that and probably broke a few keyboards in the process It sounds simple and it is really once you get the hang of it But there are a few things to watch out for especially if you are coming from a different language background

Let’s be frank this isn’t rocket science or even a hard interview question Its basically a random number generator thingy that gives you two options 0 or 1 in this case we interpret 0 as tails and 1 as heads The key is getting that random number generation correct and handling it nicely in your code

So first let’s talk about my history with this simple problem I remember back in my early days like maybe 2012 I was working on this really janky browser game It had a lot of randomness and yes it needed a coin flip function I kid you not I used Mathrandom() like everyone else initially but then I ran into this weird bias in the distribution of outcomes My flips were landing heads like 60% of the time and that was not fun or fair I debugged like a madman for hours thinking there was something wrong with my game logic Turns out my usage of Mathrandom and bit operations were not that great My lesson learned was that even the smallest of seemingly trivial things can screw things up in a big way so paying attention to detail matters a lot

Okay enough of the reminiscing Lets dive into the code First the basic version which most people use and probably the one you were trying to write yourself

```javascript
function coinFlipBasic() {
  return Math.random() < 0.5 ? "heads" : "tails";
}

console.log(coinFlipBasic())
console.log(coinFlipBasic())
console.log(coinFlipBasic())
console.log(coinFlipBasic())
```

This works of course Its simple to the point but not very robust Its using the built in `Mathrandom()` which gives you a floating point number between 0 and 1 we are checking if that is less than 05 if so heads otherwise tails Simple as that but as I mentioned earlier there might be subtle biases depending on how `Mathrandom()` is implemented in the browser so better to understand the limits of things

Now lets step up a notch lets think about a slightly better approach I like to use integer random numbers and then map that to heads or tails Using modulo is actually a pretty good practice to keep the logic clear and predictable I find this more controllable than dealing with floating point numbers in this particular case

```javascript
function coinFlipInteger() {
  const randomInt = Math.floor(Math.random() * 2);
  return randomInt === 0 ? "tails" : "heads";
}

console.log(coinFlipInteger())
console.log(coinFlipInteger())
console.log(coinFlipInteger())
console.log(coinFlipInteger())
```

This is more robust in my experience This version uses the same random number generator but first converts the result into a whole number 0 or 1 Its less likely to suffer from slight biases I mentioned before because the actual floating point output of Mathrandom is only used to generate an integer it wont matter that much if there were some weird subtle biases in the output distribution

Okay so its important to always be skeptical especially in randomness If you need higher quality random numbers because your application is more sensitive for fairness like a lottery or something like that you can check out a technique called using a cryptographically secure random number generator You may not need it for a simple coin flip but its something good to have in the toolkit

For the sake of showing off here is how you would use `crypto.getRandomValues` to get a more secure outcome

```javascript
function coinFlipSecure() {
  const randomArray = new Uint8Array(1);
  crypto.getRandomValues(randomArray);
  return randomArray[0] % 2 === 0 ? "tails" : "heads";
}

console.log(coinFlipSecure())
console.log(coinFlipSecure())
console.log(coinFlipSecure())
console.log(coinFlipSecure())
```

This is basically getting random bytes directly from the operating system its much more robust but its definitely an overkill for coin flipping If you want to get more info on this type of random numbers the book "Cryptography Engineering" by Niels Ferguson et al is a solid reference it dives into a ton of details about how these systems actually work at the low level stuff Also the RFC document RFC 4086 "Randomness Recommendations for Security" gives insights into how to handle pseudo random numbers so that you can avoid subtle pitfalls in your code

So now we have 3 example options a basic one that works most of the time a more robust integer based version and a crypto based one For a simple coin flip I always recommend the integer version unless you really really need super secure randomness (that is rarely the case in basic applications) Its a good balance of simplicity and reliability

So yeah that’s basically my history and thoughts on the coin flip in javascript Just to recap use `Math.random()` but be aware of its limitations its sometimes a bit biased especially in older browsers If you are doing anything important you should use the modulo way or if you are doing some security related work consider using the crypto APIs and never ever underestimate the simplicity of the problem it can bite you in unexpected places in the future I still see some people making mistakes with this simple concept even after all these years it just shows its important to understand the fundamentals

Finally one last thing if you are having issues with Mathrandom behaving badly make sure to not have other libraries conflicting with it i saw a case once where a badly written library overrode Mathrandom in a sneaky way this was a long day for me but hey we all learned from that day I think I had more hair back then haha that's the only joke I can put in here
