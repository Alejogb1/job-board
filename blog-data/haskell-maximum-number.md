---
title: "haskell maximum number?"
date: "2024-12-13"
id: "haskell-maximum-number"
---

 so you're asking about finding the maximum number in Haskell right Yeah I've been there man it's one of those things that seems simple on the surface but you can really dig in and find some interesting stuff under the hood depending on what you mean by "maximum number"

First off let's assume you're talking about a list of numbers something like `[1 5 2 9 3]` or maybe something floating point like `[3.14 2.71 1.61]` Or maybe you have a list of a custom type that you can compare or even a data structure you have to go through It can vary

Now a super straightforward way in Haskell to get the maximum from a list of numbers is with the `maximum` function It's in the `Prelude` so no need to import anything

```haskell
myList :: [Int]
myList = [1, 5, 2, 9, 3]

maxNumber :: Int
maxNumber = maximum myList

main :: IO ()
main = print maxNumber -- Output: 9
```

Pretty simple stuff right This is a classic beginner case if you're familiar with other languages you see that it does basically the same thing the magic lies in that its already defined

But what if you have an empty list You get an error the `maximum` function doesn't handle that it's undefined for empty lists So you need to handle that case like some form of validation

```haskell
safeMaximum :: (Ord a) => [a] -> Maybe a
safeMaximum [] = Nothing
safeMaximum xs = Just (maximum xs)

main :: IO ()
main = do
    print (safeMaximum [1, 5, 2, 9, 3])  -- Output: Just 9
    print (safeMaximum [])           -- Output: Nothing
```

Here I introduced `Maybe` which is your friend in Haskell for dealing with situations where a value may or may not exist This is more or less how I handle all cases

I've seen plenty of people jump directly into fold patterns for this but i think there are more clear ways to handle it like the `Maybe` monad This is definitely a more Haskell like way of handling such cases and its less likely to break

  so now let's talk about something more obscure Lets say you don't have a list but a bunch of things you want to compare on the fly In that case you would use the max function that comes with Ord which is a typeclass for types that can be ordered

```haskell
myMax :: (Ord a) => a -> a -> a
myMax a b = max a b

main :: IO ()
main = do
    print (myMax 10 5) -- Output: 10
    print (myMax 3.14 2.71) -- Output: 3.14
    print (myMax "apple" "banana") -- Output: "banana"
```

You see how the types are inferred by the compiler if you dont give explicit types You dont always need types which is great

 so here's a story from my past life I once had to process some sensor data from a bunch of IoT devices I was getting these streaming of data point and my job was to get the maximum of the last N sensor values in a rolling window. I didn't want to store everything just the last N numbers and i didnt want to store all the values so I did a circular buffer implementation I tried some functional approaches first but then I had to use some dirty mutability for performance. it was not fun I should have done it in a different language honestly.

So you know the `maximum` function it has a complexity of O(n) if we think about lists and we dont have any better way to process it for each point we would take each point and get all N elements of the array and call the max function that would be O(N*n) on a stream which is not optimal if you are using a streaming algorithm so a better approach is just keeping track of the max as you go This is called a max heap and can do all operations in O(log(n)) for each operation

Another thing that you should be aware of is floating point numbers there's always a gotcha I had issues with floating point comparison because of precision errors floating point numbers are not precise and they are not what you think they are and it got me into a lot of issues I would say you need to read up the section on numerical precision in "Numerical Recipes" by Press et al this is a crucial part of any computation. If your using floating point numbers in any context you have to read this book. I mean you can ignore but then you will have weird bugs that you cannot debug.

There are also parallel ways to compute maximum value but I feel like for your question it's not worth going into this territory. You can always read "Parallel and Concurrent Programming in Haskell" by Simon Marlow for more information about concurrency in Haskell. It's not for the faint of heart though.

And look sometimes you just have to deal with the fact that finding a max of anything can be a bit tedious its not always that simple If you are thinking the max value is always there and is always a good idea it can be that you are doing it wrong

Anyways to wrap this up if you have an ordered list use the `maximum` function. If you have an empty list deal with `Maybe` . if you have to compare two values use the `max` function from `Ord`. If it's not a simple list you may have to write your own algorithm but usually you will get the result faster and more readable with these methods. And keep in mind floating point arithmetic can be a problem and try to read "Numerical Recipes" by Press et al. Also never try to parse an HTML with Regex because everyone knows that Regex is not for parsing HTML. If you try to do that it will create some Cthulhu like monsters

Hope that helps I'm here if you have any other questions
