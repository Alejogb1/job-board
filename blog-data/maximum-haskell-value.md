---
title: "maximum haskell value?"
date: "2024-12-13"
id: "maximum-haskell-value"
---

so you're asking about getting the maximum value in Haskell right Been there done that so many times

First off let's be clear you're not gonna get "maximum" in some absolute sense like an integer that's bigger than every other integer Haskell like most languages has limits You're asking how to find the biggest element within some data structure or a set of values thats what its about right

My first rodeo with this was during my undergrad when I was building a program to analyze network traffic I had a huge list of packet sizes and I needed the biggest one to figure out some bottleneck stuff Back then I was barely awake trying to juggle lectures projects and sleep so I made some mistakes and had to debug late nights

The simplest case you're probably thinking of is finding the max from a list of numbers Haskell has the `maximum` function for that Straight up its in the prelude so you dont need to import anything

```haskell
myNumbers :: [Int]
myNumbers = [1, 5, 2, 8, 3]

biggestNumber :: Int
biggestNumber = maximum myNumbers -- output 8
```

Boom Done You can do the same with floats or any type that is an instance of the `Ord` typeclass which basically means you can compare them with operators like `<` and `>` So `maximum` is your workhorse here

But lets say you're dealing with something more complicated Maybe you have a list of custom types and you need to find the maximum based on a specific field This is when the party gets interesting

Lets imagine you have a type representing a product with a price and name

```haskell
data Product = Product { productName :: String
                     , productPrice :: Double }
                     deriving (Show)

products :: [Product]
products = [Product "Laptop" 1200.0, Product "Mouse" 25.0, Product "Keyboard" 75.0, Product "Monitor" 300.0]

```

Now you want the most expensive product How do you tell `maximum` what to use for comparison Its about what criteria you want to use to compare not just about a simple number compare

We need to use `maximumBy` instead which lets you specify a comparison function This function should take two values of your custom type and return `Ordering` which can be `LT`, `GT` or `EQ` which stands for less than greater than or equal to respectively

Here is the solution using `comparing` from `Data.Ord` which is usually the way to go its way cleaner

```haskell
import Data.Ord (comparing)
import Data.List (maximumBy)

mostExpensiveProduct :: Product
mostExpensiveProduct = maximumBy (comparing productPrice) products -- output Product {productName = "Laptop", productPrice = 1200.0}
```

So `comparing productPrice` makes a function that compares products by their price it basically constructs the comparison function for you.

 lets say you are dealing with something completely different You are working with monads and you want to find the "maximum" within a monadic context like for example `Maybe` lets say you have a `Maybe Int` and want the biggest element or `Nothing` if its empty This one is easy if you use the `optional` monad transformer from `transformers`

```haskell
import Control.Monad.Trans.Maybe (MaybeT)
import Control.Monad (foldM)
import Data.Maybe (fromMaybe)

maybeMax :: MaybeT [] Int
maybeMax = foldM max (MaybeT $ return Nothing) (map (MaybeT . return . Just) [1,5,2,8,3])
```

This snippet uses `foldM` which performs a monadic fold and max is the simple max function It takes each element from the list transforms into a `MaybeT` value. After that it uses `foldM` and the `max` function to do the comparison This will return a `MaybeT [] Int` value and you can simply use fromMaybe to get the value if you are sure it is not empty for example or to give a default value

Now here's the thing people ask about using `maximum` with empty lists If you give an empty list to `maximum` directly it'll throw an exception Because well there's nothing to compare Right?

So you need to make sure the data structure is not empty before giving it to `maximum` you could filter it first or handle the case with something like `maybe` its a common pattern you'll use all the time but if you try to give an empty list to max I will come to your house and delete all your files no just kidding but it will fail for sure So be aware

The biggest thing that usually gets people confused here is thinking that maximum is going to apply in the same way in every single situation especially when your data type changes That was my issue with the network packet analysis I was comparing the wrong fields by a mistake in my comparison function

For resources you definitely want to check out "Real World Haskell" by Bryan O'Sullivan Don Stewart and John Goerzen if you are still fresh on Haskell Its a classic book that covers all the important things I would say its a must read.

Also "Thinking Functionally with Haskell" by Richard Bird is a great resource for understanding functional programming concepts in depth which really helps to understand what is going on under the hood specially when you are dealing with more complex situations or custom data types

Also the Haskell report is helpful it has the documentation of the language with all the details of all the functions and types so it is a good reference to have

Lastly another important tip dont overthink it just use the functions already available in the prelude usually the solution is right in front of you especially for most common operations like calculating the max value

And remember no matter how many times you have dealt with this problem you will still be caught up sometimes by silly issues So keep debugging and practicing and never give up
