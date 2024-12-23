---
title: "how does one perform the exp operation element wise in juila?"
date: "2024-12-13"
id: "how-does-one-perform-the-exp-operation-element-wise-in-juila"
---

 so you need to element-wise exponentiate in Julia right Been there done that Seems like a simple question but it trips up a lot of folks new to Julia And frankly I’ve seen some pretty horrific code trying to get around this one

Let me tell you about this one project I worked on back in the day It was some signal processing stuff involving large arrays I was using NumPy in Python at the time and honestly the syntax was so clean it was one of the only things that worked I had a python notebook going and all that with `np.exp(my_array)` doing the thing Then I was told by my boss that I had to port it to Julia because "performance" yeah yeah So I jumped right into Julia thinking it would be the same right I naively just tried `exp(my_array)` and boom error city

Julia wants to be told when you need element-wise operations It’s explicit about it which is actually pretty awesome once you wrap your head around it but the initial shock was real And that’s where the dot comes in That sweet sweet dot

So the key here is the dot syntax It's not a typo or some weird punctuation quirk it's Julia's way of saying "hey I want to apply this function to each element of this array or matrix or whatever iterable"

So instead of `exp(my_array)` you want `exp.(my_array)` See that dot after the function name That’s the magic

Let's walk through a few examples and you can play with them yourself

First a simple 1D array

```julia
my_array = [1 2 3 4 5]
result = exp.(my_array)
println(result)
```

That'll spit out the exponential of each element as expected. No loops needed no fuss just clean and efficient code

Now for a 2D matrix because why not

```julia
my_matrix = [1 2 3; 4 5 6; 7 8 9]
result_matrix = exp.(my_matrix)
println(result_matrix)
```

Same principle applies It works beautifully for matrices as well Each element is exponentiated on its own

And for the people that want to get really crazy with it let’s try it with a more complex case say you have a custom type

```julia
struct ComplexNumber
    real::Float64
    imaginary::Float64
end
function my_exp(cn::ComplexNumber)
    return ComplexNumber(exp(cn.real) * cos(cn.imaginary) ,exp(cn.real)*sin(cn.imaginary) )
end
my_complex_array= [ComplexNumber(1.0,1.0) , ComplexNumber(2.0,2.0), ComplexNumber(3.0,3.0)]
result_complex = my_exp.(my_complex_array)
println(result_complex)
```

Again the dot operator will automatically call your custom `my_exp` function for each custom element in the array You can see how this is helpful when working with structs that are not the typical floats and ints of standard numerical computing you can easily define operations in your custom types and apply them all at once

So basically any function you can write Julia can "vectorize" it using the dot operation You can even chain these operations together

```julia
my_array = [1 2 3]
result_chained = sin.(exp.(my_array))
println(result_chained)
```

This calculates the exponential and then the sine of all the array components and it does so efficiently without explicit for loops

The dot is also a broadcast operator that is it's not just for functions with one array input it works for two inputs and you can combine different shapes via broadcasting if they are broadcast compatible For example lets try power operation

```julia
my_array = [1 2 3]
my_powers = [2 3 4]
result_pow = my_array.^my_powers
println(result_pow)

my_scalar = 2
result_pow_scalar = my_array.^my_scalar
println(result_pow_scalar)
```

The first one is each element of my_array is raised to the power of the respective element of my_powers

The second one is a vector is raised to scalar power using broadcasting

Now the really interesting thing about this in terms of performance which you mentioned earlier is that this is really good for performance This dot syntax isn't just for syntactic sugar It's baked into Julia's core It gets compiled down to super optimized machine code so you're not paying some crazy performance penalty for using the nice syntax

Another thing to note if you want to avoid the overhead of allocation for your output you can use the in place operations which modify the array that is passed into the function instead of allocating a new array

```julia
my_array = [1.0 2.0 3.0]
exp!.(my_array)
println(my_array)
```

the "!" at the end indicates the function will modify its inputs

This is great if you're working with large arrays and you don't want to constantly allocate memory. Note that the "!" version of a function needs to exist for this to work and not every function has an in-place version available

And that my friend is it It’s not just `exp` you can use the dot notation on almost any function in Julia for element-wise operations so it is useful to learn it

I know its just simple `exp` we are doing here but these things become so important when you are using linear algebra or more complicated things in your code when you need to do element wise operations and this knowledge you gain here will serve you in those more complicated situations

Now resources for you to learn Julia and more about its amazing way to vectorize your programs Let me recommend a couple of things

For general Julia learning the "Think Julia" book is a fantastic resource it's free online it's well-written and it covers all the basics and then some You might find it at https://benlauwens.github.io/ThinkJulia.jl/latest/book.html it's kind of a no brainer if you want to get into Julia You should probably start from there

And then if you want to go a bit deeper into Julia’s performance and how this dot syntax actually works I would recommend you to go through the Julia Documentation its actually really good you can easily google search julia docs or go through https://docs.julialang.org/en/v1/

Oh and one more thing since we're talking about Julia performance did you hear about the programmer who got lost in the documentation He kept going in circles because he got into an infinite loop haha

  I’m done But yeah that dot its your friend remember it for all the element-wise stuff
