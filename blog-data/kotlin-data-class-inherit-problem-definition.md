---
title: "kotlin data class inherit problem definition?"
date: "2024-12-13"
id: "kotlin-data-class-inherit-problem-definition"
---

Alright so you're asking about data classes and inheritance in Kotlin right been there done that got the t-shirt a few times actually data classes are cool they’re like these super handy shortcuts for creating classes that primarily hold data you get equals hashCode toString copy all that jazz automatically its great less boilerplate saves me a ton of time so naturally you’d think ok inheritance lets throw in some hierarchy build a system make reusable classes easy peasy not quite

See the problem here isn't that kotlin prevents you from inheriting from a data class because it doesnt it does allow it what they don't let you do is inherit _into_ a data class It makes sense kinda when you think about it data classes rely on all those generated methods like equals and hashcode using all the primary constructor properties if you try to tack on properties through inheritance that's where things get wonky

For example lets say I had a few models in my first project ever in kotlin it was in college I did some android stuff at the time its nostalgic kinda I wanted a base class for things that had an id and then a data class for something concrete I was modelling users and posts on a sort of social media app a real simple one

I thought ok cool ill do something like this

```kotlin
open class BaseEntity(val id: Int)
data class User(val name: String, val email: String) : BaseEntity(id = 0) //compile error
```

Yeah that aint gonna fly the kotlin compiler yells at you cause it can't generate the equals hashcode etc cause BaseEntity doesnt have constructor props and we are passing a 0 in that is not something Kotlin will let you keep this way it says data classes cannot inherit from a class with parameters in primary constructor this has to do with the way the compiler creates the data class and the parameters it uses

So then I thought ok well lets maybe we can make the base entity into another abstract class so it can have its properties and then the user can implement those and the compiler will understand how to make the hash and stuff I tried this

```kotlin
abstract class BaseEntity(open val id: Int)
data class User(val name: String, val email: String, override val id:Int) : BaseEntity(id)  //Still compile error
```

Still no bueno kotlin’s compiler does not like that that still aint the proper use case you cannot inherit into data class and pass props that way and it complains like "hey you cant have these constructor params here they need to be in the primary constructor of the class so i can generate my things" its an error because you need to pass all the parameters via primary constructor and also it needs to be the same amount and type of parameters otherwise it wont generate the code you want also note how I have to use `override` keyword in the data class because id is an abstract property in the base class

The solution is you should do it the other way around you make your data class into a normal class and then inherit from that this was a hard lesson learned back then but after a few hours of trying and reading the kotlin docs I finally got it this works

```kotlin
open class BaseEntity(open val id: Int)
data class User(val name: String, val email: String, override val id: Int) : BaseEntity(id) //no error this is a good way
```

You see here we are inheriting from the base entity so that is fine and we are just passing the values from the constructor of the `User` to the `BaseEntity` and in this case because `BaseEntity` is an `open` class we can inherit from it and the compiler now is happy this pattern is actually pretty common

Ok so what is going on here fundamentally and why is it like this well its tied to how kotlin generates equals hashcode and toString and copy methods It relies heavily on the primary constructor parameters for these if you start injecting parameters through inheritance in a data class that causes issues

It breaks the contract so to speak of what a data class is supposed to be its supposed to be a simple data holder not something with complex inheritance hierarchies This is why they don't let you inherit into a data class only the other way around they are quite different beasts at the end

There are some workarounds I explored later when I needed to do something similar for real use cases but not at this stage in my college days

One way is to use composition over inheritance so basically have the data class hold an instance of the other class instead of inheriting it like instead of `User` inheriting from `BaseEntity` it would have a `BaseEntity` property of the id you have the same information but structured differently this is less "inheritance" more a has-a relationship

Another option and a pattern that I now prefer if you have complex class structures is to use sealed classes and interfaces that’s what I use now when I need to deal with this

Sealed classes allow you to represent a restricted hierarchy which is great for cases like when you are representing various states or conditions or different types of events You cant have them as `data` classes but you can have a number of nested data classes that are easy to work with also they offer good coverage and good when statements for all cases its quite handy to use it

Then interfaces allow you to define contracts that multiple data classes can implement if you need to say force the implementation of some methods this keeps things modular and reusable

So back to that original code that did not work that first error I had was a stupid one it was simple but it took me a bit of time to figure out what was happening. I remember staring at the screen for some time thinking this cant be that difficult can it it was but now it is obvious it was a hard lesson that I still remember vividly I even laugh sometimes remembering this when I get back to using Kotlin

I actually had a coworker once I had to explain this to I thought that he would know about this as he was supposed to be a senior dev and he asked why `BaseEntity` class had to be `open` we ended up chatting about this for some time until he finally understood the way the inheritance worked and not everyone can get it so dont feel bad about it

If you want to dive deeper into the nitty-gritty of how this works I would highly recommend checking the kotlin language documentation on data classes and inheritance the official docs are really good and well written also the "Effective Kotlin" book by Marcin Moskala is great it explains these concepts quite well and in detail with practical examples

Also if you are interested in learning more about OOP principles and the patterns we talked about check "Design Patterns Elements of Reusable Object-Oriented Software" this book is quite old but it explains fundamental principles for OOP in general regardless of the language and you will learn a lot about the why behind things like composition vs inheritance also it helps to understand what makes a good class structure when you are modeling problems

Remember also that this was in Kotlin I dont know if you are using other languages but if you do be aware that other languages might treat this case differently and the way classes work might vary

In the end the takeaway here is that data classes are designed for simple data structures without inheritance into it when you need inheritance you are better off making your data class into a normal class or go with composiiton or use sealed classes which provides more flexibility but remember to always check the documentation

Now i'm going back to what I was doing before writing a test for the user service i need to remember how it all worked hope this helped you out
