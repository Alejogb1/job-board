---
title: "self used before all stored properties are initialized swift error?"
date: "2024-12-13"
id: "self-used-before-all-stored-properties-are-initialized-swift-error"
---

Alright so you're hitting that classic Swift roadblock right "self used before all stored properties are initialized" Been there done that got the t-shirt and probably a few gray hairs too lets dive in

I've seen this particular error pop up more times than I care to remember especially back in my early Swift days Like when I was first dabbling with UIKit and custom views boy oh boy did that bite me more than once I recall one project this weather app where I had this elaborate custom view for displaying the temperature It was a real mess of subviews and animations and the initialization sequence was more tangled than my headphones after a week in my backpack I spent a solid three hours debugging this exact error only to realize I had a single non optional variable declared right at the top without an initial value Talk about feeling dumb

But lesson learned it's all part of the journey right

Basically this error means exactly what it says Swift's initialization system is very strict especially when it comes to value types that dont have default values Its not like JavaScript where you can get away with undeclared variables things are a bit more disciplined here In Swift all your non optional stored properties must be initialized before you can use self even within the initializer this self rule prevents you from trying to access a variable that hasn't been setup yet and that would clearly cause a crash or unexpected behavior Think of it like building a house You cant start putting up the roof until the foundation and walls are in place Swift initialization works on that same principle

The compiler needs to be absolutely sure that by the time the initializer is done all the stored properties have a concrete value ready for use It's a good design because its meant to prevent your code from using garbage values which is often a very nasty bug to debug

Now you have a couple of solutions to get around this error

The most straightforward approach is to provide default values at the time of declaration If your property can have a reasonable default value use it

```swift
class MyClass {
  var myProperty: Int = 0 // default value is given here
  init() {
     // No initialization needed for myProperty
    print("myProperty initialized")
    print("Self can be accessed here no error will happen")
  }
}
```

In this example since `myProperty` has a starting value of zero you dont have to explicitly set it inside the initializer

Now what if your property doesnt have a logical default value? Well another way is you use an initializer where you can set the values in the initializer directly

```swift
class AnotherClass {
    var name: String
    var age: Int

    init(name: String, age: Int) {
        self.name = name
        self.age = age
        print("name and age properties were initialized")
    }
}

```

In this second example `name` and `age` must be set within the initializer otherwise the compiler will complain Also the moment you assign the values to name and age its all good self can be safely accessed. This shows that if there is an initializer of your class you can use this pattern

And you know what they say about the last resort If neither of the options are good for you its always a last choice that you can go with and that is using an optional or implicitly unwrapped optional type

```swift
class OptionalClass{
    var optionalProperty: String? //This property can be nil initially
    var forceUnwrappedOptionalProperty: String! //Force unwrap will cause a runtime error if its nil but in this case it is not

    init(){
        print("Start init")
        optionalProperty = "Initialized"
        forceUnwrappedOptionalProperty = "Initialized and we promise it will have a value"
        print("initialized")
    }
}

```

In the above code if you initialize the `optionalProperty` to `nil` there would not be any error because the value can be set to nil and that can happen before the initializer finishes in theory It does not need to be set in the class definition or in the initializer, However `forceUnwrappedOptionalProperty` has the same logic except when you use it you must not forget to handle the nil case so it wont crash your app this is not good practice generally speaking

If you have complex initialization logic where you need to call methods on your class or use computed properties its always wise to make sure all stored variables have initial values before you try to do these complex operations inside the initializer This is where the order in which you do things in your initializers actually matter

I'd also recommend if you're getting bogged down in initializers read up on two phase initialization this is often the culprit in the more complex cases It's not rocket science but a solid understanding of how Swift initializes its values will save you a lot of time in the long run If you're looking for resources check out the Swift documentation on initialization its very comprehensive A book that I often recommend to junior developers is "Effective Swift" by Matt Galloway This will take you from a beginner to intermediate level with a lot of practical examples

Oh also I had this one situation where my property was declared as a computed property with a getter but it had an underlying stored property inside the computed getter that I forgot to initialize It was like the compiler was going "dude seriously you're trying to access a property that doesn't exist" I facepalmed so hard that I think I saw my own skull that time Its amazing how much time you can waste trying to find these small details

To wrap up the error "self used before all stored properties are initialized" its a feature not a bug It ensures your Swift code doesnt run off the rails using uninitialized data You need to either give default values at the time of the property declaration initialize it in the initializer or make it optional ( use it with caution)

So next time you see that message don't panic just go methodically check your stored property definitions and initialization order and you'll nail it
