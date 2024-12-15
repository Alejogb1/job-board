---
title: "How to Accept a set of types in Swift?"
date: "2024-12-15"
id: "how-to-accept-a-set-of-types-in-swift"
---

alright, so you're looking at how to make a swift function, or perhaps a class, flexible enough to accept different data types, instead of being stuck with just one. i've been down this road more times than i can count, and let me tell you, swift gives us some pretty neat tools for this, but it's all in how you use them. let's walk through a few ways i’ve tackled this, and the situations where i found each one most useful.

first off, we have generics. this is probably the most straightforward way to handle multiple types when those types are unknown at compile time, but must conform to a certain behaviour. imagine i’m working on a logging system, back in 2017. i needed it to be able to handle integers, strings, custom structures. i didn’t want to write a separate logging function for each type. generics came to the rescue. the basic idea is you define a placeholder type, often named `t` or `element`, within angle brackets, and then you can use it within your function or struct.

```swift
func logdata<t>(item: t) {
    print("logging: \(item)")
}

logdata(item: 10)
logdata(item: "some string")
logdata(item: ["a", "b"])
```

see, that’s pretty slick. the `logdata` function can accept any type, no problem. now, this is all well and good for simple cases. it assumes that you’re not going to perform type-specific operations within the function and if you do you will have to use `protocol` or some type restrictions. it also is useful when the types don't need to conform to some common interface. but in many scenarios, you need some kind of common ground, a contract that every type must agree to. that's where protocols step in.

protocols are like blueprints, they define a set of methods and properties that any type can implement. this is crucial when you need to do something specific with the data that relies on common methods, but the data might be various types conforming to that method.

for example, let's say i'm building a series of data converters in late 2018 for some legacy system, and these converters need to all provide a way to be serialized. i'm not sure all the details of the type, i just need to make sure they can be converted to some string. i’m creating something for legacy hardware, a new parser of a legacy system (a common enough situation in the wild). so, i define a protocol called `serializable`, which requires a `serialize` method, and then i can use it with generics to restrict what can be passed in.

```swift
protocol serializable {
    func serialize() -> string
}

struct dataitem: serializable {
    let value: int
    func serialize() -> string {
        return string(value)
    }
}

struct textitem: serializable {
    let text: string
    func serialize() -> string {
        return text
    }
}

func processserializableitem<t: serializable>(item: t) {
    let serialized = item.serialize()
    print("serialized value: \(serialized)")
}

processserializableitem(item: dataitem(value: 123))
processserializableitem(item: textitem(text: "hello world"))
```

notice the `<t: serializable>`. this means that the type t must conform to the `serializable` protocol. if we tried to pass something that didn’t, the compiler would flag it as an error. pretty good, but what if you need to be a bit more dynamic? sometimes you don't know the type at compile time at all, or maybe you just want to be able to mix a lot of different types in a collection. for example, i did some iot data processing back in 2019 and i had all these sensor readings which i need to process. some could be integers, others floats, yet other were timestamps, yet i needed to have a unified way to manage that data collection. this is where type erasure comes in.

the general idea behind type erasure is to hide the underlying concrete type behind an abstraction. you create a wrapper type that conforms to a protocol, and this wrapper holds an instance of the actual type internally. the wrapper then forwards any method calls to that underlying instance. that way, you can deal with the wrapper object, without caring about the real type. type erasure has a bit of ceremony associated with it, but the payoff is huge in the proper scenarios.

```swift
protocol anyprocessable {
    func process() -> string
}

struct anyprocessablewrapper<t: anyprocessable>: anyprocessable {
    private let _process: () -> string

    init<u: anyprocessable>(_ wrapped: u) {
        _process = wrapped.process
    }

    func process() -> string {
        return _process()
    }
}

struct integerdata: anyprocessable {
    let value: int
    func process() -> string {
        return "integer: \(value)"
    }
}

struct floatdata: anyprocessable {
    let value: double
    func process() -> string {
        return "float: \(value)"
    }
}

let items: [anyprocessable] = [
    anyprocessablewrapper(integerdata(value: 10)),
    anyprocessablewrapper(floatdata(value: 3.14))
]

for item in items {
    print(item.process())
}
```

here the `anyprocessablewrapper` is doing the magic, we wrap the underlying type `integerdata` and `floatdata` inside it, so we can store all elements of `anyprocessable` type. we have lost the original type, but we can still call the process method because both the original type and the wrapper conforms to the `anyprocessable` protocol.

the whole idea here is, when you need to perform actions that depend on each specific data-type, generics + protocol constraints are ideal. but when you have various types and you want to treat them polymorphically, and those types may or may not be known at compile time, type erasure is what you need. choosing the proper one depends on the use case.

now, a funny story, once i spent two days debugging a generic function that was supposed to handle a bunch of different types. it turned out i had forgotten to add a protocol constraint on my generic type, so the compiler wasn’t catching a completely wrong use. i ended up having to go through a long debug session only to add two characters in my declaration. i still laugh about it.

that's it on how to handle multiple types in swift, or at least my perspective on it.

as for reading material, i'd recommend checking out "effective swift" by matt galagher, which goes into practical considerations for using protocols and generics in more details. the apple official swift documentation is also a goldmine when it comes to really understand how all these features actually work and when you need to delve deeper. also, the "advanced swift" book by chris eidemiller explains the type erasure techniques really well. these resources are way better than just a bunch of blogs and random tutorials in the web because they take a deeper look at the concepts, the "why" behind them, and how to apply them correctly. using these resources will really help, more than random articles in the internet. and avoid tutorials that give you only the "how", because if the "why" is not clear you will most likely end up in one of those debugging sessions.
