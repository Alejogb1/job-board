---
title: "no exact matches buildexpression swift static method?"
date: "2024-12-13"
id: "no-exact-matches-buildexpression-swift-static-method"
---

Okay so you're hitting that "no exact matches buildexpression swift static method" wall huh I get it been there done that like literally yesterday it feels like Let's break this down I mean I spent the better part of last week wrestling with something similar so trust me I know the pain

Alright so you want a static method in Swift that somehow builds something via this `buildExpression` deal and the Swift compiler is just throwing fits at you saying it can't find the right match or something like that right It's usually a problem with type constraints or generics which is like the compilers kryptonite it always finds some hidden edge case you totally missed Let me tell you about my little adventure with this a while back

So I was working on this internal DSL thingy for parsing configurations think something along the lines of like a simplified YAML or something not exactly that but close I needed a way to take like a flat list of tokens and construct a complex hierarchical data structure Now this DSL had a bunch of different node types and the whole thing was super generic I had this builder class which had this method `buildExpression` that was supposed to figure out which type of node to build from those token inputs And it was declared static because well because it makes sense in a builder context right That was my first mistake

I went with this approach similar to this in the beginning because I thought it was clear and understandable

```swift
struct Token {
    let type: String
    let value: String
}

protocol Node {
    static func buildExpression(from tokens: [Token]) -> Self?
}


struct ConfigNode: Node {
    let key: String
    let value: String
    static func buildExpression(from tokens: [Token]) -> ConfigNode?{
        guard tokens.count == 2 && tokens[0].type == "key" && tokens[1].type == "value" else {return nil}
            return ConfigNode(key: tokens[0].value, value: tokens[1].value)
    }

}

struct ListConfigNode: Node {
    let items: [Node]
    static func buildExpression(from tokens: [Token]) -> ListConfigNode?{
        guard tokens.count > 0 && tokens[0].type == "list-start" && tokens.last?.type == "list-end" else {return nil}
        
        let innerTokens = Array(tokens[1..<tokens.count - 1])
        var items: [Node] = []
        var currentSubTokens: [Token] = []
        var level = 0
        for token in innerTokens{
            if token.type == "list-start"{
                level += 1
            }
            if token.type == "list-end"{
                level -= 1
            }
            if level == 0 && token.type == "separator"{
               if let item = ConfigNode.buildExpression(from: currentSubTokens) {
                    items.append(item)
                }else if let item = ListConfigNode.buildExpression(from: currentSubTokens) {
                    items.append(item)
               }
                currentSubTokens = []
            }else{
                currentSubTokens.append(token)
            }
        }
         if let item = ConfigNode.buildExpression(from: currentSubTokens) {
             items.append(item)
         }else if let item = ListConfigNode.buildExpression(from: currentSubTokens) {
            items.append(item)
         }
        return ListConfigNode(items: items)
    }
}
```

I had this naive hope that Swift could magically figure out which implementation of `buildExpression` to call at each stage based on the type it expected at runtime but of course that's not how static dispatch works. I mean it’s not like the compiler has a crystal ball its just logic you know I got a bunch of compiler errors complaining about not being able to resolve the static method I was pulling my hair out for a couple of hours. It turns out the Swift compiler cant know which type you will want in the current context if it’s static unless you explicitly tell it and make it a protocol requirement

The problem is that swift cannot see at the method definition that the return type of buildExpression will match each time the context of type you are using it so what we need to do is use the Self type associated with the concrete protocol type so we need to work with associated types

The first thing to realize here is that we need to make our protocol aware of the type it needs to build or rather it needs to refer to its own type. So instead of returning `Self?` we are going to use associated types because that allows us to abstract away the return type of `buildExpression`

```swift
protocol Node {
    associatedtype NodeType
    static func buildExpression(from tokens: [Token]) -> NodeType?
}
```

With the changes in the protocol definition we need to also adapt our code to match the associated type.

```swift
struct ConfigNode: Node {
    typealias NodeType = ConfigNode
    let key: String
    let value: String
    static func buildExpression(from tokens: [Token]) -> ConfigNode?{
        guard tokens.count == 2 && tokens[0].type == "key" && tokens[1].type == "value" else {return nil}
            return ConfigNode(key: tokens[0].value, value: tokens[1].value)
    }

}

struct ListConfigNode: Node {
    typealias NodeType = ListConfigNode
    let items: [Node]
    static func buildExpression(from tokens: [Token]) -> ListConfigNode?{
        guard tokens.count > 0 && tokens[0].type == "list-start" && tokens.last?.type == "list-end" else {return nil}
        
        let innerTokens = Array(tokens[1..<tokens.count - 1])
        var items: [Node] = []
        var currentSubTokens: [Token] = []
        var level = 0
        for token in innerTokens{
            if token.type == "list-start"{
                level += 1
            }
            if token.type == "list-end"{
                level -= 1
            }
            if level == 0 && token.type == "separator"{
               if let item = ConfigNode.buildExpression(from: currentSubTokens) {
                    items.append(item)
                }else if let item = ListConfigNode.buildExpression(from: currentSubTokens) {
                    items.append(item)
               }
                currentSubTokens = []
            }else{
                currentSubTokens.append(token)
            }
        }
         if let item = ConfigNode.buildExpression(from: currentSubTokens) {
             items.append(item)
         }else if let item = ListConfigNode.buildExpression(from: currentSubTokens) {
            items.append(item)
         }
        return ListConfigNode(items: items)
    }
}
```

This works I mean you can see the compiler is now happy because each time you call `buildExpression` you are making sure that the type it will be returning is one that the context can accept which is the `associatedtype NodeType` of the struct that is conforming to the Node protocol. I was still finding some issues because of the way I was managing errors and empty arrays so that’s another fix that I need to do on this code but as for the compiler issue of not finding the right buildExpression static methods this works really well. I ended up using this approach not just for this configuration builder but in a few other projects too it's really flexible once you get the hang of it. One thing that I did that helped a lot was to use a more generic type instead of hardcoding the return type to `Self` to make the function more reusable in the future I will show you what I mean.

So I made this protocol very generic.

```swift
protocol ExpressionBuilder{
    associatedtype BuildResult
    static func buildExpression(from tokens: [Token]) -> BuildResult?
}
```

And now all of my expressions build things of the generic type BuildResult which makes it easy to reuse if I change how my nodes are constructed

```swift
struct ConfigNode: ExpressionBuilder {
    typealias BuildResult = ConfigNode
    let key: String
    let value: String
    static func buildExpression(from tokens: [Token]) -> ConfigNode?{
        guard tokens.count == 2 && tokens[0].type == "key" && tokens[1].type == "value" else {return nil}
            return ConfigNode(key: tokens[0].value, value: tokens[1].value)
    }

}

struct ListConfigNode: ExpressionBuilder {
    typealias BuildResult = ListConfigNode
    let items: [Node]
    static func buildExpression(from tokens: [Token]) -> ListConfigNode?{
        guard tokens.count > 0 && tokens[0].type == "list-start" && tokens.last?.type == "list-end" else {return nil}
        
        let innerTokens = Array(tokens[1..<tokens.count - 1])
        var items: [Node] = []
        var currentSubTokens: [Token] = []
        var level = 0
        for token in innerTokens{
            if token.type == "list-start"{
                level += 1
            }
            if token.type == "list-end"{
                level -= 1
            }
            if level == 0 && token.type == "separator"{
               if let item = ConfigNode.buildExpression(from: currentSubTokens) {
                    items.append(item)
                }else if let item = ListConfigNode.buildExpression(from: currentSubTokens) {
                    items.append(item)
               }
                currentSubTokens = []
            }else{
                currentSubTokens.append(token)
            }
        }
         if let item = ConfigNode.buildExpression(from: currentSubTokens) {
             items.append(item)
         }else if let item = ListConfigNode.buildExpression(from: currentSubTokens) {
            items.append(item)
         }
        return ListConfigNode(items: items)
    }
}

```

I know this is a lot but it’s basically the gist of my experience with this issue. One thing I found to be really helpful was to actually draw out the types and see how the compiler was inferring them. It was like solving a puzzle only a type system is much less forgiving than a puzzle that you can just take apart and put back together again. You can’t really do that with compiler errors and type systems. Anyway you will find that reading the swift generics manifesto really helps to understand what’s going on and how the compiler works with generics [The Swift Generics Manifesto](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md) this was a game changer for me I really recommend that read

Also this book that goes deep into swift type system is a must read for you: [Advanced Swift](https://www.objc.io/books/advanced-swift/) by objc.io it's a paid one but honestly worth every penny. It really dives into the intricacies of how the language really works under the hood.

And if you’re still struggling feel free to ask more questions and please provide the code snippets or the error that you are having I’m happy to help. But please don't ask me about monads I’m still trying to figure that one out. It seems like something that I should know but it’s one of those things where its like trying to solve the halting problem you know what I mean.
