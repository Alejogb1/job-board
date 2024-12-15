---
title: "Why am I getting an Error when saving to keychain using SecItemAdd?"
date: "2024-12-15"
id: "why-am-i-getting-an-error-when-saving-to-keychain-using-secitemadd"
---

ah, keychain errors when using `secitemadd`, that's a classic. i've been down that rabbit hole myself more times than i care to remember. it's usually not a single obvious thing but a combination of little details that trip you up. let's break down what could be happening and how i've tackled these issues in the past.

first off, `secitemadd` returns errors for a whole bunch of reasons. the most common one, and the one i've personally spent far too much time on, is related to the `attributes` dictionary. you *have* to get that dictionary just so. it's picky. very picky. think of it like that one piece of code that you inherited from a previous developer and that nobody touches because it has like 4 layers of abstraction and it works, but nobody understands why it works, but we are not touching it.

let's say you are trying to save a password, and your `attributes` dictionary could look something like this (but of course with your specific values):

```swift
let attributes: [String: Any] = [
    kSecClass as String: kSecClassGenericPassword,
    kSecAttrAccount as String: "my_user_account",
    kSecAttrService as String: "my_app_service",
    kSecValueData as String: "your_super_secret_password".data(using: .utf8)!
]
```

notice the `kSecClass`, `kSecAttrAccount`, and `kSecAttrService` keys. if any one of these is missing or is set to the wrong value it could be trouble. in my own experiences, especially when i started out with ios development, i'd often mess up the values. i mean, you get used to all these `kSecSomething` constants, but at the beginning that’s just a pile of constant noise. also, that `kSecValueData` has to be a `data` object. if you pass a string it will go boom. it just throws you into some obscure error message. and i can tell you, that when you are under pressure this can be really annoying, because you might feel that you have the code correct but actually you are missing just one thing.

the error codes themselves aren't always crystal clear. you might see something like `-34018`, which means `errsecparam`. that's a pretty generic error, it basically says "something's wrong with your parameters." which, well, thanks. that helps a lot. but trust me, after a while, you start memorizing those numbers. i feel sometimes that i talk with the keychain in a non-human language. that feels weird some times.

one time, i remember i was building a user authentication system, and i was pulling my hair out because `secitemadd` kept failing. i was convinced that the problem was with my code, so i went through the code several times and of course, i found nothing. i tried logging all the values, checked the types, the constants, and everything seemed fine. later, i realized that it was a silly mistake on my part. turns out that another developer who had worked on the project before me had a different set of keychain access groups configuration. i was not even close to understanding why that could be the problem, but after the fact, everything started making more sense. i feel that it might be a very similar situation with you at this moment.

another common issue is dealing with access control attributes. let's say you want to make the item accessible only when the device is unlocked. you can add these attributes to your `attributes` dictionary:

```swift
let accessControl = SecAccessControlCreateWithFlags(kCFAllocatorDefault,
                                                    kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
                                                    .none,
                                                    nil)!

let attributes: [String: Any] = [
    kSecClass as String: kSecClassGenericPassword,
    kSecAttrAccount as String: "my_user_account",
    kSecAttrService as String: "my_app_service",
    kSecValueData as String: "your_super_secret_password".data(using: .utf8)!,
    kSecAttrAccessControl as String: accessControl
]
```

here we’re using `secaccesscontrolcreatewithflags` to specify `ksecattraccessiblewhenunlockedthisdeviceonly`, which means this keychain item will only be accessible when the device is unlocked. if you use other values, like `ksecattraccessibleafterfirstunlock` for example you can introduce bugs. also, always make sure you create the `secaccesscontrol` before adding it to the `attributes` dictionary. i learned that the hard way too. again, those parameters have to be just so.

if you are working with a development build, you also have to make sure that your team id is properly set up. i've seen several posts where people are having problems with `secitemadd`, and in most cases, it is because they did not set the team id correctly. this can cause problems in keychain access, specifically with access groups. check that everything is set up correctly in your apple developer account and in xcode. this is very very crucial, because apple keychain is not a joke.

debugging keychain issues can be a pain. printing the `attributes` dictionary might be helpful, but it's not always enough. sometimes, i've found myself using the `security` command-line tool to inspect the keychain and see if the item was actually added or not and what exactly the attributes are. this is especially helpful if you have many items in your keychain, because sometimes it is very difficult to find the items in the keychain. you might find the `security` command-line tool helpful.

here’s another piece of advice based on my own blunders. sometimes the issue is not with your `secitemadd` call, but with a previous operation. maybe you have a duplicate entry already in the keychain, or maybe you are trying to overwrite an item with different access control. in those cases you may have to use `secitemupdate` or `secitemdelete` if you have to remove the item and then try adding it again with `secitemadd`. sometimes you have to clean your existing keychain to get things to work again. if you don't do this, it can create very confusing situations. so always try to clean your keychain, remove the application, and then try again. it is a good practice.

```swift
let query: [String: Any] = [
    kSecClass as String: kSecClassGenericPassword,
    kSecAttrAccount as String: "my_user_account",
    kSecAttrService as String: "my_app_service"
]

let status = SecItemDelete(query as CFDictionary)
if status == errSecSuccess {
    print("item deleted")
}
else{
    print("error deleting item: \(status)")
}
```

this code deletes the item with the `ksecclass`, `ksecattraccount` and `ksecattrservice`. this is good to use if you want to create new items without any conflicts. this can help you in finding and fixing bugs with `secitemadd`.

in my own experience, i've created a thin wrapper around the keychain api that encapsulates all this complexity. it's not perfect, but it reduces the chance of making mistakes. it basically sets some standard rules and enforces them, so i do not have to worry about it anymore.

as a general suggestion, always check the apple documentation about keychain items and access control, it really helps. the apple documentation for `secitemadd` can also give you some other clues on what might be happening. also, you should explore the documentation for `secitemupdate` and `secitemdelete`, because you may need them if you have to fix things. but this is the best documentation that you should read: "security framework programming guide" by apple, it is a good read. there are also other books on ios security that can give you some insights on how this stuff works but this one is very good and it is more than enough for dealing with the issues that you might encounter on daily basics. i have been reading it for the past 5 years, and i still learn a lot from it. if that does not help, you should ask in places like stackoverflow, because sometimes it takes another person to actually see what might be wrong with your code. i know this very well, because i was helped more times than i care to remember in forums like that. sometimes your brain gets saturated.

finally, remember that keychain is super sensitive to parameters and device setup. if you are having a hard time dealing with it, do not feel bad. it is not easy to get your head around it. i've been working with ios for many years and it still gives me headaches from time to time.

that's about it. keychain errors can be tricky. but once you get the hang of it, it gets easier. it's kind of like trying to remember that old irc chat bot you used to play with back in 1999, it’s hard to remember at the beginning, but after a while it feels like second nature. i hope it helps you out!
