---
title: "Why JetBrains IDEs: How to add a semicolon automatically?"
date: "2024-12-15"
id: "why-jetbrains-ides-how-to-add-a-semicolon-automatically"
---

alright, so you're looking at automatically adding semicolons in jetbrains ides. this is a classic, and i've been down this rabbit hole myself more times than i care to count. let me break down what i've learned, and what i've found works well.

first off, it’s good to understand *why* this even matters. for me, it started back in my early days of javascript when i was switching between languages that did and didn’t require semicolons. i'd constantly forget them, and i got tired of the linter yelling at me and of the unexpected weird behaviours. i remember this one project using node.js and it was really a nightmare because of this issue. it was some kind of backend thing to control some custom hardware. the company decided to adopt a semicolons-mandatory style rule and i had to re-format the whole codebase because some developers did not use them, it was quite a task. that experience, needless to say, made me really appreciate automatic semicolon insertion. i wanted to avoid that pain in the future. the same happened when i had to work with typescript after a long java period. the transition was a bit hard at the beginning. so, yeah, semicolons matter, or at least, consistency in the use of semicolons matters.

now, let's dive into how jetbrains ides handle this. essentially, the auto-completion that happens when you are writing, and the reformatting capabilities are what we're going to be exploiting here.  there isn't a single, giant 'add semicolons' button. it's more about configuring the ide to insert them in the correct places as you code and, of course, when reformatting.

the key part lives in the settings. here's where you need to go:

*   open your jetbrains ide (e.g., intellij idea, pycharm, webstorm, etc)
*   go to ‘file’ -> ‘settings’ (or ‘intellij idea’ -> ‘preferences’ on macos).
*   navigate to ‘editor’ -> ‘code style’
*   select the language you are working with (javascript, typescript, kotlin, etc.)
*   check for something named like "use semicolons" or something related.

each language has its own specific way to configure code style, but usually the important options will be related with semicolon-handling. look for the 'punctuation' or 'other' tab if you don't see it immediately. enable the option to "enforce semicolons" or something similar. this is the first and most important step.

once that is set, the ide should start adding semicolons in auto-completion contexts and during code formatting. but this leads to the next question... what about *existing* code that doesn't have semicolons?

here's where the ‘reformat code’ function comes into play. use it after having enabled the setting. it’s usually a keyboard shortcut. `ctrl+alt+l` (or `cmd+alt+l` on macos) is the standard one but you can customize it. this shortcut will format the code based on the code style settings that you just configured. that includes adding missing semicolons based on the rule you’ve just configured.

let me give you an example. imagine i have javascript code like this:

```javascript
function calculateSum(a, b) {
  let sum = a + b
  return sum
}
console.log(calculateSum(5, 3))
```

after using `ctrl+alt+l` with the right semicolon configuration, it transforms into:

```javascript
function calculateSum(a, b) {
  let sum = a + b;
  return sum;
}

console.log(calculateSum(5, 3));
```

it’s that easy.

here’s another example. i work a lot with typescript. a common scenario would be to have an interface or a type without semicolons. let’s say that the initial code is this:

```typescript
interface User {
  id: number
  name: string
  email: string
}

const myUser:User = {
    id: 1,
    name:"John Doe",
    email: "john@example.com"
}
```

and after reformatting with `ctrl+alt+l` we get

```typescript
interface User {
    id: number;
    name: string;
    email: string;
}

const myUser: User = {
    id: 1,
    name: "John Doe",
    email: "john@example.com"
};
```

see how easy and effective it is?

and here's one last example with kotlin. this is how i would write a very simple class:

```kotlin
class Car (val brand: String, val model:String) {
    fun getFullName() = "${brand} ${model}"
}

fun main() {
    val car = Car("tesla", "model 3")
    println(car.getFullName())
}
```

after reformatting it becomes:

```kotlin
class Car(val brand: String, val model: String) {
    fun getFullName() = "${brand} ${model}"
}

fun main() {
    val car = Car("tesla", "model 3")
    println(car.getFullName())
}
```

notice how kotlin does not really need semicolons. but i did not use semicolons in the first place and after reformatting nothing happened. kotlin is designed in such way that if you include them it works but if you don’t include them it also works. i guess that was one of the things that they were thinking when they created kotlin, make it easier and more flexible than java. but i’m more of a semicolon-lover, even in languages where they are not mandatory. it's a personal preference that usually i enforce when i start working on a new project. it makes things more explicit and i find code more readable this way.

now, a note of caution: while the ide is smart, it's not perfect. there might be edge cases where it won't add a semicolon in the exact place you expect. especially with javascript it is tricky in some cases. i once spent half a day debugging a react app that was behaving weirdly, and it turned out to be a missing semicolon. the problem was not the semicolon itself but that the code ended up not doing what i expected.

also, beware that if you are collaborating in a team, and other developers in the team use different code style, then you may get conflicts because code style is not the same. it's not ideal, but there is an easy solution for that: agree on a code style, and use a linter like eslint (javascript, typescript) or detekt (kotlin) to enforce that same style across all members of the team and if possible configure your ide code-style to use that configuration. that will minimize future problems. usually you can share settings with the team by using a git repository that contains the configuration. it’s what i’ve been doing in my latest projects.

for further reading, i recommend checking out “clean code” by robert c. martin for general code style best practices and also "effective java" by joshua bloch if you are interested in java specific best practices. the official documentation for each language usually has a code style section too. also the documentation of the ide you are using has usually a section about code style. they are all great resources.

and now, a joke related to the subject: why did the javascript developer refuse to use semicolons? because he was always living on the edge... i am sorry, i could not resist. i tried to be serious and not to use funny words during the explanation but i am just a human, i’m allowed to make jokes sometimes, even bad ones like this one.

anyway, there you have it: how to get jetbrains ides to automatically handle semicolons. it’s all about configuration and the ‘reformat code’ function. good luck, and may your code always compile the first time.
