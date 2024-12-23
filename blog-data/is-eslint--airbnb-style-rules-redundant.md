---
title: "Is ESLint + Airbnb style rules redundant?"
date: "2024-12-23"
id: "is-eslint--airbnb-style-rules-redundant"
---

Okay, let's delve into this. I remember a particularly challenging project a few years back, a sprawling single-page application built with react. we were a mid-sized team, and while everyone was fairly experienced, coding styles varied wildly. the result was a codebase that, while functional, was incredibly difficult to navigate and maintain. this experience solidified my perspective on linters and style guides, specifically regarding eslint and the airbnb configuration, which i'll discuss.

the fundamental question here is whether incorporating both eslint and the airbnb style guide leads to redundancy. the short answer is: generally, no, they aren’t redundant, but the specifics demand a more nuanced exploration.

eslint, at its core, is a pluggable linting utility for javascript and its related syntaxes. it provides a framework for identifying and reporting patterns in code. these patterns might pertain to syntax, potential runtime errors, or deviations from established best practices. it’s the workhorse, if you will. it uses configurable rules to perform this analysis. you can, and many do, customize these rules extensively to match the specific needs of a project.

the airbnb javascript style guide, on the other hand, is a pre-defined collection of these rules, crafted based on considerable experience in the javascript ecosystem. it's opinionated, yes, and that’s kind of the point. it provides a comprehensive blueprint for writing consistent, maintainable, and readable javascript code. think of it as a specific set of pre-configured rules for eslint. it's not just about code formatting, though formatting plays a part. it also covers architectural principles, best practices, and common pitfalls within javascript development.

so, why not just use eslint's default rules? why bring airbnb into the mix? well, eslint's default rules are relatively minimal. they aim to catch blatant errors and syntax issues, but they don't provide the same level of in-depth guidance on how to write *good*, maintainable code that a style guide like airbnb does.

the key differentiator here is *opinion*. the airbnb configuration has a strong stance on many things: variable naming conventions, preferred use of specific syntax, function structure, object declarations, and so on. this provides consistency *across* your team and *throughout* your project. this saves developers from needing to make continuous style decisions, leading to a more efficient development process. this uniformity is incredibly valuable, especially in collaborative environments, and is where the true power lies.

now, let's look at some examples to illustrate the practical interplay between eslint and the airbnb configuration.

**example 1: consistent spacing & semicolons**

without any pre-set config, eslint might only flag syntax errors, but not inconsistent spacing or missing semicolons.

```javascript
//without a config, the below code would likely run fine
function myFunction(a, b){
const result = a+ b
return result}
console.log(myFunction(1,2));
```

however, with the airbnb configuration applied via eslint, this same code would generate several warnings or errors:

```javascript
// airbnb style enforced version
function myFunction(a, b) {
  const result = a + b;
  return result;
}

console.log(myFunction(1, 2));
```

notice the added space within the function declaration parenthesis, the spaces around the `+` operator, and the semicolon after `const result = a + b`. these aren't *errors* in terms of execution, but they're stylistic inconsistencies that make the code harder to read and maintain, which airbnb specifically addresses. eslint, with the airbnb preset, enforces those specific formatting conventions.

**example 2: consistent use of const and let**

eslint on its own may not enforce a distinction between `const` and `let` other than syntax error. it'll generally just allow either. the airbnb style guide, however, encourages using `const` by default, and `let` only when a variable needs to be reassigned.

```javascript
// default eslint rules may allow the below, it's syntactically valid.
let counter = 0;
counter = counter + 1;
console.log(counter);
```

but, with airbnb style applied, this code is flagged since `counter` is only reassigned once. and for example, in some cases it may promote using a more functional approach as an alternative to reassignment if possible.

```javascript
// airbnb configuration suggests
let counter = 0;
counter += 1;
console.log(counter); // if reassignment is the only option
// or better... if you don't need to reassign
const counter = 1;
console.log(counter);

```

the point is not that the initial code is inherently bad, but that the style guide promotes a more reasoned approach to declaring variables, and eslint, along with the airbnb preset, ensures the code adheres to that principle.

**example 3: enforcing array destructuring**

eslint alone will not enforce the usage of array destructuring, whereas, airbnb promotes it for clarity.

```javascript
function arrayFunction() {
  const myArray = [1,2,3];
  const first = myArray[0];
  const second = myArray[1];
  console.log(first, second);
}

arrayFunction();
```

when using eslint with airbnb config, this will generate a warning and recommend something akin to:

```javascript
function arrayFunction() {
 const myArray = [1,2,3];
 const [first, second] = myArray;
 console.log(first, second);
}

arrayFunction();
```

again, the initial code is valid javascript but it lacks readability compared to the destructuring implementation.

in conclusion, eslint provides the mechanism for code analysis, while the airbnb style guide provides the ruleset. you can use eslint without any pre-set style, but you lose the consistency and best-practice approach. you *can* configure eslint to enforce your own team-defined style guide, which is a viable alternative, but it requires more upfront work and meticulous maintenance. therefore, using the airbnb rules, or another popular style, is often a very efficient approach for development teams, and should not be considered redundant alongside eslint itself.

if you want a deeper understanding, i’d suggest reviewing *clean code* by robert c. martin for general principles. for specific eslint intricacies and plugins, examine the official eslint documentation thoroughly. for style guide philosophy, delving into the airbnb javascript style guide on their github repository would prove fruitful. remember that these tools are designed to aid, not hinder, and embracing their capabilities can lead to significantly more maintainable and scalable codebases.
