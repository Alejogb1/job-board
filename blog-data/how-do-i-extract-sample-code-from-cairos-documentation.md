---
title: "How do I extract sample code from Cairo's documentation?"
date: "2024-12-23"
id: "how-do-i-extract-sample-code-from-cairos-documentation"
---

Alright,  Extracting sample code from Cairo documentation can sometimes feel like parsing ancient scrolls, especially when you're aiming for something that fits directly into your codebase without much tinkering. I've certainly been there – a few years back, when Cairo was still relatively fresh, I spent a frustrating afternoon trying to get a seemingly simple proof function integrated. The documentation was solid, but pulling out the snippet and making it cooperate wasn’t always straightforward. It’s not just a copy-paste job, and here’s what I’ve found works best, along with some examples to illustrate.

First, it's crucial to understand *how* Cairo documentation typically presents code examples. They're generally embedded within narrative explanations, often spanning several lines and sometimes even across multiple sections. They're seldom given as neat, ready-to-use blocks. This means you'll need to pay close attention to context. What I’ve often found helpful is to mentally trace the logic of the surrounding text *before* focusing on the code itself. This clarifies the intended function and the assumptions behind the presented snippets. Consider the surrounding text as a kind of implicit unit test describing the functionality. I think of it as doing 'manual static analysis' on the explanation.

The second important point is to discern the *type* of code snippet you are extracting. Is it a definition, a function call, a complete proof, or merely an excerpt? For example, in the documentation describing Cairo's felt type, you may find isolated code fragments demonstrating the use of felt values in various contexts. These individual fragments are useful as isolated examples, but do not represent a self-contained program. This distinction greatly influences how you’ll use the extracted code. A snippet demonstrating a `using` statement requires all the associated structures in scope to compile, which is often not directly supplied by the documentation.

Let's look at some scenarios, with concrete code snippets, which should help drive the point. Suppose I want to understand how to use implicits in a Cairo function. I remember wrestling with this early on, and the documentation usually provides good examples, but they’re embedded.

**Example 1: Extracting function definition with implicits**

Assume, after a bit of searching, that I found this (simplified) example in a section describing implicits:

```
// Documentation text:
// To write a function that can allocate a memory cell using the range check builtin, you need to
// include the `range_check_ptr` as an implicit argument. This allows you to verify that results are in the
// expected range using the builtin. Here is a function that demonstrates this.

func allocate_and_check{range_check_ptr : RangeCheckPointer}() -> felt{
    let val = 5;
    assert val < 10; // using the range check builtin
    return val;
}

// and further down the documentation it shows how to call it...

func main(){
    let res = allocate_and_check();
}
```

Here, the documentation presents the `allocate_and_check` function and then uses it in `main`. My extraction goal would be to isolate `allocate_and_check` for use in a real file.

**Extracted Code:**

```cairo
%builtins range_check

func allocate_and_check{range_check_ptr : RangeCheckPointer}() -> felt{
    let val = 5;
    assert val < 10; // using the range check builtin
    return val;
}

func main() {
    let res = allocate_and_check();
    return ();
}
```

Notice that I added the `%builtins range_check` statement, as this is required to utilize the range check builtin, which `allocate_and_check` utilizes via implicits. This step is usually *not* explicitly shown in the documentation when they're explaining a concept because it's implicit in their example, but is usually explicitly stated earlier in the document. In practice, I usually cross reference documentation to pull these dependencies into my final implementation. This is, in my experience, a frequent cause of compilation errors following extraction. In this example, we extracted a complete function definition, and it was relatively straightforward, but needed the implicit context to be included for it to run correctly.

**Example 2: Extracting a trait definition**

Now consider another case - pulling out trait definitions. Assume I'm looking at the documentation discussing traits and find this example:

```
// Documentation Text:
// A trait allows defining behavior that multiple types can implement. Here we define a simple
// Display trait, that includes an associated function `display`.

trait Display {
    func display(self : Self) -> felt;
}


// ...later, a struct implmentation is given
struct MyStruct {
    field1 : felt,
    field2 : felt
}

impl MyStructDisplay of Display<MyStruct> {
    func display(self : MyStruct) -> felt{
        return self.field1 + self.field2;
    }
}
```

My extraction goal here is to pull the `Display` trait and the `impl` block into a working file. Note that in a real implementation, this would likely be in different files, but for the purposes of this demonstration I will keep them together.

**Extracted Code:**

```cairo
trait Display {
    func display(self : Self) -> felt;
}

struct MyStruct {
    field1 : felt,
    field2 : felt
}

impl MyStructDisplay of Display<MyStruct> {
    func display(self : MyStruct) -> felt{
        return self.field1 + self.field2;
    }
}

func main(){
    let s = MyStruct{field1: 1, field2 : 2};
    let result = s.display();
    return ();
}
```

Here, we extracted a trait and its implementation for a specific struct. Again, context is key; the surrounding documentation explained the purpose of the trait, and its implementation was presented in a way that was easy to isolate once I understood the trait's intent. The implementation includes type parameters ( `<MyStruct>` ) which can be easy to miss if you are quickly scanning the text for code examples. Furthermore, like the first example, I had to extend the code to include `main` so that it could be executed for demonstration purposes, but this would not typically be present in the extracted code.

**Example 3: Extracting a proof function**

Finally, let's take the somewhat more complex task of extracting a proof function. Suppose the documentation explains some details about modular arithmetic with an example of an assert statement within a function body:

```
// Documentation text:
// When working with large numbers, modulo operations are often used. To verify
// computations within a program, we can use an assert statement to verify
// correctness. Here is a function that does modular arithmetic to show how this works.

func modular_arithmetic_check(a: felt, b: felt, modulus: felt){
    let result = (a * b) % modulus;
    assert result < modulus; // verify this calculation

    return ();
}

// and later on, how the proof should be used:

func main() {
    modular_arithmetic_check(5, 3, 7);
    return ();
}

```

Now my goal is to extract the proof function and the main entry point.

**Extracted Code:**

```cairo
func modular_arithmetic_check(a: felt, b: felt, modulus: felt){
    let result = (a * b) % modulus;
    assert result < modulus; // verify this calculation
    return ();
}

func main() {
    modular_arithmetic_check(5, 3, 7);
    return ();
}
```

Here, the extraction was relatively straightforward, however the documentation would have often presented this in a different location compared to other examples. Proof functions are often separated from example usages in the text and this separation may require cross referencing the documentation to piece together the entire example. The main function can also be separated by several paragraphs from the original example.

In conclusion, extracting code snippets from Cairo documentation requires a careful, systematic approach. Don’t rush through it; understand the context of each fragment before trying to use it. Cross-reference examples, be attentive to hidden implicit dependencies, and be prepared to write surrounding code to make the extracted code function as a standalone example.

For further reading, I highly recommend the official Cairo documentation itself, particularly the sections dealing with implicits, traits, and proof functions. The Cairo by Example repository also provides valuable examples of working Cairo code in a more structured form. Also, the "Programming in Cairo" book by Eli Ben Sasson, and the white paper on StarkWare's proof system are excellent sources for deeper dives into Cairo's underlying concepts. These resources should give you a solid foundation for tackling those documentation snippets. Remember, practice makes perfect, and over time, this process will become much more intuitive.
