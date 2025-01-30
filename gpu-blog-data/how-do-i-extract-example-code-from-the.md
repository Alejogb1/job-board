---
title: "How do I extract example code from the Cairo documentation?"
date: "2025-01-30"
id: "how-do-i-extract-example-code-from-the"
---
The Cairo documentation, unlike some programming language resources, does not typically offer readily copyable code blocks directly within its textual explanations. Instead, code examples are often embedded within the narrative, structured as snippets within explanation paragraphs, or presented as isolated complete programs within example folders associated with the documentation repository. Extracting useful, functional code requires a careful approach combining both parsing the text and understanding the intended context.

My experience developing StarkNet applications using Cairo has revealed that effectively extracting these code samples demands a methodical approach, moving beyond merely copying highlighted text. Often, snippets provided within the documentation are not complete programs. They represent small functional units intended to demonstrate a single concept or a specific syntax feature. Thus, directly pasting these fragments into a Cairo source file will predictably result in compilation errors. Instead, one must analyze the snippet, understand its purpose, and integrate it appropriately within a complete Cairo program structure. This often entails creating necessary function declarations, defining types, and setting up the execution context. Further complexity arises because Cairo code often relies heavily on StarkNet-specific abstractions like storage variables and contract interfaces. Therefore, code extracted from the documentation, especially concerning storage and interaction with StarkNet itself, needs to be combined with appropriate contract declarations and necessary context, a process not always immediately intuitive.

The general approach I typically take is the following: first, I identify the specific section of the Cairo documentation relevant to the task at hand. This might be documentation concerning a particular built-in function, data structure, or contract feature. Once identified, I meticulously read through the text, carefully observing the code examples embedded within the explanation. It is essential to distinguish between code snippets designed for illustration and code which could realistically form part of a contract or program. A visual clue is the syntax highlight: Cairo code is frequently presented in a monospaced font and potentially with syntax coloring, although not always consistently across all documentation formats. Once a snippet has been located, I evaluate its dependencies. If the code refers to a type, function, or variable that isn't explicitly defined within the code itself, I then search for the relevant definition elsewhere in the documentation or accompanying example directories.

For example, imagine the documentation states: *"To declare a structure with three members, an example is:"* followed by:

```cairo
struct MyStruct {
    member1 : felt,
    member2 : u256,
    member3 : bool,
}
```

This code, by itself, does not define a complete program, and simply compiling it will lead to an error. Therefore, to use this within your program you would need to embed it within a suitable contract definition, function declaration, or at file scope (in the case of simple types like structs) – depending on the use case.

Here is how that code would be used inside a contract:

```cairo
%lang starknet

struct MyStruct {
    member1 : felt,
    member2 : u256,
    member3 : bool,
}

@storage_var
func stored_struct() -> MyStruct{
    return MyStruct(0, 0, false);
}

@external
func test_struct(a: felt, b: u256, c: bool) {
    let my_struct = MyStruct(a, b, c);
    stored_struct.write(my_struct);

    return ();
}

```

In this example, I've embedded the struct definition from the documentation within a StarkNet contract, defining a storage variable of type `MyStruct` and an external function to demonstrate how to use the type constructor to initialize an instance of this structure. The snippet is now placed in context, demonstrating the value of not just copying the code verbatim, but rather integrating it correctly.

Another example one might come across in the Cairo documentation relates to using arrays: *"Arrays in Cairo are represented by a pointer to the first element, and the length of the array. Example:"* followed by code something like this:

```cairo
func array_manipulation(arr: Array<felt>, len: felt) {
    assert len == arr.len();
    // do something with arr
    return ();
}
```
This snippet demonstrates manipulating an array. To create a function that uses this, I must create an initial array, pass it into that function, and provide the required length parameter. The following code demonstrates that:
```cairo
%lang starknet

func array_manipulation(arr: Array<felt>, len: felt) {
    assert len == arr.len();
     let first_element = arr[0]; // Accessing the first element.
    let len_felt = len; // Explicitly declare a felt variable to store len
    return ();
}

@external
func example_array() {
    let initial_arr = array![1, 2, 3];
    let initial_arr_len = array_len(initial_arr);
    array_manipulation(initial_arr, initial_arr_len);
    return ();
}
```

In this example, the core logic from the documentation snippet – using the `array.len()` function (now `array_len()`) and the `arr[0]` array indexing – is preserved, but now operates within a full Cairo program context, which initializes an array with the `array!` macro and calls the function. Notice that I had to add a call to `array_len` to provide the function with the required length, and that I needed to explicitly declare a variable `len_felt`, which wasn't present in the documentation code example. This highlights the requirement to integrate code snippets with other functionality.

A third example of a typical code snippet is related to performing bitwise operations, the documentation might say *"The bitwise operations are performed using the `&`, `|`, and `^` operators. Example:"* followed by the code below:
```cairo
func bitwise_example(a: felt, b: felt) {
    let c = a & b;
    let d = a | b;
    let e = a ^ b;
    return ();
}
```

Again, this function lacks any calling context and needs to be placed within a runnable program. Here's an example implementation:
```cairo
%lang starknet

func bitwise_example(a: felt, b: felt) {
    let c = a & b;
    let d = a | b;
    let e = a ^ b;
    return ();
}

@external
func perform_bitwise() {
    let a = 5;
    let b = 3;
    bitwise_example(a, b);
    return ();
}
```
In this case, the documentation provides all the required instructions for using the bitwise operators and no changes were required to the original source code, other than to embed it in a functional program that provided the necessary context (defining the variables `a` and `b`).

These examples illustrate the common process I follow when working with the Cairo documentation. I do not assume that example code can be simply copy-pasted; instead, I analyze, adapt, and integrate the snippets into the broader context of my program or contract. This includes accounting for variables and types, considering dependencies, and ensuring the overall program structure is valid.

Finally, I recommend the following when working with Cairo documentation. First, the official Cairo documentation website provides the most detailed information, and the reader should always cross-reference that when questions arise. Second, the Cairo language repository itself on platforms like GitHub often include example programs alongside the core code, these are invaluable in demonstrating the real-world usage of language features. Lastly, the StarkNet documentation, typically distinct from pure Cairo documentation, is required for working with contracts and storage related features, and the reader should utilize this whenever these sorts of features are being explored. By referencing these resources, you'll find that the ability to extract, understand, and effectively use example code from the documentation improves considerably. I believe this process results in a better understanding of both Cairo's specific syntax and its application in a StarkNet development context.
