---
title: "How can reusable templates be created in Cairo?"
date: "2024-12-23"
id: "how-can-reusable-templates-be-created-in-cairo"
---

Let's tackle this one. Templates, or rather, reusable code patterns, are absolutely crucial when dealing with the kind of complexities Cairo throws at you, especially if you're aiming for maintainability and reducing redundant code, and I've certainly seen my fair share of that over the years. Cairo, while potent, requires some cleverness to structure things in a way that promotes reuse. I remember back in my early days on a project dealing with secure multi-party computation on StarkNet, we had a real mess of duplicated code until we finally nailed down a proper templating strategy. We had to refactor quite a bit, but the effort paid off enormously.

The core issue in Cairo, stemming from its low-level nature, is the lack of traditional template mechanisms akin to what you might find in, say, C++ or generics in languages like Java or C#. We don't have type parameters we can directly inject. This requires a more hands-on approach. When thinking about reusability in Cairo, I focus on a few specific techniques. The first is to design functions to accept and return abstract types, often using `felt252` as a sort of all-purpose data container, as well as `Span<T>`, which is also critical for generic data handling. The key here is to operate on data structures in a consistent way regardless of the exact type they might hold. This is especially useful if you find yourself needing to perform similar actions on different types.

Second, and closely related, is to utilize Cairo's built-in features like `Span<T>` and `Tuple`. These are fundamental for handling collections and composite data types abstractly. With `Span<T>`, you don't need to rewrite functions for arrays of `felt252` versus arrays of custom struct instances, for instance, so long as you design your functions with the Span at the input and output. It's about operating on views into data, not necessarily the data itself, which lends itself perfectly to reusable components.

Finally, I always advocate for a highly modular design philosophy. This means splitting your logic into small, digestible functions, each doing one thing very well. These smaller functions are easier to reason about, test, and ultimately, to reuse in different contexts. While not strictly "templating," it facilitates building higher-level logic by composing reusable lower-level pieces. It is, in essence, the principle that guides all good software design, regardless of the language.

Let's see how this looks in some actual code. The first example illustrates abstract type handling and function reuse via a simple `Span` based summing function.

```cairo
%lang starknet
from starkware.cairo.common.math import assert_nn
from starkware.cairo.common.alloc import alloc

func sum_span(span : Span{felt252}) -> (res : felt252):
    let res : felt252 = 0;
    let len = span.len;
    if len == 0:
        return (res=res,);
    end

    let (ptr) = span.ptr;
    loop:
        if len == 0:
            return (res=res,);
        end
        let current_value = ptr[0];
        assert_nn(current_value);
        let res = res + current_value;
        let (ptr) = ptr + 1;
        let len = len - 1;
        jmp loop;
end

@view
func test_sum_span_felt() -> (result : felt252):
    let (arr : felt252*) = alloc();
    assert arr[0] = 1;
    assert arr[1] = 2;
    assert arr[2] = 3;
    let span : Span{felt252} = Span(arr, 3);
    let (result) = sum_span(span=span);
    return (result=result,);
end

@view
func test_sum_span_struct() -> (result: felt252):
    struct MyStruct:
        member a : felt252
        member b : felt252
    end

    let (arr : MyStruct*) = alloc();
    assert arr[0] = MyStruct(a=1, b=1);
    assert arr[1] = MyStruct(a=2, b=2);
    assert arr[2] = MyStruct(a=3, b=3);

    let (arr_ptrs : felt252*) = alloc();
    assert arr_ptrs[0] = arr;
    assert arr_ptrs[1] = arr + 1 * 2 * size_of_MyStruct();
    assert arr_ptrs[2] = arr + 2 * 2 * size_of_MyStruct();
    let span_struct : Span{felt252} = Span(arr_ptrs, 3);
    let (result) = sum_span(span=span_struct);
    return (result=result,);

end
```
In this example, notice how the `sum_span` function is designed to operate generically on a `Span{felt252}`. It doesn't care whether that `Span` points to an array of plain `felt252` values, or an array of pointers to structs; the key thing is that we are passing a `felt252` span to the function, allowing reuse across various data types.

For a more complex example, imagine having to perform a similar operation to sum on various complex struct, but this time based on a specific member of the struct, so we also require a function pointer to a getter function.
```cairo
%lang starknet
from starkware.cairo.common.math import assert_nn
from starkware.cairo.common.alloc import alloc

struct MyStruct:
    member a : felt252
    member b : felt252
end

func get_struct_a(obj: MyStruct) -> (res : felt252):
    return (res=obj.a,);
end

func sum_struct_span(span : Span{felt252}, get_member : (MyStruct) -> (felt252)) -> (res : felt252):
    let res : felt252 = 0;
    let len = span.len;
    if len == 0:
        return (res=res,);
    end

    let (ptr) = span.ptr;
    loop:
        if len == 0:
            return (res=res,);
        end
        let current_value : MyStruct = ptr[0];
        let (val) = get_member(obj=current_value);
        assert_nn(val);
        let res = res + val;
        let (ptr) = ptr + 1;
        let len = len - 1;
        jmp loop;
end


@view
func test_sum_struct_span_with_getter() -> (result: felt252):

    let (arr : MyStruct*) = alloc();
    assert arr[0] = MyStruct(a=1, b=1);
    assert arr[1] = MyStruct(a=2, b=2);
    assert arr[2] = MyStruct(a=3, b=3);
    let (arr_ptrs : felt252*) = alloc();
    assert arr_ptrs[0] = arr;
    assert arr_ptrs[1] = arr + 1 * 2 * size_of_MyStruct();
    assert arr_ptrs[2] = arr + 2 * 2 * size_of_MyStruct();

    let span_struct : Span{felt252} = Span(arr_ptrs, 3);
    let (result) = sum_struct_span(span=span_struct, get_member=get_struct_a);
    return (result=result,);
end
```
This example takes things a step further by introducing a function pointer (`get_member`). This allows us to pass any function with the `(MyStruct) -> (felt252)` signature, achieving an even higher level of generalization. Here, the `sum_struct_span` can now add the result of any `felt252` returned from the passed getter function.

Finally, let's illustrate a more specific case, where you want to pass `Span<T>` with different types, and require a custom size to use that data properly:
```cairo
%lang starknet
from starkware.cairo.common.math import assert_nn
from starkware.cairo.common.alloc import alloc

struct MyStruct:
    member a : felt252
    member b : felt252
end

func process_data{T}(span : Span{felt252}, size : felt252, process_function : (felt252) -> ()):
    let len = span.len;
    if len == 0:
        return ();
    end

    let (ptr) = span.ptr;
    loop:
        if len == 0:
            return ();
        end
        let (current_value : felt252) = ptr[0];
        process_function(value=current_value);
        let (ptr) = ptr + size;
        let len = len - 1;
        jmp loop;
end


func print_felt(value: felt252) -> ():
    //placeholder function for printing
    return ();
end
func print_struct(value: felt252) -> ():
   //placeholder function for printing
   return ();
end

@view
func test_process_data_felt() -> ():
    let (arr : felt252*) = alloc();
    assert arr[0] = 1;
    assert arr[1] = 2;
    assert arr[2] = 3;
    let span : Span{felt252} = Span(arr, 3);
    process_data(span=span, size=1, process_function=print_felt);
    return ();
end

@view
func test_process_data_struct() -> ():
   let (arr : MyStruct*) = alloc();
    assert arr[0] = MyStruct(a=1, b=1);
    assert arr[1] = MyStruct(a=2, b=2);
    assert arr[2] = MyStruct(a=3, b=3);
    let (arr_ptrs : felt252*) = alloc();
    assert arr_ptrs[0] = arr;
    assert arr_ptrs[1] = arr + 1 * 2 * size_of_MyStruct();
    assert arr_ptrs[2] = arr + 2 * 2 * size_of_MyStruct();
    let span_struct : Span{felt252} = Span(arr_ptrs, 3);
    process_data(span=span_struct, size=2 * size_of_MyStruct(), process_function=print_struct);
    return ();
end
```
Here we have the function `process_data`, which takes a span, a size parameter and a process function. This allows us to iterate over the data with the correct step, while also allowing any type of function to be used on the data. This means we can perform data processing on different types of spans without rewriting the data loop.

In essence, "templating" in Cairo is about embracing abstraction, focusing on operating on data views and pointers, and utilizing function pointers to achieve flexible and reusable code patterns. It's not as straightforward as in other languages, but with careful design, it's definitely achievable.

For delving deeper, I highly recommend exploring the Cairo documentation thoroughly. Specific sections on `Span` and pointer arithmetic are invaluable. Moreover, studying the core Cairo library, especially the `starkware.cairo.common` modules, will show practical examples of these techniques. Consider reading up on "Software Engineering Principles and Practices" by Shari Lawrence Pfleeger for general reusable code techniques and designs. For Cairo-specific examples and best practices, keep an eye on the StarkNet documentation and community forums, as they often demonstrate emerging patterns.
