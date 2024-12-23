---
title: "num1 assembly parameter passing example?"
date: "2024-12-13"
id: "num1-assembly-parameter-passing-example"
---

 so num1 assembly parameter passing that’s a classic. I've tangled with this more times than I care to remember especially back when I was working on embedded systems and those custom processors that nobody else wanted to touch. It’s not rocket science but it’s fundamental and you gotta get it right or your whole system will just spectacularly implode. Seriously I've seen it.

First off when we're talking about assembly we're talking about directly manipulating registers and memory. No fancy abstractions here you’re on your own. Parameter passing in assembly is all about moving data into the right places before you call a subroutine or function and making sure it’s in the right place when the routine returns the results. The most common approach it’s just moving values into pre-defined registers or locations in memory but those places are all based on conventions or architectures. For x86_64 and ARM for example there's a clear set of rules these are called calling conventions and ignoring them is a recipe for disaster so pay attention to your platform's specification. I've seen some junior guys mess it up and it’s always a debugging nightmare.

Let’s break down a very simplified example. Imagine we want to add two numbers a and b and we’ll pretend we have a totally ficticious processor with some registers let's call them r0 r1 and r2 and a simple add instruction.

```assembly
; Hypothetical Processor Assembly
; Assuming that r0 holds first parameter a and r1 holds the second parameter b
; and r2 will hold the result

add_function:
    ADD r2 r0 r1  ; Add r0 and r1 store in r2
    RET          ; Return

; Main code where we call the function

main:
    LOAD r0 5  ; Load 5 into register r0
    LOAD r1 10 ; Load 10 into register r1
    CALL add_function ; Call subroutine
    ; At this point r2 should contain the result 15
    ;... rest of the code
```

This is basic but it illustrates the fundamental process. Before calling `add_function` we load the values into `r0` and `r1` that are by convention used for parameters. Inside `add_function` we perform the addition storing it into `r2` also assumed as our results register. When we return the result is available in `r2`. Of course real processors are way more complex than this but this is the core concept. I used a lot of these single register approach in early days of my career working with micro-controllers it's not pretty but it gets the job done if you are on a tiny device with just a few registers.

Now let’s get a bit closer to something real. Let's look at a x86_64 example using the AT&T syntax often found in linux. In x86_64 parameters are often passed using registers like `rdi`, `rsi`, `rdx`, `rcx`, `r8` and `r9` for the first 6 integer or pointer arguments respectively after which the stack is used. Return values are in `rax`. Here’s an example of a simple add function that takes two integers and returns their sum.

```assembly
# x86_64 Assembly AT&T Syntax
# rdi holds a and rsi holds b
# rax will hold the result

.globl add_func_x64
add_func_x64:
    movq %rdi %rax  # Move a from rdi to rax (copy a into rax)
    addq %rsi %rax # add b to the rax (rax now holds a+b)
    ret # Return the result in rax

# Main code example of usage

# Setup params
# mov $5, %rdi #move number 5 to rdi
# mov $10, %rsi #move number 10 to rsi
# call add_func_x64

# At the end rax will contain 15
```

Here the `movq` instruction is used to copy `rdi` to `rax` and then `addq` to add `rsi` to `rax`. `ret` simply returns the content of `rax` which is the result. The calling convention is very strict and all the registers must be used as described. That’s why you use the `mov` command because you can't modify the original values passed to your function. You can think of them as read-only when you enter the function even though you actually can modify their content this is why the conventions exists. I got burned by this once because I was testing a function with some hardcoded values. It worked perfectly I thought I was super smart then I passed new values the same function and the program crashed immediately because my hardcoded version modified the registers it was not supposed to touch.

Another alternative way of passing parameters or results is by using the stack. This method is commonly used when there are a large number of parameters or when the parameters are larger than a register (like structures). The stack is just an area of memory allocated for each function call to store local variables and parameters. The stack pointer (usually `rsp` in x86_64) keeps track of the top of the stack. The parameters are pushed onto the stack before the function call and then popped off the stack inside of the subroutine or function.

Let's do a last example using the stack for x86_64 this one is a bit more involved but essential to understand in deep systems development. This time lets pretend we want to calculate a sum of 3 integers stored on the stack that we passed to our function.

```assembly
# x86_64 Assembly using stack AT&T syntax
# Assuming first element is at the top of the stack
# parameters pushed in reversed order on the stack before calling
# the result is returned in rax
.globl add_func_stack

add_func_stack:
    pushq %rbp # Save the previous stack base pointer
    movq %rsp %rbp # Set current stack base pointer
    
    movq 16(%rbp), %rax # Fetch the third argument (stack grows down) at [rbp + 16]
    addq 8(%rbp), %rax  # Add the second argument at [rbp + 8]
    addq 24(%rbp), %rax  # Add the first argument at [rbp + 24]

    popq %rbp   # Restore the previous stack base pointer
    ret     # Return the result in rax
# Example of using the stack
# push $10
# push $20
# push $30
# call add_func_stack
# at the end rax will hold 60
```

Here you can see we first store the base pointer register `rbp` for later restore. Then we take the stack pointer to the base pointer for easier parameter access. The order of arguments on the stack are in reverse order since we are pushing each of them onto the stack. Because of that we use `16(%rbp)` `8(%rbp)` and `24(%rbp)` to access the correct arguments on the stack. We are adding them up and placing the result in `rax`. It’s crucial to understand this stack-based parameter passing since all programming languages rely on the stack for the same kind of operation.

I remember when I tried to optimize an old image processing library I got lost for 3 days trying to find why my parameters weren’t passed correctly. And you know what the issue was? I had reversed the order of parameters in my push operations. Oh the pain!

Resources if you want to dive deeper into this subject. I highly recommend "Computer Organization and Design" by David A Patterson and John L. Hennessy that covers the basic principles of computer architecture and the basics of low level programming you can also read "Modern Operating Systems" by Andrew S Tanenbaum to understand how the operating system use all this concepts. There are tons of processor manuals out there too each processor will have a different set of rules for parameter passing so always check the documentation for your specific hardware. Don't ever assume one architecture is the same than another otherwise you are going to have a very very bad time.

So yeah parameter passing in assembly is a fundamental skill that takes time to master and if you are not careful with it you will end up spending hours scratching your head on very tricky issues. Just remember to follow conventions check documentation and never be afraid to use a debugger to inspect those registers and the stack. I hope this helps you with your assembly adventures and maybe save you from some of the mistakes I've already made so you don’t have to!
