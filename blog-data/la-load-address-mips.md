---
title: "la load address mips?"
date: "2024-12-13"
id: "la-load-address-mips"
---

Okay so you're asking about load addresses in MIPS specifically huh Yeah I've been there done that got the T-shirt and probably a few compiler errors to go with it Let's break this down

First off `la` in MIPS assembly is the load address pseudo-instruction It's basically a convenient way for us to get the memory address of something into a register MIPS itself is a RISC architecture which means it tries to keep things simple and that simplicity can sometimes feel a little… verbose Let me just say MIPS was probably designed by engineers who love parentheses a little too much Anyway `la` isn't a single instruction it's a shortcut a macro if you will The assembler will convert it into a sequence of actual MIPS instructions

Now why do we need it Well when we work with data especially strings or arrays we need their addresses to actually manipulate them You can't just say "Hey MIPS give me the value in memory location X" You first need to tell MIPS where that memory location even is And that is where `la` shines like a freshly polished CPU

Back in my early days of coding I had a rather peculiar problem involving a large array in a MIPS program It was a simulation of a very very simple virtual ecosystem you know predator prey stuff and it worked ok until I tried scaling up the population dramatically Suddenly things started behaving weirdly The creatures moved in bizarre patterns acted like zombies or just simply disappeared after initialization I spent nearly two weeks scratching my head debugging with `printf` statements all over the place which in hindsight is just horrible practice I eventually figured out the problem was I wasn't initializing the creature structures properly I was using some address hardcoded by hand and it was so far from the heap that I got some garbage data and therefore zombie creatures I was basically manually trying to do what `la` does much better This was a very good lesson on why using symbolic names and relying on the assembler to do address calculation correctly is so important

So the basic form is `la $register label` where `$register` is where you want to store the address and `label` is the name of the data you want to access Now MIPS itself doesn't do addressing like x86 where you can have base registers and offsets all in one instruction MIPS uses a two step process usually involving `lui` and `ori` or `addi` to generate the address

Here's how it works under the hood Imagine the label is located at address 0x10010004 The `la` pseudo instruction will be expanded into two instructions like so

```assembly
lui $at 0x1001    ; load upper immediate part of 0x10010004 into temporary register $at
ori $v0 $at 0x0004  ; or $at with the lower immediate part 0x0004 and store it in $v0 which is our target register
```

So `lui` (load upper immediate) will place the 16 higher bits of 0x10010004 into register $at and then `ori` will put the lower 16 bits of the address in `v0` ( our destination register )

Let's see some examples to really solidify this

Example 1 Simple string loading

```assembly
.data
    message: .asciiz "Hello MIPS!"
.text
    .globl main
main:
    la $a0 message ; Load the address of the string "message" into $a0
    li $v0 4      ; System call code for printing a string
    syscall
    li $v0 10     ; System call code for exiting the program
    syscall
```

In this one we declare a string and then we use `la` to load its address into register `$a0` which is used as the argument for the print string system call

Example 2 Array access

```assembly
.data
    array: .word 10 20 30 40 50
.text
    .globl main
main:
    la $t0 array      ; Load the base address of the array into $t0
    lw $t1 4($t0)     ; Load the second element of the array into $t1
    li $v0 1         ; System call code for printing an integer
    move $a0 $t1     ; Move the loaded value to $a0 so it will be printed
    syscall
    li $v0 10        ; System call for exiting program
    syscall
```
Here we have array we use `la` to store the starting address of the array into register `$t0` Then we access elements by using offsets from the base address. I remember during another project I tried to optimize by avoiding the `lw` instruction by doing some pointer arithmetic manually and it worked but that was another rabbit hole to debug for a good hour when a simple `lw` was the correct and less headache inducing solution

Example 3 Function address loading

```assembly
.text
  .globl main
main:
  la $t0 myfunc ;Load the address of myfunc to register $t0
  jalr $t0 ; Jumps to address stored at $t0
  li $v0 10
  syscall

myfunc:
  li $v0 1
  li $a0 13
  syscall
  jr $ra
```
Here `la` allows us to store the address of a function into a register and then use the `jalr` instruction to jump to that location This can be useful when you want to have a table of function pointers a common thing to see in operating systems where they do system call dispatch I almost went mad when I was doing my first OS kernel project

A common pitfall especially if you are coming from high level programming is thinking that the label itself holds the data its just a memory address marker So remember we need `lw` (load word) or `lb` (load byte) to read the value from the memory location stored at the register that we just loaded via `la`

One more thing to keep in mind is that the `$at` register is a temporary register the assembler uses It's good to remember to not use it for anything important that your code relies on because it is almost guaranteed it will be overwritten by the assembler at some point and your program will behave in unexpected ways and thats not fun at all It took me 2 whole days to figure out why one particular small part of a MIPS assembler I did was not behaving correctly and of course it was because of that

So you want to become a MIPS master huh Here's my recommendation I highly recommend "Computer Organization and Design The Hardware/Software Interface" by Patterson and Hennessy It has plenty of in depth information on the MIPS architecture including all the gory details about addressing modes Also the "MIPS Assembly Language Programming" by Robert Britton is great to learn by practicing a lot as MIPS assembly is more of a hands on experience it's a more low level approach to programming you have to internalize the fundamentals by working through coding problems using the language I don't think reading is enough here you actually need to write tons of code to get comfortable with the architecture The web is full of resources but these two references are good starting points because of their depth

Now it would be good to learn how the whole process works from assembler to actual hardware execution of these instructions but that's another story altogether for another time I guess You can have a look at compiler design papers and computer architecture papers to get a deeper understanding if you want to. You will see that MIPS is really simple in comparison to x86_64 the later one being the result of many additions and backward compatibility requirements

So yeah that's `la` for you I hope this helps a little and remember to always test your code don't be like young me who thought that by writing the code in one go it would compile and run correctly I've learned the hard way debugging can be painful but also very insightful if you learn the right lessons from it

And one last thing remember to always comment your code or you'll be asking yourself why you did certain things 2 weeks later and trust me you won't remember unless you have comments. In fact I once wrote a really complex recursive MIPS assembly function and I did not comment it at all because "I knew it by heart" well I was wrong it took me 3 days to understand it again because I had forgotten why certain registers were used that way and I thought I was very very smart back then but I guess I was not.

Oh and before I forget: Why did the MIPS programmer quit his job? Because he didn't get enough *address space* ha ha you are not laughing I guess low level humour is not for everyone anyway that's it really hope this helps and let me know if there is any other specific MIPS question that you have I'm always around or lurking somewhere around
