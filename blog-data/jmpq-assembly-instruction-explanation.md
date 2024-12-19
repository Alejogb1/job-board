---
title: "jmpq assembly instruction explanation?"
date: "2024-12-13"
id: "jmpq-assembly-instruction-explanation"
---

Okay so you want to dive into `jmpq` right alright i get it it's one of those assembly instructions that seems simple on the surface but then you start digging and its like whoa there's a whole lot more going on. I've been there trust me. So let's break down `jmpq` specifically in x86-64 since that's what everyone is using these days unless you're doing embedded stuff which is a whole other rabbit hole that I've been down a few times with some obscure hardware back in my early days debugging proprietary firmware for a client that insisted on using some CPU no one ever heard of. Those were dark times but I digress.

Basically `jmpq` is a jump instruction. It's the boss of flow control in assembly it tells the CPU "hey stop doing what you're doing and go over there and start executing from that new memory location". But it's not *just* a jump it's a *long* jump. The 'q' at the end is important in the x86-64 world because it signifies that we're dealing with a 64-bit address. So the target address that `jmpq` is going to is a 64-bit value. If you try to use a short jump `jmp` without the q suffix or a `jmpw` which is 16-bit on a 64-bit system you're gonna have a bad time a very very bad time. The CPU will complain possibly with a segmentation fault or just act unpredictably which I had that happen back in my college days when I was learning this the hard way. Let me tell you tracking down that segmentation fault where a `jmp` was being used instead of a `jmpq` when I was just starting was a nightmare it took me almost a whole weekend to debug it.

The basic syntax looks something like this:

```assembly
jmpq  *memory_address_or_register
```

or

```assembly
jmpq  label
```

So you see there are two main ways to specify the destination that's why it gets a little complex. Either you give it a literal address which could be from a memory location *or* it's directly stored in a register or you can give it a symbol that's declared in your program which will then be resolved by the assembler/linker to an actual memory location. The asterisk `*` is there because in the first scenario you are dereferencing a memory location getting the content in that memory location as an address and then jumping to that address.

For example if you had the following in your assembly code:

```assembly
mov    $0x400000,%rax
mov    $0x401000,(%rax)
jmpq   *(%rax)
```

Here’s what is going on step by step

1 We load `0x400000` in the `rax` register
2 Then we load `0x401000` in memory location pointed by `rax` this means at memory location `0x400000` we are storing `0x401000`
3 Finally the `jmpq` is dereferencing the memory location stored in `rax` (which is `0x400000`) so we jump to memory location `0x401000` which could potentially be the location of a function.

See I had this weird edge case where I was doing some dynamic code generation on a very specific piece of hardware where I had to calculate function addresses on the fly and I messed up once and ended up jumping to some random memory location. The CPU wasn't having a good time and neither was I because debugging it without proper tools was like finding a needle in a haystack.

Now if you're using a label it's a little more straightforward because the assembler takes care of getting the right memory address and all that heavy lifting of dealing with registers. Let's say we have this:

```assembly
start:
  mov $60, %rax    # syscall number for exit
  xor %rdi, %rdi   # exit code 0
  syscall
my_function:
  mov $1, %rax    # syscall number for write
  mov $1, %rdi   # file descriptor 1 (stdout)
  mov $message, %rsi # message address
  mov $13, %rdx   # message length
  syscall
  jmpq start
message:
 .ascii "Hello world\n"
```

In this example the CPU will enter the program from the `start:` label the exit the program after that it will jump to the `my_function:` label it will execute the write syscall and then it will jump to the label `start:` again. If you run this you'll get an endless loop printing `Hello world` to the screen.

Now let's talk about how `jmpq` interacts with the stack. Unlike `callq` which saves the return address onto the stack `jmpq` doesn't. It just blindly jumps to the specified address. This is super important to remember because if you’re jumping into a function that expects a return address on the stack you're gonna have issues. You'll likely end up with a corrupted stack and your program will crash. Been there done that got the t-shirt. It's not a fun t-shirt to wear.

Another thing `jmpq` can be used for is tail call optimization. Instead of doing a `callq` followed by a `retq` which pushes the return address on the stack followed by a return the jump instruction can be used for a jump directly. So the stack doesn't have to be used.

Here's a slightly more complex example involving indirect jump using a memory location that we will have to setup:

```assembly
.data
  jump_table:
    .quad my_function  # 8 bytes for address of my_function
    .quad another_function #8 bytes address of another function
.text
.global _start
_start:
  mov $0, %rcx        # Initialize loop counter to zero
loop:
  cmp $2, %rcx        # Compare loop counter to 2
  je endloop        # If counter equal to 2 exit loop
  mov jump_table(,%rcx,8), %rax # Load function address from jump table
  jmpq *%rax         # Indirect jump based on function address
  inc %rcx       # increment counter
  jmp loop

my_function:
  mov $60, %rax    # syscall number for exit
  xor %rdi, %rdi   # exit code 0
  syscall
another_function:
  mov $1, %rax    # syscall number for write
  mov $1, %rdi   # file descriptor 1 (stdout)
  mov $message, %rsi # message address
  mov $18, %rdx   # message length
  syscall
  ret
endloop:
  mov $60, %rax    # syscall number for exit
  xor %rdi, %rdi   # exit code 0
  syscall
message:
 .ascii "Jumped correctly\n"
```

In this example a `jump_table` is being created at the `.data` section. In this jump table there are two addresses the address of `my_function` and the address of `another_function`. The main loop loads an address using the register `rcx` for index in the jump table it then performs an indirect jump to that location. This loops twice using `my_function` once and then using `another_function`. This is how for example virtual method tables are implemented in object oriented programming.

Now i know this can sound like a lot but the key here is to practice and experiment. Try to write small snippets of code with `jmpq` and see how things work. Start with basic jumps to labels then try messing around with indirect jumps. This is by far the best way to learn. You don't want to be that person who has a `jmp` when they should have used a `jmpq` and is scratching their head for days when you are debugging. Speaking from personal experience of course.

There are tons of great resources out there on this stuff. The Intel Software Developer Manuals are basically the bible for this information. Don't be scared by them they are very very detailed and the are the absolute source of truth for the instruction set. You can also find plenty of tutorials and blog posts online but the intel manual is the best. Also the book "Computer Organization and Design" by Patterson and Hennessy will be a great asset to fully understand this type of instructions.

I hope this clears things up a bit. If you have any specific questions just ask.

Oh and here’s a joke for you

Why do programmers prefer dark mode? Because light attracts bugs.
