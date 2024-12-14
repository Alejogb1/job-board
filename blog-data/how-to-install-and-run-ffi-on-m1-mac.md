---
title: "How to install and run ffi on M1 Mac?"
date: "2024-12-14"
id: "how-to-install-and-run-ffi-on-m1-mac"
---

alright, let's talk about getting ffi running on an m1 mac. i've been down this road before, and it's... well, let's just say it can be a bit finicky, especially if you're coming from an intel-based machine. the arm architecture introduces some quirks that the older x86 systems didn't have. i’ve been doing low level stuff for many years starting with the 68000 assembly language on my amiga, so believe me i've seen some weird hardware stuff.

first things first, the 'ffi' you're asking about most likely refers to foreign function interface. this is essentially a mechanism that allows a program written in one language to call code written in another language. think python calling c functions, or ruby interacting with compiled libraries. it’s a powerful tool, but it does require that the various pieces are compatible with each other and, in our case, with the hardware.

the biggest hurdle we face with an m1 mac is that many pre-compiled libraries, especially those involving low level interactions, are often built for intel processors (x86_64). so, if you try to just grab some random ffi related package, it might not work, or worse it will just crash at runtime. the magic ingredient here is rosetta 2, apple's translation layer that allows many x86_64 programs to run on arm. however, rosetta is not a silver bullet and not everything runs perfect. so, we’d better try to get things compiled for arm64. that way we will save performance because this is not emulation.

so, step one is to make sure your development tools are in order. the most crucial tool is the compiler, usually clang. apple provides a version of clang that supports arm64, so that is the one we want. verify your version by running `clang --version`. you should see something along the lines of "apple clang version". if it doesn't, you might need to install the command line tools by running `xcode-select --install`. this command will pop up a dialog box to install those utilities.

next, you'll need the specific ffi library. there are many options but libffi is probably the most common one used by different languages. for example in python it uses the `ctypes` package that uses libffi under the hood. sometimes it could be precompiled, but for the purpose of this document we want to compile it from source, to make sure we have an arm version. you need to download the source code from the libffi official website (search with google, is easy to find).

once you have the source, you'll need to unpack it and navigate to the root folder and prepare to compile it with the following commands in the terminal:

```bash
./configure --prefix=/usr/local
make
sudo make install
```

the `--prefix=/usr/local` part tells configure to install the library in `/usr/local`, a common location for manually installed software. if you use a different prefix, you will need to adjust accordingly, but for now we will assume we are installing in the standard location. then, `make` will compile the source code, and `sudo make install` will place the compiled files in their designated locations. the `sudo` is needed because it requires admin rights to write in the `/usr/local` folder.

now, let's delve into a python example because many times i’ve seen people using ffi with python. let’s pretend you have a c function you want to use in python and it’s called `add_numbers`. first, the c code:

```c
// add_numbers.c
#include <stdio.h>

int add_numbers(int a, int b) {
    return a + b;
}

```

compile this with clang using the following in the terminal:

```bash
clang -o add_numbers.so -shared -fPIC add_numbers.c
```

this generates `add_numbers.so`, a shared library. the `-fPIC` flag is critical as it makes it position-independent code, which is needed for shared libraries. then, the corresponding python code that uses the library:

```python
# ffi_example.py
import ctypes

# Load the shared library
lib = ctypes.CDLL("./add_numbers.so")

# Define the function's return type and argument types
lib.add_numbers.restype = ctypes.c_int
lib.add_numbers.argtypes = [ctypes.c_int, ctypes.c_int]

# Call the function
result = lib.add_numbers(5, 10)

# Print the result
print(f"result: {result}")

```

here, `ctypes.cdll` is used to load the shared library. then, `restype` and `argtypes` specify the return type and argument types of the c function, respectively. `ctypes.c_int` represents a c integer. finally we call the function and print the result. save it as `ffi_example.py` and execute with `python ffi_example.py`. you will see the result "result: 15" printed.

a similar thing could be done with a different language. for instance with go. you'll need the `cgo` tool, which is included with the go compiler. a tiny go example is the following:

first, you c code is the same as above which is `add_numbers.c`. then the go code:

```go
package main

// #cgo CFLAGS: -Wall
// #cgo LDFLAGS: -L. -ladd_numbers
// #include "./add_numbers.h"
import "C"
import "fmt"

func main() {
    result := C.add_numbers(C.int(7), C.int(3))
    fmt.Printf("Result from C: %d\n", result)
}
```

and a small header file named `add_numbers.h`:

```c
#ifndef ADD_NUMBERS_H
#define ADD_NUMBERS_H

int add_numbers(int a, int b);

#endif
```

compile the c library using:

```bash
clang -o libadd_numbers.so -shared -fPIC add_numbers.c
```

and then you can compile and run the go program using:

```bash
go run main.go
```

the comments at the beginning are important for the `cgo` tool, because those are configurations for the c compiler and linker that needs to be used to compile the program. `-ladd_numbers` specifies to link against the libadd_numbers shared library and `L.` tells the linker to search in the current directory for the library. note we don’t include the prefix `lib` because it’s not needed in this case.

now, you may be wondering where to go deeper with this? well, i always recommend reading the original documentation and papers, so i would suggest the following resources: “system interface design” by david r. butenhof and “operating systems: design and implementation” by andrew s. tanenbaum. those books delve into how operating systems and libraries interact, which is critical for a full understanding. you can also investigate the source code of libffi itself; that is often the best way to learn how something is implemented.

regarding debugging, things can sometimes go wrong. if you’re getting unexpected results, verify the types you’re passing between languages. a type mismatch between the c code and the python/go code can lead to a crash, or even worse, to silent data corruption. use a debugger to step into the code and see what values are actually being passed around. if all else fails, don't be afraid to sprinkle print statements in your c code to better understand what’s going on. remember the old debugging saying, “if it compiles, it works.” (it’s a joke, don’t try this at home).

the world of ffi can be tricky, but it's also very powerful. once you get the hang of it, you can build some pretty neat applications. just remember to double-check your configurations, and ensure all pieces of software are compatible with the arm architecture. take it one step at a time, and you'll get there. it took me some time to get this all figured out when i first switched to the m1 mac, but after some days i was able to have a pretty solid configuration. good luck.
