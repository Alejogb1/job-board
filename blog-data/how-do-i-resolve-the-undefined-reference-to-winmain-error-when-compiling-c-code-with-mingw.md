---
title: "How do I resolve the 'undefined reference to `WinMain`' error when compiling C++ code with MinGW?"
date: "2024-12-23"
id: "how-do-i-resolve-the-undefined-reference-to-winmain-error-when-compiling-c-code-with-mingw"
---

Ah, the dreaded “undefined reference to `WinMain`” error. I recall vividly troubleshooting this particular gremlin back in the mid-2000s when I was heavily involved in developing a cross-platform game engine. The issue always seemed to surface at the most inopportune times, usually right before a major demo. It's a classic symptom of a mismatch between your program’s entry point and what the Windows linker expects for executable applications. Let's break down why this happens and how to fix it.

Essentially, the "undefined reference to `WinMain`" error occurs when the linker, a crucial step in the compilation process, can’t find a function named `WinMain` that it believes should be present. This function, as it turns out, is the standard entry point for Windows GUI applications. When you’re using MinGW (Minimalist GNU for Windows), a port of the GCC compiler suite to Windows, this error commonly means that you've either written a console application but are trying to compile it as a Windows GUI program, or you've inadvertently omitted the correct entry point function.

The key concept here revolves around the distinction between console and GUI applications in Windows. Console applications, which you typically interact with through a command prompt, traditionally start their execution with the `main` function. On the other hand, GUI applications, which use windows and graphical elements, initiate with the `WinMain` function. The MinGW compiler and linker are quite particular about this. When your source code is compiled, the linker scans the object files created by the compiler for an entry point. If the linker is expecting to find `WinMain`—typically because you’ve configured your project to build as a GUI application—and it encounters only `main`, it throws the "undefined reference to `WinMain`" error. Conversely, if you are trying to build a console application and the linker looks for and does not find `main`, you might get a different but similar "undefined reference" error, perhaps to `main`.

Let's look at some common scenarios and their solutions with code examples.

**Scenario 1: Console Application Accidentally Linked as a GUI Application**

This is perhaps the most common pitfall. You have a straightforward console application using the standard `main` entry point, but somehow the linker is trying to make it a GUI program.

```cpp
// example1.cpp
#include <iostream>

int main() {
    std::cout << "Hello from a console application!" << std::endl;
    return 0;
}
```

If you compile this with a command line like this (which might be a default setting in some build environments):

```bash
g++ example1.cpp -o example1.exe -mwindows
```

The `-mwindows` flag tells the linker to create a GUI executable, forcing it to search for a `WinMain` function, which is not present in our source. You’ll get the error. To fix this, remove the `-mwindows` flag when compiling a console application.

```bash
g++ example1.cpp -o example1.exe
```

This will correctly link your console application and no longer expects `WinMain`, resolving the undefined reference error.

**Scenario 2: Intentionally Creating a GUI Application with the `WinMain` Function**

Let's say you genuinely want to create a Windows GUI application. Then, you need to implement the `WinMain` function correctly. This function follows a specific prototype:

```cpp
// example2.cpp
#include <windows.h>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    MessageBox(NULL, "Hello from a GUI application!", "Hello", MB_OK);
    return 0;
}
```

Compile this using the `-mwindows` flag, which tells the linker we're creating a GUI application:

```bash
g++ example2.cpp -o example2.exe -mwindows
```

This command will create a working windows gui executable and not give an error because it is able to find the function `WinMain` as it expected.

**Scenario 3: Potential Linking Issues with Subsystems or Libraries**

Sometimes, an issue could arise from an incompatibility with libraries you are linking. For instance, if your project uses certain third-party libraries that are expecting to be linked as GUI applications, even if the rest of your code is designed as a console application, you might still encounter this. While this situation is less frequent, it can confuse many newcomers to windows development. For example, consider a scenario where a library expects `WinMain`, but your main project only provides `main`:

```cpp
// example3.cpp
#include <iostream>

// Assume some library called 'library_that_needs_winmain.lib' that expects winmain

int main() {
    //... code that utilizes the above library that expects WinMain.
    std::cout << "This is just a console application example" << std::endl;
    return 0;
}
```

Attempting to link this directly against `library_that_needs_winmain.lib` without a proper windows gui entry point can result in the issue. The fix is either to adapt your application to use the `WinMain` entry point, use a different library that doesn't make that assumption, or compile the library to be compatible with a console application. Let's assume we must change the application.

```cpp
// example3a.cpp
#include <windows.h>
#include <iostream>

// Assume some library called 'library_that_needs_winmain.lib' that expects winmain

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
   //... code that utilizes the above library that expects WinMain.
    std::cout << "This is now using winmain" << std::endl;
    MessageBox(NULL, "Hello from a GUI application!", "Hello", MB_OK);
    return 0;
}
```

Compiling this with the correct `-mwindows` flag resolves the problem:
```bash
g++ example3a.cpp -o example3a.exe -mwindows
```

The key takeaway here is to be mindful of how your application and any external libraries are compiled and linked and to make sure to avoid an entry point mismatch.

**Recommendations for Further Reading**

For a more thorough understanding of Windows programming concepts, I strongly suggest diving into the following resources. First, **"Programming Windows" by Charles Petzold** is the go-to bible for anyone developing Windows applications, though it can be very thorough. Pay particular attention to chapters covering window management, message loops, and the overall structure of Windows programs, including the critical difference between console and GUI application startup procedures. Additionally, the documentation available directly from Microsoft on the Windows API is an invaluable resource. You can typically find this in their developer section. For a more compiler focused deep dive, **"Linkers and Loaders" by John R. Levine** offers detailed insights into how linkers work and how they resolve symbol references during the compilation process which can help you diagnose more complex linker errors.

In conclusion, resolving the "undefined reference to `WinMain`" error primarily boils down to ensuring consistency between the type of application you're building (console or GUI) and the linker's expectations, which is dictated by the presence or absence of the `-mwindows` flag when using g++. Understanding the role of `main` and `WinMain`, and how the `-mwindows` compiler flag changes the linking behavior, is paramount for any Windows developer using MinGW. With these insights, that annoying error can be banished for good.
