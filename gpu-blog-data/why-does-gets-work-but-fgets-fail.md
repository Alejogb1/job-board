---
title: "Why does `gets` work but `fgets` fail?"
date: "2025-01-30"
id: "why-does-gets-work-but-fgets-fail"
---
The critical distinction between `gets` and `fgets`, which often leads to the perceived failure of `fgets`, hinges on the way each function manages input buffer overflows and how the programmer interprets their behaviors. I've spent considerable time debugging legacy C systems where this precise issue was the root cause of intermittent crashes, particularly in applications processing user-provided strings. The problem isn't that `fgets` fails outright, but rather that it enforces a boundary – a safety mechanism that `gets` egregiously omits.

`gets`, in its simplest form, reads characters from standard input until it encounters a newline character ('\n') or end-of-file (EOF). It then appends a null terminator ('\0') to the end of the string. The significant flaw here is that `gets` does not accept a maximum buffer size. It reads and stores input without knowing the length of the destination buffer provided by the programmer. Should the input exceed the buffer's capacity, a buffer overflow occurs. This overwrite corrupts adjacent memory locations, leading to unpredictable behavior including, but not limited to, program crashes and security vulnerabilities like code injection. The program might appear to "work" in some scenarios, specifically with inputs shorter than the allocated buffer, while failing catastrophically when the input surpasses it. This unreliable, deceptive nature is why `gets` is considered unsafe and was officially deprecated in C99 and removed in C11.

`fgets`, on the other hand, is specifically designed to prevent such overflows. It takes three arguments: the destination character array (`char *str`), the maximum number of characters to read (`int num`), and the input stream (`FILE *stream`). The function will read at most `num - 1` characters from the input stream, appending the null terminator. This crucial difference is where many newcomers to C programming, accustomed to the simplistic nature of `gets`, misunderstand `fgets`. `fgets` will *not* read past the specified buffer size. Moreover, if a newline character is encountered before `num - 1` characters are read, it is included in the string that `fgets` stores in the buffer, *before* adding the null terminator. Unlike `gets` that always discards the newline, `fgets` preserves it. This means that after using `fgets`, the programmer is often left with a newline character at the end of the string that they might not expect. This can lead to the perception that the function "failed" if the code then expects, for example, string comparisons to work without handling the trailing newline.

Here are three code examples to demonstrate this behavior, along with explanations:

**Example 1: `gets` - Demonstrating an Overflow**

```c
#include <stdio.h>
#include <string.h>

int main() {
    char buffer[5]; // Intentionally small buffer
    printf("Enter a string: ");
    gets(buffer);
    printf("You entered: %s\n", buffer);
    return 0;
}
```
*Commentary:* In this example, a buffer of only 5 characters is allocated. If the user enters a string longer than 4 characters (plus the null terminator), `gets` will write past the boundary of the `buffer` array, causing undefined behavior. The program might appear to work initially if a string with fewer than five characters is input. However, anything longer will corrupt memory. In a debugging environment, you would likely see a segmentation fault or similar error if the overwritten memory is being actively used or is a protected memory location. This example underscores why using `gets` is unsafe and should be strictly avoided.

**Example 2: `fgets` - Correct Usage with Newline Handling**

```c
#include <stdio.h>
#include <string.h>

int main() {
    char buffer[10]; // Larger, safer buffer
    printf("Enter a string: ");
    fgets(buffer, sizeof(buffer), stdin);

    // Remove newline character if present
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len-1] == '\n') {
        buffer[len-1] = '\0';
    }

    printf("You entered: %s\n", buffer);
    return 0;
}
```

*Commentary:* Here, a buffer of 10 characters is declared. `fgets` is used to read from `stdin`, specifying the maximum read size as the size of the `buffer`. If a string longer than 9 characters is entered, `fgets` will only read the first 9 and terminate the string there. The code then checks if the last character read is a newline. If so, it is replaced with a null terminator, thus cleaning the input string. If a shorter string followed by enter is entered, it'll still capture the newline character. This is the expected and correct behavior, ensuring no buffer overflows and also illustrating the need to handle newline characters.

**Example 3: `fgets` - Illustrating Failure to Read Full Input**

```c
#include <stdio.h>
#include <string.h>

int main() {
    char buffer[5]; // Small buffer
    printf("Enter a string: ");
    fgets(buffer, sizeof(buffer), stdin);
    printf("You entered: %s\n", buffer);
    return 0;
}
```
*Commentary:* This example uses a 5 character buffer again, similar to the first example with `gets`, but it uses `fgets` instead. If the user enters more than 4 characters, `fgets` will only read the first 4 characters plus the null terminator. It *does not* trigger any undefined behavior, but the input string will be truncated. Additionally, if the user types less than 5 characters and hits the enter key, the newline will be included at the end of the string. While `fgets` does prevent a buffer overflow in this scenario, the user may not realize that only part of their input was read, this can also lead to incorrect program behavior if the user’s full input was required for an intended operation. Therefore, understanding how `fgets` truncates input and handles newlines is paramount to utilizing it effectively.

In essence, `fgets` enforces boundaries, forcing the programmer to manage input carefully. This contrasts with the behavior of `gets`, which implicitly trusts the size of the buffer that is being passed to it, thereby making it incredibly dangerous to use in any situation. This “failure” of `fgets` to produce an expected output is not really a failure, rather a responsible implementation to prevent buffer overflows, that should be understood by the developer.

For those seeking deeper knowledge, I strongly suggest reviewing a good C programming textbook or the online documentation for C standard library functions, particularly the sections on input/output and string handling. Several books cover this subject in great depth, and the online materials available from reputable sources provide a more complete understanding of standard library functions. Also consider experimenting with these functions under a debugger to directly observe the behavior of the memory and the state of the variables. Understanding how the system behaves at this level offers invaluable insights.
