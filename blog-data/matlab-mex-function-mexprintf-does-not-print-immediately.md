---
title: "matlab mex function mexprintf does not print immediately?"
date: "2024-12-13"
id: "matlab-mex-function-mexprintf-does-not-print-immediately"
---

 so you're hitting that classic mexPrintf buffering problem eh I've been there I think every MATLAB mex dev has honestly Let me tell you my story with this thing it's a rite of passage honestly

Back in the day I was working on this hardcore real-time audio processing thing with MATLAB we had to squeeze every last millisecond of performance out of the thing We had this custom algorithm running inside a mex function it was some heavy DSP stuff you know complex numbers FFTs filtering the whole shebang

Naturally we had lots of debugging output with mexPrintf at that time but man did it drive us crazy The prints would appear only *after* the mex function finished executing or sometimes they wouldn’t appear at all and it was like yelling into the void

So the thing is mexPrintf doesn't act like your standard printf function it's not directly writing to the console at that exact moment There’s a whole buffering thing happening under the hood MATLAB's I/O stream has some sort of internal buffer where mexPrintf output gets accumulated then when MATLAB decides to flush that buffer those prints finally show up in the command window or wherever you're expecting them. It's like being in a crowded stadium and trying to hear your friend shouting a message to you. Its there but you will hear it after some delays

This whole buffering thing is usually to make I/O operations more efficient MATLAB doesn't want to spend all its time writing characters directly to a console especially in real time applications that I am used to So it batches things together and sends them in larger chunks at a lower frequency which usually is a fine optimization but for the mexPrintf debugging outputs it gets really annoying

So what can you do? It's a surprisingly common issue and there are a couple of standard workarounds

First the most common trick and usually the first one someone suggests is using `fflush(stdout)` after your mexPrintf call This forces the standard output stream to flush its buffer immediately and all pending output will be written out so the prints will appear in real time I mean that is the name of the function and what we want it to do so the name is very descriptive This is what i did at that time it helped a lot. Here is a code example

```c
#include "mex.h"
#include <stdio.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mexPrintf("This is a mexPrintf message\n");
    fflush(stdout); // Force the buffer to flush immediately
    mexPrintf("This is another one\n");
    fflush(stdout);
    // some actual calculation goes here later
}
```

Using `fflush` after each `mexPrintf` is the way to go if you really need to debug the inner working of your algorithms. But it can become a bit verbose if you have a ton of prints all over your code. So while I was using it I was not happy with it.

Another approach which some people use is using `mexEvalString` to call `drawnow` this is more MATLAB specific which forces MATLAB to update its event queue and flush its output streams. This is a bit more of an indirect approach and I’m not the biggest fan since this forces MATLAB to do more than what is needed and it is also not direct but still I’ve seen it around a lot. This is an example of this method:

```c
#include "mex.h"
#include <stdio.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mexPrintf("This is a mexPrintf message\n");
    mexEvalString("drawnow;"); // Force the flush
    mexPrintf("This is another one\n");
    mexEvalString("drawnow;");

}
```

This method can help when you need updates on the figure handles and other graphics elements in addition to the `mexPrintf` debugging messages so that is a use case where it is a better choice than the `fflush` which is more focused on command line outputs and such

Finally if you have a super complicated situation that `fflush` and `drawnow` does not really solve I recommend making your own debug flag and output file mechanism instead of `mexPrintf` . The way I did this before was passing a "debug" parameter to the mex function and then when the flag is on I will output data to a text file that I can inspect later it's a bit more of setup in the beginning but it will give you total control of how and when the data is outputted

```c
#include "mex.h"
#include <stdio.h>
#include <stdbool.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Check for the debug input flag
    if (nrhs > 0 && mxIsLogical(prhs[0]) && mxGetScalar(prhs[0]))
    {
        bool debug_mode = true;
        FILE* fp = fopen("debug_output.txt", "a"); // Open file in append mode
        if (fp == NULL){
            mexPrintf("Error opening the debug file\n");
            return;
        }
        fprintf(fp, "Starting some calculations now\n");
        fprintf(fp, "Variable x is: %f \n", 10.0);

        // Close file when done
        fclose(fp);
    }
    else{
      // Do the normal calculations
      // here just some sample calculation as usual
    }
}
```
This method is great when you want to do heavy debugging with some high frequency real time computation and the standard I/O calls are not enough for you and they slow you down . I also used this for logging the output for long experiments that can last hours so I can inspect the data after it has finished.

Now a small joke about debugging: I once had a bug that was so persistent that I seriously considered starting a support group for other developers who had the same bug. But hey thats just software development for you.

A couple more things to keep in mind that are always a pain for new people to MEX development

1. **Be mindful of your data types**: When using mexPrintf pay extra attention to your format specifiers. For example using %f for a `double` and %d for an `int` or else you will just get garbage outputs and will confuse you more. It is like trying to use a screwdriver on a nail you will not get any good results and might destroy something
2.  **String handling in C:** C strings and MATLAB strings are not the same. So when you try to print some data be very careful that you are not trying to print a mxArray pointer with mexPrintf as it will give you some garbage address that you don’t know where it came from. You must first convert the mxArray to a C type string for it to work this is always a good test to know how new someone is to C and MEX development

And finally some advice for good resources to read up on these topics. If you want to dig into the nitty gritty details of standard I/O you can check out the classic "Advanced Programming in the UNIX Environment" by W. Richard Stevens it has more than you ever need to know on the topic of system level I/O operations. Also some good reading materials about MEX API are in MATLAB documentation but they are a bit dry so be sure to find other tutorials and forum posts about MEX function development.

 so to summarize you're hitting the buffered I/O issue of mexPrintf The standard solution is to use `fflush(stdout)` immediately after the `mexPrintf` calls if that is not enough use `mexEvalString` along with `drawnow` or as a last resource use your own debug logging system. Good luck and happy debugging its one of the hardest problems in programming but its also one of the most rewarding
