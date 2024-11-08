---
title: "Why My mexPrintf in Matlab Doesn't Show Up Right Away - Easy Fix!"
date: '2024-11-08'
id: 'why-my-mexprintf-in-matlab-doesn-t-show-up-right-away-easy-fix'
---

```c
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    // ... Your MEX function code ...

    // Print output and flush the buffer
    mexPrintf("Output from MEX function: ...\n");
    mexEvalString("drawnow;");
}
```
