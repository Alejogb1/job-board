---
title: "How to Achieve Fast, Local AI with LlamaFile"
date: "2024-11-16"
id: "how-to-achieve-fast-local-ai-with-llamafile"
---

dude so this llama file video right  it's basically a hype-fest about making ai accessible to everyone not just the google-microsofts of the world  they're all about open source and local ai  think running massive language models on your grandma's laptop not some cloud server that's spying on your every keystroke  it's like a total underdog story for ai  and it's hilarious


first off the whole setup is amazing  they start with this super chill vibe  like  "hey we're here to talk about llama file its cool" and then BAM they drop this  "turning weights into programs" line which sounds like pure wizardry  i mean  weights  programs  what's going on  it's like magic but in a techy way  they actually show a slide with this super simple equation somewhere in the presentation  it's supposed to be the key to some magic but i have no idea what it meant.  it had something to do with making ai inference faster.  then this guy steven starts riffing about how gpu dependence is a total buzzkill  he throws in a  "no disrespect almighty jensen first of his name master of market cap" joke which cracks me up  it's like a perfect blend of tech talk and stand-up comedy



one of the key ideas is this whole "single-file executable" thing  it's nuts  they've basically packed an entire llm into one file  no installers no dependencies  it just runs  this is insane because they showed a single file running on windows mac and linux.  the magic is in cosmopolitan and its ability to cross-compile to all major platforms and architectures. you can imagine the nightmare of handling all those different compilation targets  but this thing just works  anywhere  like if they showed a slide with  a  single file running on a raspberry pi and the same file running on a supercomputer.  think about it  you download a file and it just runs on your windows machine your mac your linux box even your toaster maybe  it's crazy efficient.  here's a tiny taste of the code magic behind this single file execution  i'm not going to reproduce the full build system but here's a snippet illustrating the idea:

```c++
// a simplified example showing the core concept of a single executable across platforms

#include <iostream>
#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

int main() {
  std::cout << "Hello from LlamaFile!\n";
#ifdef _WIN32
  MessageBoxA(NULL, "Running on Windows!", "LlamaFile", MB_OK);
#elif __APPLE__
  system("osascript -e 'display dialog \"Running on macOS!\"'");
#else
  system("zenity --info --text=\"Running on Linux!\"");
#endif
  return 0;
}

```

pretty simple  right  but imagine doing that across every os and architecture they mentioned  that's the real sorcery


another big deal is the cpu inference speedup  they're not just talking about running things  they're talking about making it FAST  like 30 to 500% faster on the cpu  that's bonkers  they got this far mostly through clever loop unrolling  which is a classic cpu optimization trick but not easy.  justine explains how unrolling the outer loop in matrix multiplication  the core of llm processing  can dramatically improve performance  it's like rearranging furniture to get better airflow in your apartment  it’s not a fundamental change but it makes a huge difference.   here's a tiny python example illustrating the idea. obviously this is a simplification

```python
import numpy as np
import time

def matrix_multiply_naive(A, B):
  C = np.zeros((A.shape[0], B.shape[1]))
  for i in range(A.shape[0]):
    for j in range(B.shape[1]):
      for k in range(A.shape[1]):
        C[i, j] += A[i, k] * B[k, j]
  return C

def matrix_multiply_unrolled(A, B):
  C = np.zeros((A.shape[0], B.shape[1]))
  for i in range(A.shape[0]):
      for j in range(B.shape[1]):
          c = 0
          for k in range(0, A.shape[1], 4):
              c += A[i,k] * B[k,j] + A[i,k+1] * B[k+1,j] + A[i,k+2] * B[k+2,j] + A[i,k+3] * B[k+3,j]
          C[i,j] = c
  return C


A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)


start_time = time.time()
C_naive = matrix_multiply_naive(A,B)
end_time = time.time()
print(f"Naive multiplication time: {end_time - start_time:.4f} seconds")

start_time = time.time()
C_unrolled = matrix_multiply_unrolled(A,B)
end_time = time.time()
print(f"Unrolled multiplication time: {end_time - start_time:.4f} seconds")

print(np.allclose(C_naive, C_unrolled))
```

they also mention using  `sync_threads`  a technique usually associated with gpus  to get better cpu performance  it's like borrowing gpu tricks for cpu work  very clever   it shows the potential of taking ideas from one architecture and applying them to another.  they showed a before and after demo with a text summarization task and the difference is night and day  the old version is crawling while the new one's zipping along  it was a pretty impressive visual demonstration


another cool thing they did is build on top of llama.cpp a super popular open source llm project  this isn't starting from scratch  it's building on existing work and giving back to the community  it's a perfect example of collaborative open-source development  they even mention contributing their performance enhancements back to llama.cpp  which is awesome  that community support and collaboration is part of their story  its an excellent example of open source and its capacity for collaborative development


the resolution is pretty clear  they're not just showing off llama file they're showing off the power of open source and local ai  they're saying that anyone  not just the big tech giants  can make a real difference in the ai space  that's empowering and it's also a smart marketing move  they launch this "mozilla builders" program to fund and support open-source ai projects  it's a complete ecosystem  they're building the tools the community and the support system to make local ai a reality  the whole point was that this is possible using open-source technology for everyone. they showed a qr code with a link to some funding program.  its all very cleverly interwoven


and lastly this entire talk was delivered with incredible enthusiasm  they even used the phrase "juicy impactful problems"  i think that's the perfect way to describe  some of the amazing challenges in computer science. the whole video is packed with energy and excitement  it's infectious  and that alone is enough to make you want to jump into the world of open source ai  it made me want to learn more about loop unrolling


one more small code snippet.  this one uses tinyblas a library mentioned in the talk. remember this is just a sample:

```c++
#include <iostream>
#include "tinyblas.h"

int main() {
  // Initialize matrices
  float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Perform matrix multiplication using tinyblas
  tinyblas_sgemm(2,2,2,1.0f,A,2,B,2,0.0f,C,2);

  // Print results
  std::cout << "Resultant matrix C: " << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}

```


the emphasis on using this library is that its small, fast and easy to include in your projects unlike many other linear algebra libraries.

so yeah that's my rambling summary of the llama file video  it's a wild ride full of techy stuff  hilarious jokes and a whole lotta open-source love  and now i kinda want to build something myself  maybe ill start with that loop unrolling thing  wish me luck
