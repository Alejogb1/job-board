---
title: "phase unwrapping signal processing code?"
date: "2024-12-13"
id: "phase-unwrapping-signal-processing-code"
---

Okay so phase unwrapping right Been there done that got the t-shirt more like several stained with late-night caffeine and debugging tears let's dive in

So you're dealing with phase unwrapping probably some signal processing thing I'm guessing from the question title maybe radar maybe interferometry maybe even just some weird audio analysis who knows doesn't really matter the pain is universal

The core problem as you probably already know is that your measured phase is typically wrapped into the range of negative pi to pi or 0 to 2 pi it's a modulo operation the phase is a circular quantity and you get that nasty sawtooth effect So you have to figure out which multiple of 2 pi to add back to the measured phase to get the actual continuous phase This isn't as straightforward as adding a constant though its like trying to solve a jigsaw puzzle where you're only getting tiny glimpses of the real picture

I've had some epic battles with this honestly In one particularly memorable project back in my university days we were working with synthetic aperture radar SAR data this was before the current deep learning explosion by the way and boy did that unwrapping kick our butts We had this beautiful set of interferograms all wrapped nice and neat like pretty little wrapped presents except inside was chaos we could not extract the real information about ground deformation or whatever it was supposed to be and the project nearly went bust I think I survived on ramen for at least a week and had to grow a beard to deal with the stress the code looked like spaghetti after 3 days I am embarrassed even now thinking about it but thats how you learn right

The first thing to remember is that the phase unwrapping is an *ill-posed problem* what does that mean it means there are multiple solutions that could fit the observed data so you need more information or reasonable assumptions about the smoothness of the phase usually the assumption is that the real phase doesn't make gigantic jumps from one sample to the next that is the heart of many unwrapping algorithms

The simplest approach or the one people try first always is path following unwrapping this is the basic method you go point by point from an initial point and add or subtract multiples of 2 pi to get a continuous phase basically you compare the difference in phase between current and previous and if the change is greater than pi or negative pi add or subtract 2pi appropriately. The problem with this is that path following is very sensitive to noise and the unwrapping result depends on the path taken it can get into some horrible states if your data has too many jumps or inconsistencies but it is a good start to see if your problem is so simple that a very basic approach will work it is very fast to implement and also good for understanding the concept if you have never seen phase unwrapping before

Here is an example in Python using numpy I mean its Python you are probably using it right

```python
import numpy as np

def path_unwrap_1d(wrapped_phase):
  """
  Path-following unwrapping for 1D phase.

  Args:
      wrapped_phase: A 1D numpy array of wrapped phase values.

  Returns:
       A 1D numpy array of unwrapped phase values.
  """
  unwrapped_phase = np.copy(wrapped_phase)
  for i in range(1, len(wrapped_phase)):
    diff = wrapped_phase[i] - wrapped_phase[i - 1]
    if diff > np.pi:
        unwrapped_phase[i:] -= 2 * np.pi
    elif diff < -np.pi:
      unwrapped_phase[i:] += 2 * np.pi
  return unwrapped_phase
```

Now that is for 1D signals what about for 2D images This is where it gets a lot more challenging you can still try path following unwrapping but now the path matters more There are two paths that you can use for example Row-wise then column-wise or Column-wise then row-wise depending on the order you use the result might vary significantly. You might have inconsistencies between paths it is a mess especially with noisy or low quality data This is not an optimal method I only recommend it for testing and to make a naive implementation

Here is a 2D numpy implementation of a row and then column-wise unwrapping method

```python
import numpy as np

def path_unwrap_2d_row_column(wrapped_phase):
  """
    Path-following unwrapping for 2D phase (row-then-column).

    Args:
        wrapped_phase: A 2D numpy array of wrapped phase values.

    Returns:
         A 2D numpy array of unwrapped phase values.
  """
  unwrapped_phase = np.copy(wrapped_phase)

  # Unwrap along rows
  for row in range(unwrapped_phase.shape[0]):
    for col in range(1, unwrapped_phase.shape[1]):
      diff = unwrapped_phase[row, col] - unwrapped_phase[row, col - 1]
      if diff > np.pi:
        unwrapped_phase[row, col:] -= 2 * np.pi
      elif diff < -np.pi:
        unwrapped_phase[row, col:] += 2 * np.pi


  # Unwrap along columns
  for col in range(unwrapped_phase.shape[1]):
    for row in range(1, unwrapped_phase.shape[0]):
      diff = unwrapped_phase[row, col] - unwrapped_phase[row-1, col]
      if diff > np.pi:
        unwrapped_phase[row:, col] -= 2 * np.pi
      elif diff < -np.pi:
        unwrapped_phase[row:, col] += 2 * np.pi
  return unwrapped_phase
```

For more robust solutions we usually move towards more global optimization methods methods that try to find the best unwrapped phase across all data points at once instead of unwrapping point by point the most used one is the least squares method which minimizes the difference between the derivatives of the wrapped phase and the derivatives of the unwrapped phase It converts the problem into a linear system problem and a solution can be found efficiently for this linear problem

That's a good improvement but still doesn't solve all the problems. Now is where you have to make some choices according to your problem if your data has more noise you can try some quality guided unwrapping approach before applying the least squares algorithm. The idea is to use the information about the data to guide the unwrapping process, for instance areas of high phase gradient or high noise have lower quality and should be unwrapped last. That is generally a good direction to improve your quality results. There is many methods of this kind which is good since you have more options to try until finding the one that works for you

I recall once trying to unwrap a very noisy interferogram. After spending days trying different methods I found a hybrid method that combines quality guided unwrapping with iterative least squares and that was the solution. I felt I had conquered mount Everest by doing that. You would not believe how happy I was. I had to celebrate with pizza and beer. I think I was so happy that I thought I could see colors that I had never seen before It is crazy how such a seemingly small step can create so much pain

There are different methods of course but a robust and widely used one is the multigrid approach It means that you can downsample the image before unwrapping using a coarse grid and then progressively unwrap the phase in finer resolutions It is like unwrapping a blurry image and then refine it This allows for a faster and more efficient unwrapping process

For instance you could try this with the following

```python
import numpy as np
from scipy.fft import fft2, ifft2

def least_squares_unwrap_2d(wrapped_phase):
    """
    Least squares unwrapping for 2D phase

    Args:
        wrapped_phase: A 2D numpy array of wrapped phase values.

    Returns:
         A 2D numpy array of unwrapped phase values.
    """

    rows, cols = wrapped_phase.shape
    x = np.arange(cols)
    y = np.arange(rows)

    kx = 2 * np.pi * np.fft.fftfreq(cols)
    ky = 2 * np.pi * np.fft.fftfreq(rows)

    k_x, k_y = np.meshgrid(kx, ky)
    k_square = k_x**2 + k_y**2
    k_square[0,0] = 1 # avoid division by zero

    dx_wrapped = np.diff(wrapped_phase, axis=1, append=wrapped_phase[:,0,None])
    dy_wrapped = np.diff(wrapped_phase, axis=0, append=wrapped_phase[0,:,None])

    dx_wrapped = path_unwrap_1d(np.angle(np.exp(1j*dx_wrapped)),)
    dy_wrapped = path_unwrap_1d(np.angle(np.exp(1j*dy_wrapped)),)

    dx_fft = fft2(dx_wrapped)
    dy_fft = fft2(dy_wrapped)

    phase_fft = (k_x*dx_fft + k_y*dy_fft) / (k_square)
    unwrapped_phase = ifft2(phase_fft)

    return unwrapped_phase.real
```

This is obviously a complex topic so if you are looking for more in-depth knowledge I would highly recommend exploring some of the classics. For the math behind all this I would point you to "Digital Signal Processing" by Oppenheim and Schafer the classic for signal processing For a practical perspective and implementation ideas you should dive into "Two-Dimensional Phase Unwrapping: Theory, Algorithms and Network Applications" edited by Dennis C. Ghiglia and Mark D. Pritt that is the book on the subject

And remember when it gets too hard there is nothing wrong with taking a break its the only way to see it from a different perspective and maybe after a good rest you will see the solution. Good luck with your phase unwrapping endeavors it's a tough nut to crack but you can do it I believe in you. Just don't do it alone there is a ton of information out there use that.
