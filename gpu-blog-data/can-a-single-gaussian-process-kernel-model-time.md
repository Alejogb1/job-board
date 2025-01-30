---
title: "Can a single Gaussian process kernel model time series data with different characteristics before and after a specific time point?"
date: "2025-01-30"
id: "can-a-single-gaussian-process-kernel-model-time"
---
A common challenge in time series modeling arises when the underlying process generating the data undergoes a change in its statistical properties. Gaussian process (GP) kernels, by nature, define a covariance structure over the entire input space. Therefore, a single, standard kernel is generally insufficient to accurately model time series data that exhibit distinct behaviors before and after a specific temporal shift. Directly applying a single, stationary kernel will tend to either undersmooth periods of high variation or oversmooth periods of low variation, depending on the kernel’s parameters. This leads to a model that poorly captures the diverse statistical traits within the data.

The difficulty arises because a GP kernel's parameters are global – they apply to all data points simultaneously. A standard radial basis function (RBF), for example, governed by length scale and variance parameters, assumes a uniform level of smoothness and variability across the entire input time range. Consequently, if the time series displays low variance and high smoothness prior to some point and then exhibits high variance with rapid fluctuations afterwards, a single kernel struggle to represent both segments. Trying to fit one kernel often results in a compromised, mediocre representation overall. This is not a failure of GPs themselves but rather a constraint of the kernel's global parameterization when applied to non-stationary data.

To handle time series exhibiting regime shifts, alternative strategies that extend standard GP modeling are necessary. These primarily revolve around the concept of piecewise modeling, where the time series is broken into segments with differing characteristics. One straightforward approach is to employ different GPs, each with its own kernel, for pre- and post-change intervals. However, this can create discontinuities at the boundaries. A more sophisticated alternative is to combine or adapt kernel functions specifically for handling non-stationarity. I have found these adaptive techniques to deliver superior results in my professional experience.

One such strategy is to use a combination of different kernels, each with parameters that can be localized to a specific time period. For instance, I have frequently used a kernel that is effectively a linear combination of two RBF kernels.

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Kernel
from scipy.spatial.distance import cdist

class PiecewiseRBF(Kernel):
  def __init__(self, length_scale_before=1.0, length_scale_after=1.0, variance_before=1.0, variance_after=1.0, switch_time=0.5):
      self.length_scale_before = length_scale_before
      self.length_scale_after = length_scale_after
      self.variance_before = variance_before
      self.variance_after = variance_after
      self.switch_time = switch_time
      self.requires_vector_input = True

  def __call__(self, X, Y=None, eval_gradient=False):
    if Y is None:
        Y = X
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y).reshape(-1, 1)
    distances = cdist(X, Y, 'sqeuclidean')

    k_before = self.variance_before * np.exp(-0.5 * distances / (self.length_scale_before**2))
    k_after = self.variance_after * np.exp(-0.5 * distances / (self.length_scale_after**2))

    k = np.zeros_like(distances)
    for i, row in enumerate(X):
        for j, col in enumerate(Y):
            if row[0] < self.switch_time and col[0] < self.switch_time:
                 k[i, j] = k_before[i,j]
            elif row[0] >= self.switch_time and col[0] >= self.switch_time:
                  k[i, j] = k_after[i, j]
            else: # Mixed pairs
              k[i,j] = np.sqrt(k_before[i,i] * k_after[j,j]) * np.exp(-distances[i,j] / (self.length_scale_before + self.length_scale_after)**2) # Geometric average of variances, harmonic of length scales

    if eval_gradient:
        raise NotImplementedError("Gradient evaluation not implemented for this kernel.")
    return k

  def diag(self, X):
      X = np.asarray(X).reshape(-1, 1)
      diag_values = np.where(X.flatten() < self.switch_time, self.variance_before, self.variance_after)
      return diag_values

  def is_stationary(self):
      return False

```

In this `PiecewiseRBF` implementation, I define separate length scale and variance parameters for before and after the switch time. For the mixed pairs I use an approximation to maintain smoothness. This allows for more flexibility in modeling different smoothness characteristics within distinct time periods. The `switch_time` parameter defines the exact time of the transition. Note this specific kernel requires a single dimensional time input. Also, I have added a `diag` method for efficient variance calculation. Finally, the `is_stationary` method returns `False` to denote it is not a standard stationary kernel. I deliberately skipped the gradient, for brevity.

Another approach I've used involves a kernel modification I refer to as 'kernel stretching'. Here, I use a time warping function that allows me to effectively compress or stretch the time axis for pre- and post-switch time periods.

```python
class TimeWarpedRBF(Kernel):
  def __init__(self, length_scale=1.0, variance=1.0, switch_time=0.5, stretch_before=1.0, stretch_after=1.0):
        self.length_scale = length_scale
        self.variance = variance
        self.switch_time = switch_time
        self.stretch_before = stretch_before
        self.stretch_after = stretch_after
        self.requires_vector_input = True

  def _warp_time(self, t):
    if t < self.switch_time:
        return t * self.stretch_before
    else:
        return self.switch_time * self.stretch_before + (t-self.switch_time) * self.stretch_after

  def __call__(self, X, Y=None, eval_gradient=False):
      if Y is None:
          Y = X
      X = np.asarray(X).reshape(-1, 1)
      Y = np.asarray(Y).reshape(-1, 1)
      warped_X = np.array([[self._warp_time(t[0])] for t in X])
      warped_Y = np.array([[self._warp_time(t[0])] for t in Y])

      distances = cdist(warped_X, warped_Y, 'sqeuclidean')
      k = self.variance * np.exp(-0.5 * distances / (self.length_scale**2))
      if eval_gradient:
          raise NotImplementedError("Gradient evaluation not implemented for this kernel.")
      return k

  def diag(self, X):
      return np.full(X.shape[0], self.variance)

  def is_stationary(self):
      return False
```

The `TimeWarpedRBF` class includes the `_warp_time` method to apply different stretching factors (`stretch_before` and `stretch_after`) before and after the `switch_time`. This creates an effective difference in the "apparent" time scale as seen by the standard RBF kernel, indirectly modifying smoothness characteristics without changing the actual kernel form. This can be particularly useful when there is not only a change in variance but also a change in frequency of the time series.

Finally, in more complex cases with abrupt shifts and potentially noisy data, I often combine a stationary kernel with a change point kernel that allows for a sudden jump in the mean. This involves building a custom kernel or utilizing a suitable existing kernel implementation.

```python
class ChangePointKernel(Kernel):
    def __init__(self, variance=1.0, switch_time=0.5):
        self.variance = variance
        self.switch_time = switch_time
        self.requires_vector_input = True

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        X = np.asarray(X).reshape(-1, 1)
        Y = np.asarray(Y).reshape(-1, 1)
        k = np.zeros((X.shape[0], Y.shape[0]))

        for i, t1 in enumerate(X):
          for j, t2 in enumerate(Y):
            if (t1[0] >= self.switch_time and t2[0] >= self.switch_time):
              k[i,j] = self.variance
            else:
              k[i,j] = 0
        if eval_gradient:
           raise NotImplementedError("Gradient evaluation not implemented for this kernel.")
        return k

    def diag(self, X):
      diag_values = np.where(X.flatten() >= self.switch_time, self.variance, 0)
      return diag_values

    def is_stationary(self):
        return False
```

In this `ChangePointKernel`, the covariance will only be non-zero when both of the time points are after the switch time. This is particularly useful in modelling a jump in the underlying mean of the process. Such a kernel can be combined with a standard RBF to model both a smooth background and a sudden shift in the process.

In summary, modeling non-stationary time series with regime changes requires more than a simple Gaussian process and a single kernel. Combining kernels, adapting length scales, or modifying the time axis itself by time warping are effective approaches.

For further study, I suggest exploring textbooks on Gaussian Processes, particularly those discussing non-stationary kernels. Publications from the NIPS conference also often contain cutting-edge research on these advanced kernel design problems. Bayesian modeling references can be helpful in understanding these techniques. In addition, familiarity with the scikit-learn and GPy libraries is extremely beneficial in practical experimentation. Always be mindful of the assumptions implicit in any particular kernel function and ensure the choice aligns with your understanding of the underlying data.
