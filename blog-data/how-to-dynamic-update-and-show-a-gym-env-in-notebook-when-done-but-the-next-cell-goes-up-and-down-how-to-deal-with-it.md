---
title: "How to Dynamic update and show a gym env in Notebook, when done but the next cell goes up and down. How to deal with it?"
date: "2024-12-15"
id: "how-to-dynamic-update-and-show-a-gym-env-in-notebook-when-done-but-the-next-cell-goes-up-and-down-how-to-deal-with-it"
---

hey,

i've been down this rabbit hole before, the dynamic gym env display in notebooks turning into a weird jumping jack. it's a pain, but not insurmountable. basically, what you're seeing is the notebook trying to render and re-render the animation in rapid succession, when the gym env is updated and a new frame should be show. the notebook's cell output is getting confused and it looks like it's scrolling up and down with each new frame, even after the env done. it's happened to me, let me elaborate.

i was working on a reinforcement learning project a couple of years back, building a custom robotic arm simulation. i was super stoked when i got the env to finally work, the arm moving and grabbing a virtual object. the problem? every time i ran a training episode, the notebook cell output would be this chaotic mess of flickering animation, going up and down rapidly. i thought i had created a new form of notebook torture. it was not my best moment. the next cell was jumping around more than my cat after catnip.

what's happening here is that the cell output is behaving like a live stream, while we want a static image updated in place. the notebook wants to render everything as soon as it's generated, and the gym env update loop is churning out frames quickly, causing the unwanted jumping. the crucial part is to control how the images are being displayed.

the main key to fix this is to make sure that the update of the frame is controlled and doesn't lead to the creation of a new output each time, but an update to the same existing output. using the `display` and `clear_output` from `ipython.display` is our friend here. `clear_output(wait=true)` will erase the previous output and `display` with the new image will create a new image in place. now we control the rendered image in a single cell output location. let me show you some code.

**basic approach using `display` and `clear_output`:**

```python
import gymnasium as gym
from IPython import display
import matplotlib.pyplot as plt
import time

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()

for _ in range(100):
    action = env.action_space.sample()
    _, _, done, _, _ = env.step(action)
    plt.imshow(env.render()) #rendering the gym env to a numpy array with plt.imshow to pass to display
    display.clear_output(wait=True) #cleaning the output
    display.display(plt.gcf()) #display the output using plt.gcf(), gcf means get current figure
    if done:
        break
env.close()
```

this simple code snippet solves the most common case, the display and clear process will show the image in the same place avoiding the up and down craziness. notice that the `clear_output(wait=true)` is critical here. it holds the clearing until new output is available, that's what keeps it in place, without the wait=true, the clearing process will be too fast and create a blank frame between each update. this makes the output flickering.

i spent a few hours before figuring that one out in my robotic arm project, i was thinking that the problem was the env itself, i even tried creating my own render function, which was a waste of time, it was just the `clear_output(wait=true)` missing. let's move to a more sophisticated example with animation.

**animation using `matplotlib.animation`:**

 sometimes, you want a smooth animation, not a sequence of static updates. for that, we can use matplotlib's animation capabilities. it allows you to create a proper animation object that's displayed within the notebook.

```python
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(env.render()) #init the image

def update(frame):
    action = env.action_space.sample()
    _, _, done, _, _ = env.step(action)
    im.set_data(env.render()) #update the image
    if done:
        return im, # return the image
    return im,

ani = animation.FuncAnimation(fig, update, frames=range(100), interval=50, blit=true) #blit needs all return from the update func to be iterable
plt.close()
from IPython.display import display
display(ani)
env.close()

```

the core idea here is creating an `animation.funcanimation` object and passing a function called update to it. this `update` function handles the simulation and updates the image. it's critical to close the plt to not duplicate the images. this prevents the notebook trying to add up multiple images, and the animation controls the display updating it in place.

the key is `im.set_data(env.render())` to update the image without creating a new object. this pattern works well for more complex animation scenarios. i've used this for visualizing training loops, where you want to see the agent's progress in the env over multiple episodes in a smooth way. i remember once, trying to make a cartpole agent learn without animation, and it was like trying to understand quantum physics with a blurry telescope. so i decided to fix it. it was much more clear after that.

**using a wrapper to control the animation (more robust and practical):**

now, let's talk about a more modular and manageable approach. let's wrap the env to handle the display logic:

```python
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display
import time

class NotebookRenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(self.env.render())


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._update_display()
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._update_display()
        return obs, info


    def _update_display(self):
        self.im.set_data(self.env.render())
        display.clear_output(wait=True)
        display.display(self.fig)

env = gym.make("CartPole-v1", render_mode="rgb_array")
wrapped_env = NotebookRenderWrapper(env)

for _ in range(100):
  action = wrapped_env.action_space.sample()
  _, _, done, _, _ = wrapped_env.step(action)
  if done:
    break
wrapped_env.close()
```

this creates a class called `notebookrenderwrapper`, this gym wrapper encapsulates all the display logic, making your main training loop cleaner and more manageable. now every time the `step` or `reset` method is called it will automatically update the display in the cell. this pattern is helpful, if you want more control over the display.

when i was debugging a custom env i was working on, i wrapped my env to a debug version, it would show in the display every useful info, and this wrapper approach made it very clear the problem with my implementation and that saved my time.

one last piece of advice: avoid rendering the env if you're doing heavy computations. it's a resource hog and will slow down your training. only use it when you need to visualize the agent's behaviour or the env itself. it's like trying to run a marathon with high heels on: not advisable.

i'd recommend reading up on the matplotlib documentation for animations, also the ipython display module documentation. they're your go-to for these kind of issues. i would also recommend the book "reinforcement learning: an introduction" by sutton and barto, not specifically for this problem but a must read for reinforcement learning concepts.

hope this helps, let me know if you have any more questions.
