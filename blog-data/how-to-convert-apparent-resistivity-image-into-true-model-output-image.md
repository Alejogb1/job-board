---
title: "How to convert apparent resistivity image into "true" model output image?"
date: "2024-12-14"
id: "how-to-convert-apparent-resistivity-image-into-true-model-output-image"
---

alright, so you're looking to go from an apparent resistivity image, which is basically what you measure in the field, to a "true" resistivity model, which represents the actual electrical properties of the subsurface. it's a common and honestly, pretty fascinating problem in geophysics. i've spent more late nights than i care to remember staring at these things. let's break it down.

first, the apparent resistivity image is not a direct picture of what's down there. think of it like this: you're shining a light on a complex object, but you're only seeing the combined reflection of light from all the parts. the apparent resistivity is what the earth *appears* to be, based on the measurements taken at the surface. it's a distorted view caused by how electrical current flows through different materials, the geometry of the electrode array, and the various depths of investigation.

the 'true' resistivity, on the other hand, is the actual electrical property of each tiny volume of the subsurface. that's what we're chasing. the process of going from the apparent to the true is called inversion. and yeah, it's not trivial. there isn't a single button to press – although that would be amazing – instead, it's a whole process that usually involves iterative calculations.

so, how does it typically work? well, we need a forward model, that's basically a calculator that tells us what the apparent resistivity *should* look like given a particular 'true' model. we use this model to compare what we should see from an initial guessed true model to what we have measured in our apparent data. then, based on how close or how far off this guessed model is from the measured data, we adjust the true model and repeat the forward model again. rinse and repeat many times until the calculated apparent resistivity reasonably matches our measured apparent resistivity. we call the process iterative inversion. the inversion algorithm is the method that guides us to adjust the model.

i remember once working on a dataset from a particularly challenging area in the atacama desert. we were trying to map a deep aquifer, and the near-surface geology was so complex it was messing with our readings big time. we spent weeks fine-tuning the inversion parameters, basically trying to balance how much faith we put in the data vs. how smooth we wanted the final image to be. that time it was an absolute coding marathon, but we did find the aquifer, it was a good day.

here is where the techy part comes in. let's talk code. i can show you some examples. bear in mind that the specifics change depending on which inversion library or method you are working with. i am using python, just because it's what i'm most comfortable with.

first, you need to have your data. let's suppose that you have an `apparent_resistivity_array` that holds the 2d apparent resistivity image values, and also an array with the x and y positions of each measurement, `x_positions` and `y_positions` respectively.

```python
import numpy as np

# let's create fake data for example purposes
x_positions = np.linspace(0, 100, 20) # 20 points in x
y_positions = np.linspace(0, 50, 10) # 10 points in y
apparent_resistivity_array = np.random.uniform(10, 100, size=(len(y_positions), len(x_positions))) # fake values between 10 and 100
```

that sets us up with a dummy apparent resistivity dataset. obviously, in the real world, you would load it from your actual file, probably using something like `numpy.loadtxt` or `pandas.read_csv`.

now, to perform the actual inversion, you need an inversion library. there are several out there, but one commonly used is `pygimli`. it's open-source and pretty powerful.

```python
import pygimli as pg
from pygimli.meshtools import createMesh2D
from pygimli.physics import ert

# here, lets assume we have the locations and app. res. in numpy arrays
# we are going to re-arrange the arrays to what pygimli expects
meas_pos_xy = np.array(list(zip(x_positions.flatten(), y_positions.flatten())))
res_array = apparent_resistivity_array.flatten()

#create a datastructure in pygimli
data = ert.DataContainerERT(meas_pos_xy)
data.set('rhoa', res_array)

#we need to create a mesh
world = pg.createWorld(start=[-10, -10], end=[110, 60], layers=[10, 20, 30])
mesh = createMesh2D(world, quality=30)

#create a modeling operator with the geometry of our measurements
forward_op = ert.ERTModelling(verbose=True, data=data, mesh=mesh)

#we need an inversion scheme
inv_scheme = ert.ERTInversion(forward_op, verbose=True)
#here is where we put some regularization, smoothness, parameters...
model, resp = inv_scheme.run(lam=20, maxIter=20, verbose=True)

#lets take a look at our results
print(f"model: {model}")
print(f"response:{resp}")

```

what this snippet is doing is using `pygimli` to perform the inversion. it's setting up the measurement geometry, the mesh representing the subsurface, and defining the inversion process. the key parameters to play with are `lam` and `maxIter`. `lam` controls the regularization (smoothness of the model) and `maxIter` is the number of iterations the inversion will run.

this example uses a smoothness constraint on the inverted model, usually known as tikhonov regularization. this is useful when we know the subsurface changes slowly. but other algorithms, like total variation constraints, exist if we know the subsurface has sharp contrasts, which would be useful in our atacama desert case.

the resulting `model` will contain the estimated "true" resistivity values for each cell in the mesh. this is still an approximation, no inversion produces a perfect image. that's why it is necessary to compare the result with other sources of information about the subsurface.

finally, a quick visualization of the result can be done with matplotlib:

```python
import matplotlib.pyplot as plt

# plotting the inverted model
pg.show(mesh, data=model, label='resistivity', colorBar=True)
plt.show()
```

this code will plot the inverted resistivity image, showing how resistivity varies within your model. in an image, higher resistivity values would be displayed with a color, while lower resistivity values will be displayed in another.

remember, the key thing in inversion is never trust a single result. always do a sanity check against other geological info and test different inversion parameters. i mean, you do not want to find a deep aquifer and only discover later it was a mathematical artifact.

now, for resources: if you are starting out, i highly recommend reading "geophysical electromagnetism" by michael nabighian. it's an older book but it has very clear descriptions of the forward modeling and the physics behind electromagnetic methods. for inversion, "inverse theory and ill-posed problems" by heinz w. engl is a classic. it will provide the mathematical background needed to understand how these algorithms work. finally, for more practical insights into using pygimli, look at their documentation and try running their examples, they are incredibly comprehensive and well-maintained.

also, before you go deep into geophysics, remember that most geophysical problems are like going to the doctors. we start with measurements at the surface (like taking your temperature), and based on those, we do some math that has a lot of uncertainties to infer what's going on inside (like a medical scan). i’ve always found funny that geophysicist call the results models.

hope this helps, and remember inversion is a mix of good math, coding, and geological understanding and a lot of fine-tuning. it is a fascinating and complex field.
