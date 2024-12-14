---
title: "In a WGAN, how do I train a Discriminator more than the generator?"
date: "2024-12-14"
id: "in-a-wgan-how-do-i-train-a-discriminator-more-than-the-generator"
---

hey there,

i've been around the block a few times with gans, and wgan's in particular, and i've definitely hit that wall where the discriminator seems to just… lag behind, or sometimes, take over completely. i get what you mean by wanting to train it more than the generator. it's a common thing and a core concept behind wgan training stability.

so, let’s break this down, it's not about some hidden magic, it's more about controlling the training dynamics.

the whole point of a wgan, or wasserstein gan, is to use a different loss function— the earth mover’s distance— and a different training approach than the original gans. the trick there is that instead of classifying fake vs real data, the discriminator tries to learn the wasserstein distance between the two distributions. this requires keeping the discriminator’s function within a family of k-lipschitz functions, which is typically done by clipping its weights ( more on that in a bit).

the usual issue happens when the generator is learning faster than the discriminator which creates a situation where the discriminator can't distinguish fake from real samples and it starts giving poor gradients, the generator ends up not learning and the training collapses. now, the opposite can happen too, where the discriminator is too strong from the start and does not provide enough gradients to the generator. we want to find a balance. and that means controlling how often we train each network.

the first thing is to clarify what we actually mean by "train more". we are not talking about using better hardware, or using more data, but rather about how many steps or iterations of gradient descent we perform per network per epoch.

here's the usual approach that i use and the common one that i see across implementations:

instead of doing one step of discriminator training then one of generator training (like in classic gans), we train the discriminator several times before training the generator. it's like giving the discriminator more time to "catch up" and be a better critic before updating the generator. in code this translates to something like this:

```python
import torch
import torch.optim as optim

# example discriminator and generator (using a very basic structure for brevity)
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)

class Generator(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 784),
            torch.nn.Tanh()  # output scaled between -1 and 1
        )

    def forward(self, x):
        return self.model(x)


# hyperparameters
latent_dim = 100
lr = 0.0001
n_critic = 5  # discriminator training steps per generator step
clip_value = 0.01 # clipping discriminator weights for lipschitz constraint


# model initialization
discriminator = Discriminator()
generator = Generator(latent_dim)

# optimizers
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)
g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)

# loss function (wgan loss)
def wasserstein_loss(predictions_real, predictions_fake):
    return predictions_fake.mean() - predictions_real.mean()

# training loop example
def training_loop(n_epochs, batch_size, data_loader):
    for epoch in range(n_epochs):
        for i, data in enumerate(data_loader):
            real_data = data[0] # assuming data loader returns tensors
            batch_size = real_data.size(0)

            # --- discriminator training steps ---
            for _ in range(n_critic):
              discriminator.zero_grad()

              # real data
              predictions_real = discriminator(real_data.view(batch_size, -1))

              # fake data
              noise = torch.randn(batch_size, latent_dim)
              fake_data = generator(noise)
              predictions_fake = discriminator(fake_data.detach())

              #loss calculation
              d_loss = -wasserstein_loss(predictions_real, predictions_fake) #negative because we need to maximise the critic predictions not minimise the distance
              d_loss.backward()
              d_optimizer.step()

              # weight clipping
              for param in discriminator.parameters():
                  param.data.clamp_(-clip_value, clip_value)


            # --- generator training step ---
            generator.zero_grad()
            noise = torch.randn(batch_size, latent_dim)
            fake_data = generator(noise)
            predictions_fake = discriminator(fake_data)

            g_loss = -predictions_fake.mean() # want to fool the critic by minimising its value for fake samples

            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"epoch: {epoch} | step {i} | d_loss: {d_loss.item():.4f}, g_loss {g_loss.item():.4f}")

# dummy data loader to test
from torch.utils.data import TensorDataset, DataLoader
dummy_data = torch.rand(1000, 1, 28, 28)
dummy_dataset = TensorDataset(dummy_data)
dummy_loader = DataLoader(dummy_dataset, batch_size=64)


training_loop(1, 64, dummy_loader)
```
this `n_critic` parameter is what controls how much more training the discriminator gets. here it's set to `5`, meaning for every step of generator training, the discriminator has 5 steps of training. you can tweak this value depending on the model and data. i've seen some papers use up to 10 or 20, while on some very sensitive datasets, setting it to 1 or 2 may be more useful. it's about finding the sweet spot in your specific case.

now, about the weight clipping, it's crucial for enforcing the lipschitz constraint on the discriminator, without it, the wgan loss may not converge at all and the training would be unstable. the `clip_value` hyperparameter typically is quite small, values like 0.01 or 0.02 usually work well, but it depends on the scale of your model weights. this ensures that the discriminator is not trying to overfit too much to the training data and it remains a reasonable measure of distance between the distributions.

there are also other things that can be done such as adaptive critic training. this was proposed in a paper called "improved training of wasserstein gans" and it consists in measuring the norm of the gradients that the critic outputs and adapting the critic training steps based on its stability. i found this to be especially useful when my critic started to either overfit too quickly, or it didn't learn at all. but the implementation is somewhat more complex and usually requires a very well-tuned learning rate.

for further reading i'd recommend checking the original wgan paper "wasserstein gan" by arjovsky et al (it is a bit math heavy), and the "improved training of wasserstein gans" by gulrajani et al, it adds gradient penalties in case the clipping makes the discriminator too weak and is much easier to use. these both should give you a solid understanding of the theoretical background and practical considerations when training wgan's.

oh, one more thing i learned the hard way during my days trying to generate realistic looking cats; be sure to keep an eye on the wasserstein loss, if the loss explodes it means the model is failing or collapsing. i once spent three days debugging my code only to find out my training data was actually dogs. (it happens to the best of us)

here is an example of the gradient penalty approach:

```python
import torch
import torch.optim as optim
import torch.nn as nn

# same generator and discriminator definition as before...
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)


class Generator(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 784),
            torch.nn.Tanh()  # output scaled between -1 and 1
        )

    def forward(self, x):
        return self.model(x)

# hyperparameters
latent_dim = 100
lr = 0.0001
n_critic = 5
lambda_gp = 10 # gradient penalty weight

# model initialization
discriminator = Discriminator()
generator = Generator(latent_dim)

# optimizers
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)
g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)

def wasserstein_loss(predictions_real, predictions_fake):
    return predictions_fake.mean() - predictions_real.mean()

def gradient_penalty(real_data, fake_data, discriminator):
  batch_size = real_data.size(0)
  alpha = torch.rand(batch_size, 1)
  alpha = alpha.expand(real_data.size()).to(real_data.device)
  interpolated_data = alpha*real_data + (1-alpha)*fake_data

  interpolated_data = interpolated_data.requires_grad_(True)
  predictions_interpolated = discriminator(interpolated_data)

  grad_outputs = torch.ones(predictions_interpolated.size()).to(real_data.device)
  grad_interpolated = torch.autograd.grad(
      outputs = predictions_interpolated,
      inputs = interpolated_data,
      grad_outputs = grad_outputs,
      create_graph = True,
      retain_graph = True,
  )[0].view(batch_size, -1)

  grad_norm = torch.sqrt(torch.sum(grad_interpolated **2, dim=1)+ 1e-12)
  return lambda_gp*((grad_norm-1)**2).mean()


# training loop
def training_loop(n_epochs, batch_size, data_loader):
    for epoch in range(n_epochs):
        for i, data in enumerate(data_loader):
            real_data = data[0]
            batch_size = real_data.size(0)

            # --- discriminator training steps ---
            for _ in range(n_critic):
              discriminator.zero_grad()

              # real data
              predictions_real = discriminator(real_data.view(batch_size, -1))

              # fake data
              noise = torch.randn(batch_size, latent_dim)
              fake_data = generator(noise).detach()
              predictions_fake = discriminator(fake_data)


              #loss calculation
              gp = gradient_penalty(real_data.view(batch_size, -1), fake_data, discriminator)
              d_loss = -wasserstein_loss(predictions_real, predictions_fake) + gp

              d_loss.backward()
              d_optimizer.step()


            # --- generator training step ---
            generator.zero_grad()
            noise = torch.randn(batch_size, latent_dim)
            fake_data = generator(noise)
            predictions_fake = discriminator(fake_data)

            g_loss = -predictions_fake.mean()

            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"epoch: {epoch} | step {i} | d_loss: {d_loss.item():.4f}, g_loss {g_loss.item():.4f}")


# dummy data loader
from torch.utils.data import TensorDataset, DataLoader
dummy_data = torch.rand(1000, 1, 28, 28)
dummy_dataset = TensorDataset(dummy_data)
dummy_loader = DataLoader(dummy_dataset, batch_size=64)


training_loop(1, 64, dummy_loader)
```

notice here how i replaced the weight clipping part by the `gradient_penalty` function. the idea is to calculate the gradients with respect to the interpolated samples and impose a penalty if the norm is greater than one. it's a more computationally expensive technique than clipping but often it provides better results. the `lambda_gp` is a hyperparameter that usually works best between 5 and 15.

and that's pretty much the gist of it. controlling the training steps and the gradient penalties are crucial for the stability of wgan's. let me know if you have any more questions.
