---
title: "How to better train a DCGAN?"
date: "2024-12-14"
id: "how-to-better-train-a-dcgan"
---

alright, so you're hitting the wall with dcgans, huh? i've been there, trust me. it’s like trying to teach a toddler to paint like van gogh – frustrating and sometimes just plain messy. i've spent a good chunk of my career wrestling with these generative models, and they can be real divas. let's talk about how to coax them into producing something decent.

first off, the vanilla dcgan, as cool as it is conceptually, often falls flat on its face in practice. it's not uncommon to end up with mode collapse, where the generator spits out only a handful of very similar, low-quality images, or gradients that just vanish into thin air. i had this project years ago where i was tasked with generating realistic textures, and my first dcgan attempt looked more like something a toddler drew with mud than actual seamless textures. it was a humbling experience.

so, what can we do? well, it's less about a single magic trick and more about a combination of careful design choices and some clever tweaks. it's about understanding the underlying dynamics, and really optimizing everything.

one key thing is the architecture itself. the original dcgan paper suggests some good rules of thumb: use convolutional layers without max-pooling, replace fully connected layers with convolutional ones, use batch normalization in both the generator and discriminator, and use relu activation in the generator except for the output layer where you use tanh, and leaky relu in the discriminator. all of those recommendations are a good starting point, but consider also adding deeper networks. i've had better results going deeper than the original dcgan implementation. try increasing the number of layers gradually and monitoring the loss. a deeper network allows the model to learn more complex features, but be careful with vanishing gradients. batch norm helps, but sometimes you need more careful initialization.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, channels, features):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(latent_dim, features * 16, 4, 1, 0), # N x 4 x 4
            self._block(features * 16, features * 8, 4, 2, 1), # N x 8 x 8
            self._block(features * 8, features * 4, 4, 2, 1), # N x 16 x 16
            self._block(features * 4, features * 2, 4, 2, 1), # N x 32 x 32
            nn.ConvTranspose2d(features * 2, channels, 4, 2, 1), # N x 64 x 64
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, channels, features):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1), # N x 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features, features * 2, 4, 2, 1), # N x 16 x 16
            self._block(features * 2, features * 4, 4, 2, 1), # N x 8 x 8
            self._block(features * 4, features * 8, 4, 2, 1), # N x 4 x 4
            nn.Conv2d(features * 8, 1, 4, 1, 0), # N x 1 x 1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        return self.net(x)
```

this is a simple architecture but with added depth compared to the original dcgan. it uses leaky relu and batch norm as usual. a proper initialization method like the kaiming initialization can help further reduce issues with vanishing gradients, instead of the default random initialization.

then there's the loss function. the original dcgan uses a simple binary cross-entropy loss. it's not bad, but it can be sensitive to the discriminator being too strong. this can lead to the generator failing to learn anything. one approach is to use wasserstein loss with gradient penalty. this helps to stabilize the training process and avoids mode collapse. a paper that goes deep into wgan is ‘wasserstein gan’ by martin arjovsky, soumith chintala and leon bottou. this paper explains the issue well, and the implementation details as well. when i used to train dcgans i sometimes simply changed the bce loss to wasserstein loss and it worked surprisingly well.

```python
def wasserstein_loss(discriminator_output, real_labels):
    # discriminator_output: output of the discriminator
    # real_labels: tensor of 1s if real sample, 0s if fake sample
    loss = -torch.mean(discriminator_output * real_labels)
    return loss

def gradient_penalty(discriminator, real_images, fake_images, device):
    alpha = torch.rand(real_images.shape[0], 1, 1, 1, device=device)
    interpolated_images = real_images * alpha + fake_images * (1 - alpha)
    interpolated_images.requires_grad_(True)
    interpolated_scores = discriminator(interpolated_images)
    grad_outputs = torch.ones(interpolated_scores.size(), device = device)
    grad_interpolated_images = torch.autograd.grad(outputs = interpolated_scores,
                                                inputs = interpolated_images,
                                                grad_outputs = grad_outputs,
                                                create_graph = True,
                                                retain_graph=True
                                                )[0]

    grad_interpolated_images = grad_interpolated_images.view(interpolated_images.shape[0], -1)
    grad_norm = grad_interpolated_images.norm(2, dim=1)
    grad_penalty = torch.mean((grad_norm - 1)**2)
    return grad_penalty
```

this code demonstrates the basic wasserstein loss and gradient penalty, where the ‘real_labels’ should be 1 if it is a real sample and -1 if its a fake sample. in essence the gradient penalty is added to the wasserstein loss function, in this way the discriminator stays ‘reasonable’.

another important thing is the data. you need a good dataset. if your data is too low quality, the dcgan won’t learn much. i had this one project where the data was collected from various sources, and it had all sort of artifacts and inconsistencies, the dcgan was producing garbage, after i cleaned the data and made it more consistent the results were much better. consider doing a careful analysis and cleaning of your data. also make sure that the inputs are scaled between -1 and 1. the dcgan tanh output layer means that outputs are between those values, if the input is for instance 0 to 255 the generator might not work correctly. also, a diverse and large dataset is also crucial. if you have just a handful of training images you are bound to see mode collapse.

finally, the training process is crucial. don’t train for too long. it's easy to overfit, especially if the discriminator is too powerful. the generator might ‘memorize’ the training data instead of learning the actual distribution of the data. check the loss curves frequently. also start with a small learning rate, and maybe use a scheduler. it’s not a matter of training for a fixed number of epochs. i've noticed it's more about monitoring the training process, it's more of a ‘hands on’ process than a set of fixed parameters. early stopping techniques can be crucial. it's also recommended to alternate discriminator and generator training. this means that you update one network only and then the other. the original dcgan paper recommends doing something like that, but it doesn't make very clear what should be the ratio between discriminator and generator updates. this ratio is a key parameter. if you train the discriminator for too long, it will become very good at identifying fake samples, making it very difficult for the generator to learn. the usual starting point is 1 discriminator update and 1 generator update but that ratio should be fine-tuned. for instance you can update the discriminator twice and then the generator once if the discriminator is not learning fast enough. it’s an art of its own, it’s not uncommon to spend a full day only fine-tuning that ratio. i had this one time where i could not get it to converge at all, i spent days changing the network structure, the loss function and so on but in the end the problem was the update ratio between generator and discriminator. i learned a lot that day.

```python
import torch.optim as optim
# assuming generator and discriminator models and loss functions are already defined

learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999

generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas = (beta1, beta2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas = (beta1, beta2))

num_epochs = 100

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        for _ in range(disc_update_ratio): # train discriminator first disc_update_ratio times
           discriminator_optimizer.zero_grad()
           noise = torch.randn(batch_size, latent_dim, 1, 1, device = device)
           fake_images = generator(noise)
           discriminator_real_output = discriminator(real_images).view(-1)
           discriminator_fake_output = discriminator(fake_images.detach()).view(-1)
           real_labels = torch.ones_like(discriminator_real_output)
           fake_labels = torch.zeros_like(discriminator_fake_output)
           #calculate loss and update
           disc_loss_real = bce_loss(discriminator_real_output, real_labels)
           disc_loss_fake = bce_loss(discriminator_fake_output, fake_labels)
           disc_loss = (disc_loss_real + disc_loss_fake) / 2
           disc_loss.backward()
           discriminator_optimizer.step()

        # train generator
        generator_optimizer.zero_grad()
        noise = torch.randn(batch_size, latent_dim, 1, 1, device = device)
        fake_images = generator(noise)
        discriminator_fake_output = discriminator(fake_images).view(-1)
        real_labels = torch.ones_like(discriminator_fake_output)
        gen_loss = bce_loss(discriminator_fake_output, real_labels)
        gen_loss.backward()
        generator_optimizer.step()

        if i % 100 == 0:
            print(f"epoch {epoch}/{num_epochs} batch {i} discriminator loss {disc_loss} generator loss {gen_loss}")
```
this code is a simple example of how to train a dcgan with separate optimizers for the discriminator and generator, as well as separate discriminator and generator loss function updates. this approach is very useful for trying different ratios. the disc_update_ratio variable is the hyperparameter that should be adjusted to improve training.

it’s not always about the magic bullet, it’s about careful design, experimenting with different configurations and fine-tuning parameters. if you are having difficulties it’s important to keep the architecture as simple as possible, and iterate in small steps. dcgans can be a headache, sometimes you just can't get them to work no matter what you do. it’s like trying to get a cat to do what you want; there’s always some level of randomness involved. i was once explaining to my grandma how i was trying to teach a computer to produce something that could look like an image, she looked at me and said “so you are teaching the computer to paint, how cute”, i think she thought i was an artist. she never really understood what i did, but i think she was proud of me anyway.

if you want to learn more about gans and dcgans in general, i recommend the book "generative adversarial networks with pytorch" by sujay jadhav and alok gupta. it goes into deep detail about the subject. also the book "deep learning" by ian goodfellow, yoshua bengio and aaron courville is a great book to understand the fundamental concepts behind gans. and of course, read the original papers, that’s always the best way to learn about a topic.
