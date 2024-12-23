---
title: "How can I monitor AutoGluon neural network training progress on TensorBoard?"
date: "2024-12-23"
id: "how-can-i-monitor-autogluon-neural-network-training-progress-on-tensorboard"
---

Alright, let's talk about monitoring AutoGluon neural network training progress with TensorBoard. It's a crucial step for any serious model development, and while AutoGluon abstracts away much of the low-level details, it doesn’t preclude leveraging tools like TensorBoard to gain deeper insight into the training process. I've found myself relying on this approach often, especially when fine-tuning or debugging particularly complex models. There was this one project back in '21 where we had a massive multi-modal dataset, and without TensorBoard, interpreting the results would have been like navigating a maze blindfolded.

The key here isn't some secret sauce or undocumented feature; it's understanding how AutoGluon's `Trainer` class works in conjunction with PyTorch Lightning, which is often under the hood. TensorBoard logging, essentially, becomes a byproduct of this interaction. AutoGluon, in many cases, uses PyTorch Lightning internally. PyTorch Lightning, fortunately, provides a seamless integration with TensorBoard. So the path to making this work involves ensuring the underlying PyTorch Lightning trainer instance is set up to emit these logs. Let's get down to brass tacks.

Fundamentally, you need to hook into the callback system provided by Lightning. AutoGluon's abstractions make it a bit less direct than a raw PyTorch Lightning setup, but the same principle applies: inject a callback that handles logging. I typically accomplish this by creating a custom callback and injecting it during the fit process. Here's how that looks in code:

```python
import autogluon.tabular as ag
import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd

class TensorBoardCallback(Callback):
    def __init__(self, log_dir="lightning_logs"):
        super().__init__()
        self.logger = TensorBoardLogger(save_dir=log_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
              self.logger.experiment.add_scalar(key, value.item(), global_step=trainer.current_epoch)
            else:
              self.logger.experiment.add_scalar(key, value, global_step=trainer.current_epoch)


    def on_validation_epoch_end(self, trainer, pl_module):
       for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
              self.logger.experiment.add_scalar(f'val_{key}', value.item(), global_step=trainer.current_epoch)
            else:
              self.logger.experiment.add_scalar(f'val_{key}', value, global_step=trainer.current_epoch)


def train_with_tensorboard(train_data, label_col, tb_log_dir="lightning_logs", **ag_kwargs):
    tb_callback = TensorBoardCallback(log_dir=tb_log_dir)
    predictor = ag.TabularPredictor(label=label_col)
    predictor.fit(
        train_data,
        callbacks=[tb_callback],
        **ag_kwargs
    )
    return predictor

if __name__ == '__main__':
    # Generate a simple example dataset
    data = {
        'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature_2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target':    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Train with TensorBoard logging enabled
    predictor = train_with_tensorboard(df, label_col='target', time_limit=10) # shortened time limit for example
    print("Training finished, check TensorBoard logs in lightning_logs.")
```

In this code, we define `TensorBoardCallback`, which inherits from PyTorch Lightning’s `Callback`. During the `on_train_epoch_end` and `on_validation_epoch_end` functions, we grab the metrics reported to the trainer. We then pass these metrics to the tensorboard logger, and they get written out in a way that TensorBoard understands. The `train_with_tensorboard` function instantiates this callback and injects it into the `fit` process through the `callbacks` argument. This allows us to use TensorBoard with any AutoGluon training session by just calling this function instead of `predictor.fit`.

Now, after you run this, you'd launch TensorBoard from the command line, something like:

```bash
tensorboard --logdir=lightning_logs
```
... and you can then navigate to the visualization in your web browser.

It's worth noting that AutoGluon uses multiple models during training. To isolate the neural network logging, we often need to be specific when setting parameters for the fit function, as well as when analyzing the results within TensorBoard itself. Typically, you’d look for the specific metric names associated with the neural network. Let's say you want to specifically see how the training loss changes only for the neural net. We'd achieve that by specifying which model to train when invoking the fit function:

```python
def train_neuralnet_with_tensorboard(train_data, label_col, tb_log_dir="lightning_logs",  **ag_kwargs):
    tb_callback = TensorBoardCallback(log_dir=tb_log_dir)
    predictor = ag.TabularPredictor(label=label_col)

    # Train only the neural network model
    predictor.fit(
        train_data,
        presets='best_quality',
        hyperparameters={'NN': {}}, # use only neural network
        callbacks=[tb_callback],
        **ag_kwargs
    )

    return predictor

if __name__ == '__main__':
    # Generate a simple example dataset
    data = {
        'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature_2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target':    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Train only the neural net with tensorboard logging
    predictor = train_neuralnet_with_tensorboard(df, label_col='target', time_limit=10) # shortened time limit for example
    print("Training finished, check TensorBoard logs in lightning_logs.")
```

Here, we've added `hyperparameters={'NN': {}}` to `predictor.fit`, which forces AutoGluon to use *only* the neural network model. As a consequence, you'll now only see logs from this specific model in your TensorBoard output. You'd then explore the graphs within the "lightning_logs" directory in tensorboard as above. The scalar graphs for loss and other metrics would provide the insight you're looking for.

Furthermore, if we want to examine specific parts of the training, we can also add metrics to the logs. This might be useful if you are troubleshooting a peculiar issue or are trying to validate your own training logic. Here is an example where we log gradient norms from each layer:

```python
class TensorBoardCallbackAdvanced(Callback):
    def __init__(self, log_dir="lightning_logs"):
        super().__init__()
        self.logger = TensorBoardLogger(save_dir=log_dir)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad)
                self.logger.experiment.add_scalar(f"grad_norm/{name}", grad_norm, global_step=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
              self.logger.experiment.add_scalar(key, value.item(), global_step=trainer.current_epoch)
            else:
              self.logger.experiment.add_scalar(key, value, global_step=trainer.current_epoch)


    def on_validation_epoch_end(self, trainer, pl_module):
       for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
              self.logger.experiment.add_scalar(f'val_{key}', value.item(), global_step=trainer.current_epoch)
            else:
              self.logger.experiment.add_scalar(f'val_{key}', value, global_step=trainer.current_epoch)

def train_advanced_with_tensorboard(train_data, label_col, tb_log_dir="lightning_logs", **ag_kwargs):
    tb_callback = TensorBoardCallbackAdvanced(log_dir=tb_log_dir)
    predictor = ag.TabularPredictor(label=label_col)
    predictor.fit(
        train_data,
        callbacks=[tb_callback],
        **ag_kwargs
    )
    return predictor

if __name__ == '__main__':
    # Generate a simple example dataset
    data = {
        'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature_2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target':    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Train with TensorBoard logging enabled
    predictor = train_advanced_with_tensorboard(df, label_col='target', time_limit=10) # shortened time limit for example
    print("Training finished, check TensorBoard logs in lightning_logs.")
```

In this example, we add an `on_before_optimizer_step` function to our callback. In it, we compute the gradient norms of each layer and save them to the TensorBoard log files. This functionality could be extended for other uses, such as examining activations or internal layer outputs during training.

For more in-depth understanding, I'd highly recommend delving into the documentation for PyTorch Lightning, focusing on their callback system and integration with TensorBoard. Specifically, exploring how logging works in PyTorch Lightning will make it easier to debug these examples. Additionally, reading the original AutoGluon papers (e.g., "AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data") can also help gain a deeper understanding of the frameworks design decisions, which will make debugging issues easier. Specifically, I have found the section on integration with third-party libraries to be very useful. Finally, a good resource is also the official TensorBoard documentation, which helps with understanding the underlying format and how it interacts with the logger callback.

In closing, monitoring AutoGluon with TensorBoard isn't as straightforward as in some frameworks, but it's definitely achievable by understanding and leveraging the PyTorch Lightning integration. It’s an essential practice that will significantly enhance your ability to understand, debug, and ultimately improve your models.
