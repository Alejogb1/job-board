---
title: "How do I plot the validation loss in Tensorboard for object detection?"
date: "2024-12-14"
id: "how-do-i-plot-the-validation-loss-in-tensorboard-for-object-detection"
---

alright, so you're looking to visualize your validation loss during object detection training in tensorboard, that's a pretty standard need, and i've been down this road more times than i care to count. it's crucial to monitor this to avoid overfitting and ensure your model is actually learning something useful, not just memorizing the training data.

i remember one particular project back in my early days, it involved training a custom object detector to identify different types of defects on silicon wafers. we had this massive dataset of high-resolution images, and our initial setup, while it could achieve impressive results on the training set, absolutely crumbled when fed with new data, we were basically training a very expensive and useless image memorization machine. it became very obvious we were overfitting. the initial logs, just numbers scrolling by were simply not cutting it, i needed graphs! it felt like staring at a wall of code without any understanding. i learnt the hard way back then: real-time visualization of metrics is a must, it saves you a ton of time and helps you understand exactly what is going on under the hood.

the key to plotting validation loss in tensorboard is to make sure you're actually calculating and logging it during your training process. most object detection frameworks have mechanisms for tracking both training and validation losses separately. the specific implementation will vary slightly depending on what library you're using (pytorch, tensorflow, etc.), but the core idea remains consistent: calculating the loss on a held-out validation dataset and pushing this value to tensorboard.

i'll show you some common approaches. imagine you are using pytorch or a similar framework and you want to track not only your training loss but also your validation loss, so you can monitor the generalization performance of your model during the learning phase.

here’s a snippet showcasing how you might do it in a simplified scenario. in this case we are not training any actual object detection model rather simulating it for the sake of clarity. it will generate some random fake losses, but it will show you the basics of how to use tensorboard with validation loss metrics.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random

# create a dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

# generate dummy data
def generate_dummy_data(num_samples):
    x = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    return x, y

# training loop
def train(epochs, train_data, validation_data, model, optimizer, criterion, writer):
    for epoch in range(epochs):
        model.train() # setting the model to training mode
        train_x, train_y = train_data
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        writer.add_scalar('training loss', loss.item(), epoch) # log training loss for the current epoch

        model.eval() # setting the model to evaluation mode
        with torch.no_grad(): # do not compute gradients during validation
          val_x, val_y = validation_data
          val_output = model(val_x)
          val_loss = criterion(val_output, val_y)
          writer.add_scalar('validation loss', val_loss.item(), epoch) #log validation loss for the current epoch

        print(f"epoch {epoch+1}/{epochs}, training loss: {loss.item():.4f} validation loss: {val_loss.item():.4f}")

if __name__ == '__main__':
    # hyper parameters
    num_epochs = 100
    learning_rate = 0.001
    num_train_samples = 500
    num_val_samples = 100

    # initialize the model
    dummy_model = DummyModel()
    optimizer = optim.Adam(dummy_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # generate training and validation data
    train_x, train_y = generate_dummy_data(num_train_samples)
    validation_x, validation_y = generate_dummy_data(num_val_samples)
    train_data = (train_x, train_y)
    validation_data = (validation_x, validation_y)

    # initialize the tensorboard writer
    writer = SummaryWriter()

    # train the model
    train(num_epochs, train_data, validation_data, dummy_model, optimizer, criterion, writer)
    print('training completed!')
    writer.close()
```

in the code above the key line is `writer.add_scalar('validation loss', val_loss.item(), epoch)`. this tells tensorboard to log a scalar value (your validation loss) at each epoch, associating it with the `validation loss` tag. tensorboard will then use this to plot the validation loss over epochs.

remember to install `tensorboard` via pip: `pip install tensorboard`. after you run this code. then you can visualize the results in a tensorboard instance running the command `tensorboard --logdir=runs` in your console. if `runs` is the default directory where `SummaryWriter` saves the logs. navigate to `http://localhost:6006/` in your browser, assuming default port.

now if you are using tensorflow here is another example using a similar approach but in tensorflow:

```python
import tensorflow as tf
import numpy as np
import datetime

# create a dummy model
class DummyModel(tf.keras.Model):
  def __init__(self):
    super(DummyModel, self).__init__()
    self.dense = tf.keras.layers.Dense(1)
  def call(self, inputs):
    return self.dense(inputs)

# generate dummy data
def generate_dummy_data(num_samples):
    x = np.random.rand(num_samples, 10).astype(np.float32)
    y = np.random.rand(num_samples, 1).astype(np.float32)
    return x, y

# training loop
def train(epochs, train_data, validation_data, model, optimizer, loss_fn, summary_writer):
  for epoch in range(epochs):
    # training step
    train_x, train_y = train_data
    with tf.GradientTape() as tape:
        output = model(train_x)
        loss = loss_fn(train_y, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    with summary_writer.as_default():
      tf.summary.scalar('training loss', loss, step=epoch) # log training loss for the current epoch

    # validation step
    val_x, val_y = validation_data
    val_output = model(val_x)
    val_loss = loss_fn(val_y, val_output)

    with summary_writer.as_default():
      tf.summary.scalar('validation loss', val_loss, step=epoch) #log validation loss for the current epoch

    print(f"epoch {epoch+1}/{epochs}, training loss: {loss.numpy():.4f}, validation loss: {val_loss.numpy():.4f}")

if __name__ == '__main__':
  # hyper parameters
  num_epochs = 100
  learning_rate = 0.001
  num_train_samples = 500
  num_val_samples = 100

  # initialize the model
  dummy_model = DummyModel()
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  loss_function = tf.keras.losses.MeanSquaredError()

  # generate training and validation data
  train_x, train_y = generate_dummy_data(num_train_samples)
  validation_x, validation_y = generate_dummy_data(num_val_samples)
  train_data = (train_x, train_y)
  validation_data = (validation_x, validation_y)

  # setup tensorboard summary writer
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = 'logs/' + current_time
  summary_writer = tf.summary.create_file_writer(log_dir)

  # train the model
  train(num_epochs, train_data, validation_data, dummy_model, optimizer, loss_function, summary_writer)
  print('training completed!')
```
here the `tf.summary.scalar('validation loss', val_loss, step=epoch)` does the magic of logging the validation loss. remember the tensorboard instance can be visualized by running `tensorboard --logdir logs`. assuming `logs` is your target directory.

and for a more concrete case, let's assume you are using an object detection specific library like `detectron2`. this one already has integrated tensorboard support, but lets assume you want to log another validation metric, for example a custom iou, but with a similar logging approach:

```python
import torch
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetCatalog, MetadataCatalog
import os
import random
import numpy as np
from pycocotools.coco import COCO

# function to simulate your data generation process for the purposes of this example
def simulate_coco_data(num_images, num_objects_per_image, image_size):
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    for i in range(num_images):
        image_id += 1
        image = {
            'id': image_id,
            'file_name': f"image_{i}.jpg",
            'width': image_size,
            'height': image_size
        }
        images.append(image)

        for j in range(random.randint(1, num_objects_per_image)):
            x = random.randint(0, image_size - 50)
            y = random.randint(0, image_size - 50)
            width = random.randint(20, 50)
            height = random.randint(20, 50)

            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'bbox': [x, y, width, height],
                'area': width * height,
                'iscrowd': 0,
                'category_id': 1, # assuming single category, for simplicity
            }
            annotations.append(annotation)
            annotation_id += 1

    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'object'}]
    }
    return coco_format

def register_coco_dataset(name, coco_data):
    DatasetCatalog.register(name, lambda: coco_data)
    MetadataCatalog.get(name).set(thing_classes=['object'])

class CustomTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
      return COCOEvaluator(dataset_name, output_dir=output_folder)

  def test(self):
      return None

  def _get_val_loader(self):
      return self.build_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])

  def _calculate_iou(self, predictions, gt):
    iou_sum = 0
    for p, g in zip(predictions, gt):
      p_box = p["bbox"]
      g_box = g["bbox"]

      x1 = max(p_box[0], g_box[0])
      y1 = max(p_box[1], g_box[1])
      x2 = min(p_box[0] + p_box[2], g_box[0] + g_box[2])
      y2 = min(p_box[1] + p_box[3], g_box[1] + g_box[3])

      intersection = max(0, x2-x1) * max(0, y2-y1)

      p_area = p_box[2] * p_box[3]
      g_area = g_box[2] * g_box[3]

      union = p_area + g_area - intersection
      iou = intersection/ union if union > 0 else 0
      iou_sum += iou
    if(len(predictions) > 0):
      return iou_sum/len(predictions)
    return 0


  def run_step(self):
    super().run_step()
    storage = self.storage
    if self.iter % 20 == 0: #logging every 20 iterations for demo purposes
      model = self.model
      val_loader = self._get_val_loader()
      with torch.no_grad():
        model.eval()
        iou_sum = 0
        for batch in val_loader:
          images = self.preprocess_image(batch)
          outputs = model(images)

          for i, output in enumerate(outputs):
             predictions = output["instances"].to("cpu").get("pred_boxes").tensor.tolist()
             gt = batch[i]["instances"].gt_boxes.tensor.tolist()

             iou = self._calculate_iou(predictions, gt)
             iou_sum += iou
        if len(val_loader.dataset) > 0:
          avg_iou = iou_sum / len(val_loader.dataset)
          storage.put_scalar("val_iou", avg_iou)
        else:
          storage.put_scalar("val_iou", 0)

        model.train()

if __name__ == "__main__":

    #generate some dummy data
    num_images = 100
    num_objects_per_image = 5
    image_size = 256
    coco_train = simulate_coco_data(num_images, num_objects_per_image, image_size)
    coco_val = simulate_coco_data(num_images//2, num_objects_per_image, image_size)

    #register the datasets
    train_dataset_name = "train_data"
    val_dataset_name = "val_data"
    register_coco_dataset(train_dataset_name, coco_train)
    register_coco_dataset(val_dataset_name, coco_val)


    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 100
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = "output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
```
in this last snippet, we are extending the `DefaultTrainer` class and adding our own custom metrics, in this case, a simple average iou. detectron2 takes care of the tensorboard integration, so we are just logging the new `val_iou` metric using `storage.put_scalar("val_iou", avg_iou)`. when running this you should be able to see the `val_iou` and any other metrics logged in tensorboard. note this example is much more involved and complex than the previous ones and the `cfg.MODEL.WEIGHTS` value refers to weights from the official model zoo for detectron2.

also. i've seen people accidentally log their training loss as the validation loss more than once, so double check you're feeding in the validation data. the easiest way to verify is to graph them both in tensorboard and check their individual behavior, when training goes well, your training loss should tend to go down as expected and your validation loss may plateau or start to increase if overfitting, a typical sign of a model that is just learning the training data by heart.

one final note, remember that you should be always monitoring your hardware utilization, if you are not using your gpu efficiently, a simple tweak on the dataloader might solve that, otherwise you could be sitting there for a long time just waiting for the model to learn, or worse, you may be overfitting your model with a really slow training process. it's like trying to fill a swimming pool with a drinking straw. just not efficient, *rimshot*.

regarding resources, i'd suggest "deep learning" by ian goodfellow, yoshua bengio, and aaron courville; that's like the deep learning bible. another good one is "programming pytorch for deep learning" by ian pointer if you're leaning towards pytorch or "hands-on machine learning with scikit-learn, keras & tensorflow" by aurélien géron, if you are more a tensorflow person. you'll find more technical descriptions, but the principle of logging validation metrics remains consistent.

remember: understanding and visualizing your validation loss is not just about debugging, it's about actually building robust and reliable machine learning systems.
