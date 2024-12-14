---
title: "Why is a Confusion matrix for instance segmentation returning no FN?"
date: "2024-12-14"
id: "why-is-a-confusion-matrix-for-instance-segmentation-returning-no-fn"
---

hey, so you're seeing no false negatives (fn) in your confusion matrix for instance segmentation, huh? that's a bit... unusual. i've been down that road, trust me. it usually points to a problem with how you're calculating things, not necessarily that your model is *that* perfect. let's break it down.

first things first, instance segmentation is more complex than simple classification or even semantic segmentation. we are dealing with multiple objects, their masks *and* their classification. this means we have to keep track of *object matching*, and this is where the fn can easily go missing if you're not careful.

i remember way back when, i was working on a project for autonomous vehicles, trying to segment pedestrians in real-time. i was so happy with my initial results because they looked visually perfect, no missing pedestrians i thought. my confusion matrix showed the same, zero fns. i started thinking i was a genius or something. turns out, my genius was actually just a bad implementation of the matrix calculation. i was simply not accounting for cases where my model missed an object completely. instead, i was only comparing predicted and gt masks based on *existing* predicted instances and matching them with the *closest* ground truth instance. objects that were not detected at all were simply ignored in the confusion matrix calculation, creating the illusion of zero fns.

the key is that you need to ensure you have a method to account for ground truth instances for which there are *no matching* predicted instances. these are the missing objects, and hence, the actual false negatives. the predicted instances with no matching ground truth are the false positives (fp) by the way.

let me explain further. the way you calculate your confusion matrix for instance segmentation relies on some sort of matching process. you can't just directly compare the masks pixel by pixel because you're dealing with *multiple* objects. usually we use intersection over union (iou) to match gt instances with the predicted ones. it's not a simple 1-to-1 correspondence.

here is a simple example of how to compute the confusion matrix, the correct way.

```python
import numpy as np

def compute_confusion_matrix(predictions, ground_truths, iou_threshold=0.5):
    """
    computes confusion matrix for instance segmentation.

    Args:
    predictions: list of dictionaries containing
        {'masks': np.array of shape (num_masks, height, width),
        'labels': np.array of shape (num_masks,)}
    ground_truths: list of dictionaries containing
        {'masks': np.array of shape (num_masks, height, width),
        'labels': np.array of shape (num_masks,)}
    iou_threshold: iou used for matching instances

    Returns:
    dictionary of tp, fp, fn count per class
    """

    num_classes = max(max(pred['labels'] if pred['labels'].size else [-1])
                    for pred in predictions)
    num_classes = max(num_classes, max(max(gt['labels'] if gt['labels'].size else [-1])
                                    for gt in ground_truths)) + 1

    tp = {c: 0 for c in range(num_classes)}
    fp = {c: 0 for c in range(num_classes)}
    fn = {c: 0 for c in range(num_classes)}
    
    for pred, gt in zip(predictions, ground_truths):

      if not gt['masks'].size or not pred['masks'].size:
        # handle empty cases
        for label in gt['labels']:
          fn[label] += 1
        for label in pred['labels']:
          fp[label] +=1
        continue

      gt_masks = gt['masks']
      gt_labels = gt['labels']
      pred_masks = pred['masks']
      pred_labels = pred['labels']

      matched_gt = [False] * len(gt_labels)
      
      for i, pred_mask in enumerate(pred_masks):
          best_iou = 0
          best_match_idx = -1

          for j, gt_mask in enumerate(gt_masks):
              iou = calculate_iou(pred_mask, gt_mask)
              if iou > best_iou and not matched_gt[j]:
                  best_iou = iou
                  best_match_idx = j

          if best_iou >= iou_threshold:
              if gt_labels[best_match_idx] == pred_labels[i]:
                  tp[pred_labels[i]] += 1
              else:
                  fp[pred_labels[i]] += 1
              matched_gt[best_match_idx] = True
          else:
              fp[pred_labels[i]] += 1
      
      for j, label in enumerate(gt_labels):
        if not matched_gt[j]:
            fn[label] += 1
    
    return {'tp': tp, 'fp': fp, 'fn': fn}


def calculate_iou(mask1, mask2):
  """
  simple iou implementation for the sake of demonstration
  """
  intersection = np.logical_and(mask1, mask2).sum()
  union = np.logical_or(mask1, mask2).sum()
  if union == 0:
    return 0
  return intersection / union
```

in this example, we iterate through each prediction, and for every predicted instance, we look for the ground truth instance with the highest iou. if this iou is above a given threshold, and if the labels match, it's a true positive. if the labels don't match, it's a false positive. if *no matching* ground truth instance is found, then it's also a false positive. *after* this process, we loop through ground truth masks and identify those that were not matched, these are the false negatives. if the ground truth or the predicted instances are empty the code handles them gracefully.

another point to think about, if you're using a library or framework, is how it handles the matching of instances when calculating the confusion matrix. some libraries might offer a simple confusion matrix for every pixel instead of instances, and that's not the matrix we are after. make sure you are looking for the confusion matrix for instances in particular. check the documentation carefully. sometimes, they make the matching process internally, and it might not be doing what you expect. i once spent hours debugging only to find that the library was using a default iou threshold of 0.0, effectively considering almost everything a match.

let me give you another code snippet, this one shows how to generate dummy data to test the confusion matrix calculation. it's very helpful to test if your function works as expected

```python
import numpy as np

def generate_dummy_data(num_samples, height, width, max_instances_per_image, num_classes):
    """generates dummy data for instance segmentation testing"""
    predictions = []
    ground_truths = []

    for _ in range(num_samples):
        pred_masks = []
        pred_labels = []
        num_pred_instances = np.random.randint(max_instances_per_image)
        for _ in range(num_pred_instances):
            mask = np.random.randint(0, 2, size=(height, width), dtype=bool)
            label = np.random.randint(num_classes)
            pred_masks.append(mask)
            pred_labels.append(label)
        predictions.append({'masks': np.array(pred_masks), 'labels': np.array(pred_labels)})

        gt_masks = []
        gt_labels = []
        num_gt_instances = np.random.randint(max_instances_per_image)
        for _ in range(num_gt_instances):
            mask = np.random.randint(0, 2, size=(height, width), dtype=bool)
            label = np.random.randint(num_classes)
            gt_masks.append(mask)
            gt_labels.append(label)
        ground_truths.append({'masks': np.array(gt_masks), 'labels': np.array(gt_labels)})

    return predictions, ground_truths

```

i find it helpful to write a couple of unit tests too. something like, create two instances that match perfectly, the confusion matrix should be 1tp and no fp nor fn. create one predicted instance, and two ground truths non overlapping, the matrix should show 1fp and 2fn. you get the idea, and you can go on with those simple tests. here is a small code example with the test i described

```python
import unittest

class TestConfusionMatrix(unittest.TestCase):

    def test_perfect_match(self):
        # test case 1: perfect match
        predictions = [{'masks': np.array([np.ones((10, 10), dtype=bool)]), 'labels': np.array([0])}]
        ground_truths = [{'masks': np.array([np.ones((10, 10), dtype=bool)]), 'labels': np.array([0])}]
        result = compute_confusion_matrix(predictions, ground_truths)
        self.assertEqual(result['tp'][0], 1)
        self.assertEqual(result['fp'][0], 0)
        self.assertEqual(result['fn'][0], 0)

    def test_one_prediction_two_ground_truths(self):
        # test case 2: one prediction and two ground truth non overlapping
        predictions = [{'masks': np.array([np.ones((10, 10), dtype=bool)]), 'labels': np.array([0])}]
        ground_truths = [{'masks': np.array([np.ones((10, 10), dtype=bool), np.zeros((10,10), dtype=bool) ]) , 'labels': np.array([0,0])}]
        result = compute_confusion_matrix(predictions, ground_truths)
        self.assertEqual(result['tp'][0], 1)
        self.assertEqual(result['fp'][0], 0)
        self.assertEqual(result['fn'][0], 1)
        

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```

for resources, i’d recommend looking at the papers on common instance segmentation benchmarks like coco and cityscapes. they often detail how the metrics are calculated, and you can see different iou based matching algorithms being discussed there. also, "computer vision: algorithms and applications" by richard szeliski is a great resource for the math behind these things. the official pytorch docs usually have a more simple and intuitive way of showing these metrics, and its an alternative if you are using that library.

i remember a particularly bad experience where i was convinced that my model was failing at detection only to find that i had accidentally swapped the ground truth and prediction masks in my evaluation script. that was a long day. it's always the simplest things that get you, like missing a closing parenthesis. it's a bit like trying to find a specific sock in the dryer, you keep looking in all the obvious places, then you find it stuck on the lint filter.

so to recap, double-check your matching algorithm, make sure you are accounting for unmatched ground truths, verify your iou threshold, and also check if the library you're using is handling things the way you expect. create unit tests with different cases, and finally, never rule out that there could be a swapped variable somewhere, that was my lesson that day.
good luck with your debugging, i hope this helps.
