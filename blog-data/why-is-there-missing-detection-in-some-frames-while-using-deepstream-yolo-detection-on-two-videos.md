---
title: "Why is there Missing detection in some frames while using deepstream yolo detection on two videos?"
date: "2024-12-15"
id: "why-is-there-missing-detection-in-some-frames-while-using-deepstream-yolo-detection-on-two-videos"
---

so, missing detections in deepstream yolo, yeah, i've been there, that's a frustrating one. it's usually not a simple "it's broken" situation, more like a puzzle with a few common culprits. and yeah, it's almost always more noticeable when you're processing multiple video streams concurrently, like you're doing. i had this whole project a couple of years back, doing real-time traffic monitoring, and the missing detections were the bane of my existence for a solid couple of weeks. i thought my model was garbage, but turns out, it was more the environment where deepstream was running that caused most of the issues.

first, let's talk about the detection process itself. deepstream, at its core, is using tensorrt, right? and tensorrt models are super optimized for a specific batch size. if your deepstream pipeline is set up with an incorrect batch size for your yolo model and the incoming frame rate from your multiple videos isn't consistent (which is almost always the case when dealing with two different video files), then you'll see these inconsistent results. think of it like a fast food drive-through, if they're only taking orders for batches of 4 burgers but only get orders for 2 at a time then the orders would pile up and take longer to deliver, something similar happens here, but instead of burgers its frame that wait to get detected. this mismatch can lead to frames being skipped or just not processed correctly. i spent an embarrassing amount of time messing with batch sizes and the number of inference engines, trying different permutations. here is a piece of pseudocode on how i debugged this initially:

```pseudocode
    initialize_deepstream(batch_size=4, num_infer_engines=2)
    for each video_stream:
      while frame_available:
         frame = get_frame(video_stream)
         preprocess_frame(frame)
         add_frame_to_inference_queue(frame)

    while inference_queue_not_empty:
      frames_batch = get_batch_from_inference_queue()
      results_batch = infer(frames_batch)
      for each result in results_batch:
          postprocess_results(results)
```

what i found was that just tweaking batch size wasn't enough. you need to consider the whole pipeline. sometimes it feels like you're chasing your own tail, changing settings only to see no difference.

now the gstreamer element is another big player here. in the deepstream world everything is done via gstreamer pipelines. when dealing with two video files the chances of both gstreamer pipelines running in sync is not 100% guaranteed if not defined well. if not managed well, the buffers can become a mess. think of the gstreamer pipeline as a set of pipes carrying water, and the frames are the water. if you have a lot of pipes feeding into a single point with not enough capacity at the joining node, you can have bottlenecks and overflow. sometimes the frames just get "lost in the pipeline," meaning the next element in the processing chain never gets to see them. gstreamer buffers are tricky, and i had to use a combination of queue elements and buffer probing to get a clear view of the buffer flow. it's a bit like trying to debug a plumbing system with a flashlight – it takes time and patience. i remember i added a simple queue to my gstreamer pipeline to better see if i had any buffers being dropped or overflowed, something like this:

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

pipeline_str = """
    filesrc location={video_file_path} !
    decodebin !
    queue !
    nvstreammux name=mux batch-size=1 width=1280 height=720 !
    nvinfer config-file-path={yolo_config_path} !
    fakesink sync=0
""".format(video_file_path="video1.mp4", yolo_config_path="yolov5.txt")

pipeline = Gst.parse_launch(pipeline_str)
pipeline.set_state(Gst.State.PLAYING)
#then the pipeline is managed and handled
# i have omitted this part because its context dependant, but this is how a pipeline can be created and how the queue element is included
#this also includes using the nvstreammux to have a uniform input for your inference.
#you need to manage the pipeline to have it running.
```

sometimes its a model issue too, but that is very rare and most of the time it is never the models fault at all. i did have a moment where i just doubted my model so much so i did a complete model re-training and got almost the same results! i learned it was not the model, but again with the inference engine and batch sizes. but if you are training from scratch, then you might have something to debug there, but lets assume that your model was trained well and the issues are pipeline related.

now, let's talk about system resources. when you're running two deepstream pipelines simultaneously, you're essentially asking your system to do a lot at the same time, and deepstream is very greedy in resources, especially memory and gpu utilization. if you have the gpu barely supporting the load you can have missing frames because if the gpu resources are fully saturated it will not process frames quickly enough and they can get dropped. it is important to consider the gpu load. i've had situations where i thought i was doing everything perfectly, but the problem was i had another program in the background hogging the gpu, leading to those annoying missing detections. resource monitor tools, like `nvidia-smi` is your best friend here. a constant high gpu load might tell you that you need to optimize your pipeline, use a bigger gpu or scale down your pipeline. or, even worse, it could be a thermal issue with your machine. i once almost melted my old laptop because of this!

also, another important note, is the data being sent to your inference engine. if you are doing any kind of preprocessing, before sending data to the inference engine, the operation must be quick and cheap. for example using opencv resizes can lead to bottlenecks if the resources are limited. make sure all your processing before inference is as minimal as possible. for example converting image color spaces should be done only when required or after the inference engine operation is done. all this depends on how your pipeline is configured.

also, video decoding can be a bottleneck. using hardware decoding can drastically reduce cpu overhead, but sometimes software decoding is what is used. always prefer using the hardware decoding whenever possible.

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

pipeline_str = """
    filesrc location={video_file_path} !
    decodebin !
    nvvidconv ! #hardware acceleration for video conversion
    queue !
    nvstreammux name=mux batch-size=1 width=1280 height=720 !
    nvinfer config-file-path={yolo_config_path} !
    fakesink sync=0
""".format(video_file_path="video1.mp4", yolo_config_path="yolov5.txt")

pipeline = Gst.parse_launch(pipeline_str)
pipeline.set_state(Gst.State.PLAYING)
#similar code as the previous snippet but the `nvvidconv` will force hardware video conversion
#this helps a lot when dealing with multiple streams
```
one time my team was scratching our heads for days because of this and we learned that the storage drive where the videos were was not able to deliver the video frames at the rate required for real time decoding and inference, that was a very simple fix after analyzing disk io usage.

so, to summarize my experience with this issue, it's rarely one thing, it's a mix. batch sizes, gstreamer buffers, system resources, preprocessing bottlenecks, and disk access are the usual suspects. it's almost never a code issue (if your model is ok) but its more about how you're using the infrastructure. there is nothing wrong with yolo either, because it works like a charm. deepstream is just very sensitive to pipeline management issues.

in terms of resources, i would recommend reading the deepstream documentation very carefully. there is a lot of hidden information there. also, the gstreamer documentation is a must, the deeper you go the better you'll become at building custom pipelines. for general gpu optimization, i recommend reading nvidia’s developer guides, it is mostly about the theory but it will give you an idea on how to debug gpu issues. also, there are lots of research papers on efficient video processing, which could be helpful, it is not that they are going to tell you step by step how to solve the problem but they provide a theoretical background that helps. reading tensorrt documentation might be a little overboard, but having a grasp of how it works is essential too.

i hope that gives you an idea of where to start looking. don't give up, you'll get it figured out eventually. and if not, well, there's always the option of just adding more gpus, *right?* (just kidding!)
