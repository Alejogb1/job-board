---
title: "logging images?"
date: "2024-12-13"
id: "logging-images"
---

Okay so logging images huh I’ve been there done that got the t-shirt and a few gray hairs along the way It sounds simple but it quickly spirals into a data management and performance pit I remember this one time back in 2012 when I was messing around with some early computer vision stuff for a prototype automated sorting system We were using a prototype camera setup pushing frames at like 30fps and we thought hey let’s just log everything because why not Turns out why not was a really good question

First things first you’re going to need to figure out how you want to store these things Images are not text they're binary blobs And shoving binary data into a regular text log file is just asking for trouble it leads to corrupted files it takes a lot of resources to convert to strings and back and it's frankly a nightmare to debug

So that’s option one you could hypothetically base64 encode the image data to store it as text But just don't Even with the overhead removed you're increasing file size and slowing down your write speeds by a lot A simple image might be 100KB but base64 converts it to roughly 133KB on top of that this means a performance hit

Here’s a better idea consider using a different file per image with the log file only storing a path or a ID that maps to it. Log files can then be managed without affecting the image file storage

Let's look at a Python code snippet that does just that using Python Imaging Library (Pillow)

```python
import os
import time
from PIL import Image
import logging

# Configure logging
logging.basicConfig(filename='image_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_image(image_data, log_dir="image_store"):
    """Logs an image to a file and records the path in a log file."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = int(time.time() * 1000) # Use milliseconds for more precision
    filename = f"image_{timestamp}.png"
    filepath = os.path.join(log_dir, filename)

    try:
        img = Image.frombytes('RGB', (128,128), image_data)  # Assuming your data is 128x128 RGB bytes
        img.save(filepath)
        logging.info(f"Image logged at: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error logging image: {e}")
        return None


# Example usage
if __name__ == '__main__':
    # Example binary image data
    dummy_image_data = os.urandom(128 * 128 * 3) # 128x128 RGB image

    # Log the image
    logged_path = log_image(dummy_image_data)

    if logged_path:
        print(f"Image logged at {logged_path}")
    else:
        print("Failed to log the image.")
```
This approach separates the logs from the images The log file `image_log.log` would contain the path to where the images are stored along with a timestamp and other details The image is stored as `image_{timestamp}.png` on disk. Note that in this example I used a generated binary data for the image but it is obviously required that you replace the dummy image data with actual image data

Now performance is a factor How fast you're generating images and how fast you need to log them is key Remember that disk I/O can be a bottleneck especially with high framerates and if the image is large So do not go for the most complex compression algorithm here I learned it the hard way back then

Another important point do not use synchronous logging for images and image paths I have learned it the very hard way. It will slow down your main processing thread especially if your writing to slow disks or an external hard disk. Always log asynchronously using a separate thread or a process. It keeps your primary application responsive.

Here's another snippet but this time with asynchronous logging using `threading` in Python:

```python
import os
import time
from PIL import Image
import logging
import threading
import queue

# Configure logging
logging.basicConfig(filename='image_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

image_queue = queue.Queue()

def image_logger(log_dir="image_store"):
    """Consumes images from the queue and logs them."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    while True:
        image_data = image_queue.get()
        if image_data is None:
            break # To exit when done
        try:
          timestamp = int(time.time() * 1000)
          filename = f"image_{timestamp}.png"
          filepath = os.path.join(log_dir, filename)
          img = Image.frombytes('RGB', (128,128), image_data)
          img.save(filepath)
          logging.info(f"Image logged at: {filepath}")
        except Exception as e:
            logging.error(f"Error logging image: {e}")
        finally:
          image_queue.task_done()


# Initialize logger thread
logger_thread = threading.Thread(target=image_logger, daemon=True)
logger_thread.start()


def log_image_async(image_data):
    """Add image to logging queue"""
    image_queue.put(image_data)


# Example usage
if __name__ == '__main__':
    # Example binary image data
    dummy_image_data = os.urandom(128 * 128 * 3)  # 128x128 RGB image

    log_image_async(dummy_image_data)
    log_image_async(os.urandom(128 * 128 * 3)) # Log more image data for testing
    log_image_async(os.urandom(128 * 128 * 3))

    image_queue.join()
    image_queue.put(None)
    logger_thread.join() # Wait for the logger thread to finish

    print("Async image logging complete.")

```

Using a queue and a separate thread lets your primary thread continue its operations while the logger thread handles saving the images and writing to the log file. This approach improves the program's responsiveness and efficiency. Now remember this is just an example and you will have to modify the image storage and queueing based on your needs especially if you are going to be having heavy concurrency issues.

Another critical thing is storage We had issues on one system where the disk ran out of space because of the logging and the application crashed. Make sure you've got sufficient storage and maybe have a process for purging old images if you're not using them. We’ve also used object storage services and cloud drives for storage. You might also consider lossless compression codecs like PNG or lossless webp if space is a major concern. Also think about image file metadata. You might want to add to the logging file details like when the image was captured source device and other details. That could mean reading metadata like EXIF which might be important for debugging later on.

One more thing image format and encoding is very important too. We once had an issue with a proprietary format that wasn’t handled properly by the imaging library and that lead to hours of debugging. It also depends on what your requirements are Are you just logging to see what's happening or do you intend to reuse the logged images later for something like model training? This will drive whether you use lossy or lossless compression etc.

This one time I was looking into a camera setup and we had an issue with image quality turns out the camera's encoding was wrong I was pulling my hair out for hours until I double-checked the encoding parameters. Always check your encoding parameters is my final advise

Now for the formats I mentioned PNG is usually good for lossless storage if you're not space-constrained but PNGs are larger in size I would suggest having a look at the JPEG XL format too. In many cases it's smaller than JPEG and also lossless too. You should choose a format that balances compression and image quality. It always depends on your use case.

For structured logging consider writing the log data itself in JSON format so that it can be ingested into log analysis tools like ELK Stack (Elasticsearch Logstash Kibana) or similar platforms. Now this means a little overhead for processing but provides good benefits in analysis.

Here's a final snippet showing logging with JSON. This is more of an illustration and you have to do a lot more based on the complexity of your logging infrastructure:

```python
import os
import time
from PIL import Image
import logging
import json
import threading
import queue

# Configure logging
logging.basicConfig(filename='image_log.json', level=logging.INFO, format='%(message)s')

image_queue = queue.Queue()

def image_logger(log_dir="image_store"):
    """Consumes images from the queue and logs them as JSON"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    while True:
        image_data_package = image_queue.get()
        if image_data_package is None:
            break
        try:
            image_data, metadata = image_data_package
            timestamp = int(time.time() * 1000)
            filename = f"image_{timestamp}.png"
            filepath = os.path.join(log_dir, filename)
            img = Image.frombytes('RGB', (128,128), image_data)
            img.save(filepath)

            log_record = {
                "timestamp": timestamp,
                "image_path": filepath,
                "metadata": metadata
            }

            logging.info(json.dumps(log_record))

        except Exception as e:
            logging.error(f"Error logging image: {e}")
        finally:
          image_queue.task_done()

logger_thread = threading.Thread(target=image_logger, daemon=True)
logger_thread.start()


def log_image_async_json(image_data, metadata):
    """Add image to the logging queue"""
    image_queue.put((image_data, metadata))


# Example usage
if __name__ == '__main__':
    # Example binary image data
    dummy_image_data = os.urandom(128 * 128 * 3)  # 128x128 RGB image

    metadata = {
        "source": "camera1",
        "location": "roomA"
    }

    log_image_async_json(dummy_image_data, metadata)
    log_image_async_json(os.urandom(128 * 128 * 3), {"source": "camera2", "location":"roomB"})

    image_queue.join()
    image_queue.put(None)
    logger_thread.join() # Wait for the logger thread to finish

    print("Async JSON logging complete.")
```

Now instead of pure text log we are having JSON data written to the log file. This makes it much better to ingest into data analysis tools. This provides more structure and the metadata helps add more information

In terms of resources consider reading papers and books related to data storage and I/O optimizations. Specifically see “Operating System Concepts” by Abraham Silberschatz Peter Baer Galvin Greg Gagne as well as some research papers relating to efficient image compression and storage. Also there are many resources online about the ELK stack for instance Elasticsearch: The Definitive Guide.

So that’s my take on logging images. It's a deep topic but following these patterns and understanding your own requirements should get you far. Be wary of performance storage and data management.
