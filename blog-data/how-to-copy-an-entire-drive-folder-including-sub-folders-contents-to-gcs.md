---
title: "How to copy an entire drive folder (including sub folders) contents to gcs?"
date: "2024-12-14"
id: "how-to-copy-an-entire-drive-folder-including-sub-folders-contents-to-gcs"
---

alright, so you want to copy a local directory, with all its subdirectories and files, up to google cloud storage (gcs)? been there, done that, got the t-shirt – or rather, spent way too many late nights debugging bash scripts and python code. it's a common task, but the devil's in the details. let's walk through a few options, and i'll share some gotchas i've encountered along the way.

first off, the most straightforward method, and probably what most people reach for initially, is the `gsutil` command-line tool. it's part of the google cloud sdk and it's really powerful for this sort of thing. if you haven't got it installed you should, it will save you a lot of headaches down the line. it handles a lot of the complexities for you, like creating the necessary directories in gcs and handling large files efficiently. a simple way of doing that will be with a command like this:

```bash
gsutil -m cp -r /path/to/your/local/folder gs://your-gcs-bucket/destination/path/
```

let's break it down: `gsutil cp` is the core command for copying data to or from gcs. the `-m` flag activates multi-threading, which can significantly speed things up, especially with larger directories. the `-r` flag tells `gsutil` to copy directories recursively, meaning it'll grab everything within the source folder, including subfolders. `/path/to/your/local/folder` is, well, the path to your directory you want to upload. finally, `gs://your-gcs-bucket/destination/path/` is where you want the data to land in gcs, specifying your bucket and an optional destination path. make sure the bucket exists before you run this command – gsutil won't create it for you. you should also have the correct permissions to write into that bucket; double check if you're using a service account or your user account.

i remember one time when i was dealing with a particularly large directory, several gigabytes, and didn’t use the `-m` option it took forever, like watching paint dry. the multithreading feature is a real lifesaver. also, i once messed up the destination path, and ended up with everything at the bucket root – not exactly ideal. always double-check.

now, `gsutil` is great for a quick-and-dirty solution, or for one-off uploads, but what if you need more control, like progress updates or want to automate this whole thing programmatically? that's where python libraries come in. the `google-cloud-storage` library provides apis for interacting with gcs from python. here’s how you might copy a directory using that:

```python
from google.cloud import storage
import os

def upload_directory(local_path, bucket_name, gcs_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(gcs_path, relative_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
            print(f"uploaded: {local_file} to {blob_path}")

if __name__ == '__main__':
    local_directory = '/path/to/your/local/folder'
    bucket_name = 'your-gcs-bucket'
    gcs_destination = 'destination/path'
    upload_directory(local_directory, bucket_name, gcs_destination)
```

let's go through that code. first it imports the google cloud storage library and the os module to work with paths. then there's a function `upload_directory` that takes the source local path, bucket name, and gcs destination path as arguments. it creates a storage client, gets the bucket reference. then it uses `os.walk` to traverse the local directory tree, file by file. for each file, it calculates its relative path to keep the folder structure in gcs. then it constructs the gcs path, creates a blob object, and uploads the file. this example also prints the upload status, which is helpful for tracking. don't forget that before executing this code, you'll need to have authenticated your google cloud account, generally either using environment variables or application credentials. i recall banging my head against the wall with authentication issues for a few hours, a very silly mistake.

when using the python approach, keep an eye on file sizes, especially if you are running it on a small system with low memory available, if you have large files you should use a streamed upload instead of uploading them whole as this example shows.

now, a slightly more advanced scenario could be the case where you need to handle large files and ensure data integrity. the above code will work but it's more fragile to interruptions or data loss because each file is sent as whole. using a stream based copy is a much better approach here since files will be sent in smaller chunks. in this case i'd like to show you how to implement a chunked based upload using the same library:

```python
from google.cloud import storage
import os

def upload_directory_chunked(local_path, bucket_name, gcs_path, chunk_size=1024 * 1024):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(gcs_path, relative_path)
            blob = bucket.blob(blob_path)

            with open(local_file, 'rb') as file_obj:
                blob.upload_from_file(file_obj, size=os.path.getsize(local_file),
                                      chunk_size=chunk_size, num_retries=3)
            print(f"uploaded: {local_file} to {blob_path}")


if __name__ == '__main__':
    local_directory = '/path/to/your/local/folder'
    bucket_name = 'your-gcs-bucket'
    gcs_destination = 'destination/path'
    upload_directory_chunked(local_directory, bucket_name, gcs_destination)
```

the difference is on the `upload_from_file` method where we are opening the file in read binary mode and passing the file descriptor to the gcs api using `size` and `chunk_size`, if an interruption occurs during the process, it will be retried up to `num_retries`. it's slower but way safer and more robust. think of it as a more cautious way of sending data. a real slow dance.

now, before i finish, i'd like to address something important about this kind of task: it will generally be slow, specially when you have a lot of small files. gcs has a latency component and network transfer limits. a solution that helps with this issue could be, if possible, grouping your files into zip or tar archives to reduce the number of individual upload operations. also, if possible make sure you are closer to gcs servers, reducing the latency. i once tried uploading some code from my grandma's house and i wasn't even able to ping google.

for further study, i would suggest checking out the official google cloud storage documentation, it's very good and up-to-date. also, exploring concepts of distributed computing and cloud storage architectures can give you an even better comprehension of the limitations of the technology. in particular, i'd suggest reading papers on eventual consistency and consistent hashing. these topics will give you a good understanding of the trade-offs and design choices behind systems such as gcs. some good examples could be "dynamo: amazon's highly available key-value storage system" and "consistent hashing". there is also a very good book that i keep always next to me is "designing data-intensive applications" by martin kleppmann.

i think that’s all the knowledge i can throw at this problem for now. remember to tailor these examples to your specific situation. you may find that you need more complex error handling or specific features. in the end, these things come down to practice and experience. hope it helps!
