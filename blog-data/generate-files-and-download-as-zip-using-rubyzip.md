---
title: "generate files and download as zip using rubyzip?"
date: "2024-12-13"
id: "generate-files-and-download-as-zip-using-rubyzip"
---

Okay so you want to generate files dynamically and then zip them up and offer that as a download to the user right I've been there trust me it's a common task especially when you're building web applications that need to deliver a bunch of generated stuff at once let me tell you about the dark days before I really figured this out oh the things I tried

Basically the core issue here is you have some data you want to represent as files maybe text maybe JSON maybe images doesnt matter much for the zip part You could write these to disk first sure then bundle them up but that's slow and clutters your filesystem especially if it's all temporary stuff that the user needs to download once and then its gone So a more efficient approach is to create these files in memory and then add them to a zip archive that you then stream to the user

I've wrestled with various libraries in the past initially I tried just raw shell commands like `zip` but you can guess how clunky that got when needing to handle names characters and such then I dabbled with a few ruby gems that were clearly abandoned and had really bad documentation finally I landed on rubyzip and honestly it's solid it does the job and it's actively maintained thankfully

So first things first you'll need the `rubyzip` gem if you dont have it already `gem install rubyzip` should sort that out Now about generating the files well that part is up to you it completely depends on the kind of data you are working with so I wont go into details about data processing but here's the basic flow I've found useful in my projects

```ruby
require 'zip'

def generate_zip_file(files_data, output_filename)
  Zip::OutputStream.open(output_filename) do |zos|
    files_data.each do |filename, content|
      zos.put_next_entry(filename)
      zos.write(content)
    end
  end
end

# Example usage (replace with your actual data generation)
data_to_zip = {
  "file1.txt" => "This is the content of file 1",
  "file2.json" => '{"key": "value"}',
  "file3.csv" => "col1,col2\nval1,val2"
}

output_zip_filename = "my_download.zip"

generate_zip_file(data_to_zip, output_zip_filename)

puts "Zip file generated: #{output_zip_filename}"

# At this point you will use the zip file and pass to the browser as download
```

This code creates a zip file in the current directory called my\_download.zip and contains the files with the contents specified in the data\_to\_zip hash. However, this isn't very useful if you're trying to serve this to a user through a web framework like Rails or Sinatra because it writes it directly to the filesystem which is often not what you want.

So if you're running on something like a Rails application this is how it looks like, a better approach is to stream the zip data directly to the response that way you dont need to create a temporary file on disk here's an example of how to do that in a Rails controller method and also it should apply on other similar frameworks

```ruby
# In your Rails controller

def download_zip
  files_data = {
    "generated_data.txt" => "Dynamic data created on the fly!",
    "another_file.json" => '{"timestamp": "' + Time.now.to_s + '"}',
    "example.csv" => "row1,row2,row3\ndata1,data2,data3"
  }

  temp_file = Tempfile.new(['download_zip', '.zip'])

  begin
    Zip::OutputStream.open(temp_file.path) do |zos|
      files_data.each do |filename, content|
        zos.put_next_entry(filename)
        zos.write(content)
      end
    end

    send_file temp_file.path,
              filename: 'my_download.zip',
              type: 'application/zip',
              disposition: 'attachment'
  ensure
    temp_file.close
    temp_file.unlink
  end
end
```

Now this sends a download response to the browser using `send_file` it creates the temporary file and deletes it using the `ensure` to clean up any resources allocated. It does use a temporary file but it's cleaned up once its used by the `send_file` method. If you need even more control over streaming you can use `StringIO` to create a buffer that you can then stream to the user but that is a bit more complex so I'll leave it at this for now.

Also if you have huge files I recommend using the `Zip::File` class and its `add` methods to add already existing files to a zip.

Now to elaborate on the file creation part, that's not within the scope of the zip operation but I can give you an example if your data is json I've been doing that for years because all our data is json here is how it usually looks like

```ruby
require 'json'

def create_json_file_data(data)
    JSON.generate(data)
end


# Example usage
example_data = {
    "user": {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com"
    },
    "preferences": {
        "theme": "dark",
        "notifications": true
    }
}

json_content = create_json_file_data(example_data)
puts json_content

# You can add this to the data_to_zip hash in previous examples

```

So that covers the basic parts: you generate your files however you want and then you use `rubyzip` to pack everything into a zip and you can send it as a download. Keep in mind that with bigger files and higher load you might need to think of optimization that involve buffering or using streams directly with response bodies so make sure you check how it is done in your web framework.

A note for very large files: using `Tempfile` can also create a bottleneck if there's a lot of concurrent downloads happening you should look for alternatives like creating temporary file on memory (RAM) this can be a bit tricky and dependent on your environment but it is an option for extreme cases. Also if you are generating the content and not just streaming files from the file system, and the content is really large make sure to generate the content in chunks instead of all at once to avoid running out of memory. The `rubyzip` also can handle different compression levels, use higher levels if file size is important or lower levels for speed.

Another point I want to mention is that you should also remember about security you don't want to get files that can be exploited using relative path names `../some_file.txt` so it's important to make sure that every name is sanitized to avoid zip slip vulnerabilities.

I know this was a lot but its just how it is when you want to generate and send ZIP files its important to do it properly if you need more info about zip format read the PKWare zip specification is all there and also RFC 1951 RFC 1952 for the underlying zlib deflate compression algorithm that rubyzip and many other libraries use as well.

So that is it. Now you know how to handle zip files like a pro I know this process from experience believe me I was there not long ago fighting with random gems and now I do everything with rubyzip so feel good about yourself that you've just learned this very useful and common task.
