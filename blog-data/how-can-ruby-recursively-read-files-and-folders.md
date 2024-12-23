---
title: "How can Ruby recursively read files and folders?"
date: "2024-12-23"
id: "how-can-ruby-recursively-read-files-and-folders"
---

Let's tackle this directly. I've certainly had my fair share of needing to recursively traverse file systems in Ruby, and it's a task that, while conceptually simple, can get nuanced quickly depending on the specific requirements. The key is understanding the combination of Ruby's `Dir` and `File` classes with a properly structured recursive function. I'll break down my approach, and we'll look at some code examples, including common scenarios and edge cases.

My work on a large-scale document processing system a few years back forced me to become quite proficient with this. We had a hierarchical structure of documents stored in subfolders, and processing each one individually and their respective subfolder was a daily requirement. We had to handle millions of documents, so efficiency, and error handling was paramount. Let's begin.

At the core, the approach revolves around using `Dir.entries` to list the contents of a directory. This gives you an array of filenames and directory names within the specified directory. Then you loop through that array. When you find a file, you act on it. When you find a directory, you recursively call the same function, but with the newly discovered directory as the argument. Crucially, you need to pay close attention to how you construct file paths to correctly identify the files and folders regardless of how nested they are. This avoids many common pitfalls that cause the script to fail or simply skip files unexpectedly.

Let's start with a basic example. Suppose we want to print out the names of every file in a folder and its subfolders. Here's the code:

```ruby
def recursively_list_files(directory)
  Dir.entries(directory).each do |entry|
    next if entry == '.' || entry == '..' # Skip current and parent directories

    full_path = File.join(directory, entry)

    if File.file?(full_path)
      puts full_path
    elsif File.directory?(full_path)
      recursively_list_files(full_path) # Recursive call
    end
  end
end

# Example usage
root_dir = './example_folder' # Replace with your directory
recursively_list_files(root_dir)
```

In this first example, I've employed `Dir.entries` which allows us to get all the names inside a directory. The `next if entry == '.' || entry == '..'` part ensures we don't get stuck in a loop where we continually try to process the current or parent directory recursively, leading to a stack overflow. The `File.join` method constructs the complete path, which is key for navigating between directories reliably, avoiding confusion with relative paths. The checks `File.file?` and `File.directory?` are crucial to distinguish between files that should be directly handled, and folders that require deeper recursive dives.

Now, let's take this a step further. What if we want to perform a specific action on each file, such as extracting some information from a text file, or perhaps renaming all the files that have the extension ‘.txt’ to ‘.text’? This is where the core of file system manipulation usually resides. Instead of just printing the name, we can execute some arbitrary block of code when processing each file.

Here's a second example, showing how you can rename all .txt files:

```ruby
def recursively_process_files(directory, file_action)
  Dir.entries(directory).each do |entry|
    next if entry == '.' || entry == '..'

    full_path = File.join(directory, entry)

    if File.file?(full_path)
       file_action.call(full_path) # Execute the action with the file path
    elsif File.directory?(full_path)
      recursively_process_files(full_path, file_action) # Recursive call
    end
  end
end

# Example Usage
root_dir = './example_folder'

rename_action = Proc.new { |filepath|
    if filepath.end_with?('.txt')
       new_filepath = filepath.gsub('.txt', '.text')
        File.rename(filepath, new_filepath)
        puts "Renamed #{filepath} to #{new_filepath}"
     end
}

recursively_process_files(root_dir, rename_action)

```

Here we introduce the concept of a `Proc` object `rename_action`, which is a code block that will be executed when a file is found. We can pass any kind of action that takes a `filepath` as an argument. This gives a huge flexibility to deal with files during the directory recursion. The key advantage here is modularity; you can pass a different action for different requirements without changing the core recursion logic itself. I've used this exact technique to build custom file processing pipelines where each file had to go through multiple stages of processing before being moved to archive.

Finally, let's consider error handling. File systems are notoriously unreliable, and you often encounter permission issues, broken symlinks, or files that have been deleted in the middle of your scan. It's crucial to add proper error handling. If you’re operating at scale, logging these kinds of things becomes critical for debugging.

Here's a third example, adding error handling and logging:

```ruby
require 'logger'

def recursively_process_files_with_error_handling(directory, file_action, logger)
  Dir.entries(directory).each do |entry|
      next if entry == '.' || entry == '..'
        full_path = File.join(directory, entry)
    begin
      if File.file?(full_path)
          file_action.call(full_path)
      elsif File.directory?(full_path)
          recursively_process_files_with_error_handling(full_path, file_action, logger)
      end
    rescue Errno::EACCES => e
      logger.warn "Permission denied for: #{full_path} - #{e.message}"
    rescue Errno::ENOENT => e
      logger.warn "File or directory not found: #{full_path} - #{e.message}"
    rescue StandardError => e
      logger.error "An unexpected error occurred: #{full_path} - #{e.message}"
    end
  end
end

# Example Usage with error handling
root_dir = './example_folder'
logger = Logger.new('file_processing.log')
rename_action = Proc.new { |filepath|
  if filepath.end_with?('.txt')
    new_filepath = filepath.gsub('.txt', '.text')
    File.rename(filepath, new_filepath)
    logger.info "Renamed #{filepath} to #{new_filepath}"
  end
}


recursively_process_files_with_error_handling(root_dir, rename_action, logger)

```

In this modified example, I've wrapped the code that handles each entry in a `begin...rescue` block. This lets us catch `Errno::EACCES` (permission denied), `Errno::ENOENT` (file or directory not found), as well as any generic error `StandardError`. Each of these are logged to the provided logger, in a configurable way (info, warning, or error) for better analysis of the scan. This approach ensures that your script doesn't crash when encountering an issue, instead, it logs it, and proceeds with the traversal where possible. This approach is absolutely indispensable when dealing with file systems at any significant scale.

If you want to dive deeper into these topics, I’d recommend starting with the official Ruby documentation for `Dir` and `File` classes. Furthermore, the "Programming Ruby 1.9 & 2.0: The Pragmatic Programmers' Guide" by Dave Thomas is a great comprehensive resource on the Ruby language and standard library. Finally, to understand more about logging, "Effective Logging in Python" by Brian Neal is a helpful resource, although the examples are in Python, the core ideas and best practices apply universally and would be equally applicable in Ruby. The book focuses on handling errors gracefully, and I've often referred to its principles when creating robust tools. These resources should provide a firm understanding and allow you to further refine your approach to recursively dealing with files and folders in Ruby.
