---
title: "How can I reduce code duplication in this Ruby snippet?"
date: "2025-01-30"
id: "how-can-i-reduce-code-duplication-in-this"
---
I've encountered similar patterns of redundancy when processing diverse data sets from disparate APIs. The challenge often lies in handling common transformations or validations across varying data structures, leading to duplicated code blocks. Specifically, the issue you're describing can be addressed by employing techniques such as method extraction, inheritance, and mixins, depending on the specific context and desired maintainability goals. Let’s examine strategies for reducing the duplication with examples.

**1. Method Extraction**

The most direct approach, when the duplicated logic is contained within distinct methods or blocks, is method extraction. We isolate the redundant code into a new, reusable method and call that method wherever the original logic was required. This simplifies the caller methods, improves readability, and reduces maintenance overhead if changes are required to the shared logic.

Consider this simplified scenario where the code attempts to validate and normalize string values received from multiple sources, some possibly `nil` or empty:

```ruby
def process_source_a(data)
  value = data[:name]
  if value.nil? || value.empty?
    "unknown"
  else
    value.strip.downcase
  end
end

def process_source_b(data)
  value = data["title"]
  if value.nil? || value.empty?
    "unknown"
  else
    value.strip.downcase
  end
end

def process_source_c(data)
    value = data.fetch("label", nil)
    if value.nil? || value.empty?
      "unknown"
    else
      value.strip.downcase
    end
end
```

This snippet demonstrates redundant null/empty checks and string manipulation logic. To address this, I would extract the core validation and normalization steps into a private helper method:

```ruby
def process_source_a(data)
    normalize_string(data[:name])
end

def process_source_b(data)
    normalize_string(data["title"])
end

def process_source_c(data)
    normalize_string(data.fetch("label", nil))
end

private

def normalize_string(value)
    if value.nil? || value.empty?
      "unknown"
    else
      value.strip.downcase
    end
end
```

By creating the `normalize_string` method, I removed the duplication. Each `process_source_*` method becomes more focused on its data access requirements, delegating the transformation. This approach is most effective when the shared logic is relatively self-contained, and differences between calling methods are primarily in the access to the data itself.

**2. Inheritance and Template Methods**

For more complex scenarios where the duplication occurs within the structure of the methods themselves, with similar flows but with variations, inheritance might be a better alternative. A base class can implement the overarching control flow while leaving some steps to be implemented by subclasses. This concept is embodied by the template method pattern.

Suppose we're working with processing various user records, each potentially containing different validation needs. Assume each user type has a general processing pipeline: data loading, basic validation, specific type validation, and post-processing.

```ruby
class UserProcessor
  def process(user_data)
    load_data(user_data)
    basic_validation
    specific_validation
    post_processing
  end

  def load_data(user_data)
      raise NotImplementedError, "Subclasses must implement load_data"
  end

  def basic_validation
      puts "Basic Validation"
  end

  def specific_validation
      raise NotImplementedError, "Subclasses must implement specific_validation"
  end

  def post_processing
       puts "Post Processing"
  end
end

class AdminUserProcessor < UserProcessor
    def load_data(user_data)
       puts "Loading Admin Data"
      @user_data = user_data[:admin]
    end

    def specific_validation
        puts "Admin Specific Validation"
    end
end

class StandardUserProcessor < UserProcessor
    def load_data(user_data)
        puts "Loading Standard User Data"
        @user_data = user_data[:standard]
    end

    def specific_validation
        puts "Standard User Specific Validation"
    end
end
```
In this example, the `UserProcessor` class defines the base processing flow in the `process` method.  Subclasses `AdminUserProcessor` and `StandardUserProcessor` inherit this flow and provide their own implementation for `load_data` and `specific_validation`. The common `basic_validation` and `post_processing` remain in the base class. This method is effective in capturing the overall structure of operations when classes have significant overlaps in behavior, but require specific customizations. The use of `NotImplementedError` ensures that subclasses are forced to provide concrete implementations.

**3. Mixins**

When shared logic isn't necessarily hierarchical but rather represents reusable functionalities across various classes, mixins offer a more flexible solution. These modules containing methods can be included into multiple classes, avoiding the tight coupling that comes with inheritance.

Consider a scenario where we want to add logging and error handling capabilities to multiple classes responsible for data access.

```ruby
module Loggable
  def log_message(message, level = :info)
    puts "[#{level.upcase}] #{Time.now} - #{message}"
  end
end

module ErrorHandler
  def handle_error(exception, message)
    log_message("ERROR: #{message} - #{exception.message}", :error)
    # Optionally re-raise or continue execution
  end
end


class DataFetcher
  include Loggable
  include ErrorHandler

  def fetch_data(url)
    log_message("Fetching data from #{url}")
    begin
      # Simulated data fetch
      sleep(1)
      raise "Failed to fetch data"
    rescue => e
      handle_error(e, "Data fetch failed")
      return nil
    end
      log_message("Data fetch completed")
      return {data: "Fetched from #{url}"}
  end
end


class DataProcessor
  include Loggable
  include ErrorHandler

   def process_data(data)
    log_message("Processing data")
     begin
        #Simulated processing
        sleep(1)
        raise "Error processing data"
      rescue => e
       handle_error(e, "Data Processing error")
        return nil
      end
     log_message("Data processed")
     return {processed: "Processed data"}
   end
end

data_fetcher = DataFetcher.new
data_processor = DataProcessor.new

data = data_fetcher.fetch_data("https://api.example.com/data")
data_processor.process_data(data) unless data.nil?
```

Here, the `Loggable` and `ErrorHandler` modules provide reusable methods. The `DataFetcher` and `DataProcessor` classes include these modules to obtain logging and error handling capabilities, without being bound by an inheritance structure. This approach fosters code reuse across disparate classes. Mixins are especially helpful when the shared functionality doesn’t represent a conceptual parent-child relationship, but rather represents cross-cutting concerns that enhance multiple classes.

In summary, reducing code duplication often involves careful assessment of the relationships between the duplicated blocks. Method extraction addresses localized redundancy. Inheritance and the template method pattern are effective where overall structures of methods can be abstracted into base classes. Finally, mixins offer flexibility when common functionalities need to be incorporated into multiple classes that don’t necessarily belong to the same hierarchy. The selection of the most suitable strategy is often a matter of balancing maintainability and flexibility and will vary case-by-case.

**Resource Recommendations**

For further understanding of these topics, I would recommend researching the following concepts and patterns: *Refactoring*, focusing on techniques for code improvement, *the Template Method pattern* for structural code reuse, and *Mixins* for sharing functionalities across classes. A solid foundation in object-oriented design principles, such as the Single Responsibility Principle, is also beneficial in identifying opportunities for code simplification and reuse.
