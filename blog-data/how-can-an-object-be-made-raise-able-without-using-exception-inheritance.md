---
title: "How can an object be made raise-able without using exception inheritance?"
date: "2024-12-23"
id: "how-can-an-object-be-made-raise-able-without-using-exception-inheritance"
---

Alright, let's tackle this. I’ve seen this particular design challenge pop up in several projects, particularly those with a strong emphasis on predictable control flow and situations where exception handling might become a performance bottleneck. The question isn't necessarily about replacing exception handling *entirely*, but more about exploring alternative mechanisms for signaling and reacting to conditions that, in more conventional setups, might trigger exceptions. Instead of leaning solely on exception inheritance to model raise-able events, we can employ techniques that often result in more explicit and controlled handling of these conditions. I remember once, while working on a high-throughput data processing system, we needed finer control over the error handling paths—exceptions were simply too costly at scale for some scenarios.

The first point of deviation from typical exception usage is moving to a system of explicit return values combined with what I’ll call "outcome" types. Instead of *raising* an error, a method *returns* a value that indicates success or a specific failure condition. This value acts as a signal that the caller must explicitly handle. The key here is to avoid simple boolean flags, which can often be ambiguous and hard to extend. We use something more descriptive, often an enumerated type or a union type that can carry specific error information.

Let's see this with a snippet of code, using Python for demonstration purposes since it is relatively concise and easy to understand:

```python
from enum import Enum
from typing import Union, Tuple, Generic, TypeVar

class ProcessingStatus(Enum):
    SUCCESS = 0
    INVALID_INPUT = 1
    RESOURCE_UNAVAILABLE = 2
    INTERNAL_ERROR = 3

T = TypeVar('T')

class Result(Generic[T]):
    def __init__(self, status: ProcessingStatus, value: Union[T, None] = None, error_message: str = ""):
        self.status = status
        self.value = value
        self.error_message = error_message

    def is_success(self) -> bool:
        return self.status == ProcessingStatus.SUCCESS

    def unwrap(self) -> T:
        if not self.is_success():
            raise Exception(f"Cannot unwrap result with status: {self.status} and error: {self.error_message}")
        return self.value


def process_data(input_data: str) -> Result[int]:
    if not input_data:
        return Result(ProcessingStatus.INVALID_INPUT, error_message="Input data cannot be empty")
    try:
        processed_value = int(input_data) * 2
        return Result(ProcessingStatus.SUCCESS, processed_value)
    except ValueError:
        return Result(ProcessingStatus.INVALID_INPUT, error_message="Invalid input format: Input data must be an integer.")

# Example Usage
result = process_data("10")
if result.is_success():
   print(f"Processed data: {result.unwrap()}")
else:
  print(f"Processing failed: {result.status} with message {result.error_message}")

result = process_data("")
if result.is_success():
    print(f"Processed data: {result.unwrap()}")
else:
    print(f"Processing failed: {result.status} with message {result.error_message}")

result = process_data("abc")
if result.is_success():
    print(f"Processed data: {result.unwrap()}")
else:
    print(f"Processing failed: {result.status} with message {result.error_message}")
```

In this snippet, instead of raising `ValueError` or some other derived exception, `process_data` returns a `Result` object encapsulating a `ProcessingStatus` which can signal various failures, including `INVALID_INPUT` and `INTERNAL_ERROR`, alongside any error message.  Crucially, the caller *must* check the `status` field to determine if the processing was successful before accessing the computed value using `unwrap()`. This forces a conscious decision about how to proceed based on the outcome, which can prevent implicit ignoring of error conditions. Note, that `unwrap()` will raise an exception if the status was not `SUCCESS`, which is intentional. This is a case where an exception is raised, not within the primary processing, but in the consumption of results when an error condition occurred. This is a subtle but significant shift in control and responsibility.

Another approach is to use something akin to a functional "Either" type or "Result" type found in languages like Haskell or Rust. These types generally encode a success value (`Ok` or `Right`) or a failure value (`Err` or `Left`). This allows for elegant chaining of operations where any one operation can return a failure without interrupting the overall process flow. If we were doing this in Java (which I've used heavily), it might look something like this:

```java
import java.util.Objects;
import java.util.function.Function;

class Result<T, E> {
    private final T value;
    private final E error;
    private final boolean isSuccess;

    private Result(T value, E error, boolean isSuccess) {
        this.value = value;
        this.error = error;
        this.isSuccess = isSuccess;
    }

    public static <T, E> Result<T, E> success(T value) {
        return new Result<>(value, null, true);
    }

    public static <T, E> Result<T, E> failure(E error) {
        return new Result<>(null, error, false);
    }

    public boolean isSuccess() {
        return isSuccess;
    }

    public T unwrap() {
        if (!isSuccess) {
           throw new RuntimeException("Cannot unwrap a failed result: " + error.toString());
        }
        return value;
    }

    public E unwrapError() {
        if (isSuccess) {
            throw new RuntimeException("Cannot unwrap a successful result, no error to be found.");
        }
       return error;
    }


    public <U> Result<U, E> map(Function<T, U> mapper) {
        if (isSuccess) {
            return Result.success(mapper.apply(value));
        } else {
            return Result.failure(error);
        }
    }

    public <U> Result<U, E> flatMap(Function<T, Result<U, E>> mapper) {
       if (isSuccess){
           return mapper.apply(value);
       } else {
           return Result.failure(error);
       }
    }

}

enum ProcessingError {
    INVALID_INPUT,
    RESOURCE_ERROR,
    INTERNAL_ERROR
}

public class DataProcessor {
    public static Result<Integer, ProcessingError> processData(String input) {
        if (Objects.isNull(input) || input.isEmpty()) {
            return Result.failure(ProcessingError.INVALID_INPUT);
        }
        try {
            int parsedValue = Integer.parseInt(input);
            return Result.success(parsedValue * 2);
        } catch (NumberFormatException ex) {
            return Result.failure(ProcessingError.INVALID_INPUT);
        }
    }

    public static void main(String[] args) {
       Result<Integer, ProcessingError> result1 = processData("10");
       if (result1.isSuccess()) {
           System.out.println("Result: " + result1.unwrap());
       } else {
          System.out.println("Failed with error: " + result1.unwrapError());
       }

       Result<Integer, ProcessingError> result2 = processData(null);
       if (result2.isSuccess()) {
           System.out.println("Result: " + result2.unwrap());
       } else {
          System.out.println("Failed with error: " + result2.unwrapError());
       }

       Result<Integer, ProcessingError> result3 = processData("abc");
       if (result3.isSuccess()) {
           System.out.println("Result: " + result3.unwrap());
       } else {
         System.out.println("Failed with error: " + result3.unwrapError());
       }

       // Chaining operations
       Result<Integer, ProcessingError> chainedResult = processData("5")
               .map(value -> value + 3)
               .flatMap(value -> processData(String.valueOf(value)));
       if(chainedResult.isSuccess()){
           System.out.println("Chained result: " + chainedResult.unwrap());
       } else{
           System.out.println("Chained operation failed: " + chainedResult.unwrapError());
       }

    }
}
```

Here, we have a generic `Result` class that either holds a success value of type `T` or an error of type `E`. The `processData` method now returns a `Result` indicating success or failure and the reason using the `ProcessingError` enum. Methods like `map` and `flatMap` allow for chaining operations in a functional manner, propagating the error state and preventing computations based on potentially invalid results. Again, exceptions aren't used for core processing, but *are* used when a consumer attempts to interact with an error state.

Finally, another design pattern that can be used is something I’d call "stateful computation". In cases where multiple steps are involved, rather than immediately throwing an error, we can encapsulate the intermediate steps within an object that maintains its internal state, allowing operations to proceed or return an error based on this state. This is more verbose, but very explicit. Let me illustrate using a simplified example in C# with more practical implications for situations where a sequence of actions need to occur and all errors and their sources need to be tracked rather than relying on individual exceptions:

```csharp
using System;
using System.Collections.Generic;

public enum OperationStatus {
    Pending,
    Success,
    Failed
}

public class OperationState<T> {
    public OperationStatus Status { get; private set; }
    public T Result { get; private set; }
    public List<string> Errors { get; } = new List<string>();

    public OperationState() {
        Status = OperationStatus.Pending;
    }

    public void SetSuccess(T result) {
        Status = OperationStatus.Success;
        Result = result;
    }

    public void SetFailure(string error) {
        Status = OperationStatus.Failed;
        Errors.Add(error);
    }

    public void AddError(string error){
      Errors.Add(error);
    }
    public void SetPending()
    {
        Status = OperationStatus.Pending;
    }
    public OperationState<T> Then(Func<T, OperationState<T>> nextOperation)
    {
       if (Status != OperationStatus.Success)
       {
         return this;
       }
       return nextOperation(Result);
    }
    public void Reset(){
        Status = OperationStatus.Pending;
        Result = default(T);
        Errors.Clear();
    }

}

public class DataProcessor
{
    public OperationState<int> ValidateInput(string input)
    {
        var state = new OperationState<int>();
        if (string.IsNullOrEmpty(input))
        {
            state.SetFailure("Input cannot be null or empty.");
            return state;
        }

        if (!int.TryParse(input, out int value))
        {
          state.SetFailure("Input must be a valid integer.");
          return state;
        }

        state.SetSuccess(value);
        return state;
    }

    public OperationState<int> ProcessValue(int value)
    {
        var state = new OperationState<int>();
        try{
           int result = value * 2;
           state.SetSuccess(result);
        }
        catch(Exception ex){
           state.SetFailure("An error occurred during processing: " + ex.Message);
        }
        return state;
    }

    public static void Main(string[] args) {
        var processor = new DataProcessor();
        var result1 = processor.ValidateInput("10").Then(processor.ProcessValue);

        if (result1.Status == OperationStatus.Success)
        {
            Console.WriteLine("Result 1: " + result1.Result);
        }
        else
        {
            Console.WriteLine("Result 1 Failed: " + string.Join(", ", result1.Errors));
        }

        var result2 = processor.ValidateInput("abc").Then(processor.ProcessValue);
        if (result2.Status == OperationStatus.Success)
        {
            Console.WriteLine("Result 2: " + result2.Result);
        }
        else
        {
            Console.WriteLine("Result 2 Failed: " + string.Join(", ", result2.Errors));
        }


        var result3 = processor.ValidateInput(null).Then(processor.ProcessValue);
       if (result3.Status == OperationStatus.Success)
        {
            Console.WriteLine("Result 3: " + result3.Result);
        }
        else
        {
            Console.WriteLine("Result 3 Failed: " + string.Join(", ", result3.Errors));
        }
    }
}
```
Here, `OperationState<T>` tracks whether an operation succeeded, failed, or is pending. Each method returns `OperationState`, allowing subsequent operations to be chained via the `Then()` method. The result is either carried through or failures are propagated and can be extracted. Notice how multiple errors can also be collected if the design is modified to allow for that. This is very explicit and helps avoid surprises down the line.

In summary, instead of solely relying on exception inheritance to signal exceptional states, one can achieve a similar and in some cases better result by using explicit return values (using an `enum`, or something like a `Result` type) and carefully tracked state. This approach can increase code clarity and offers more granular control over error handling, especially in performance-critical code. If you're looking for more on these patterns, I would strongly recommend checking out books such as "Functional Programming in C#" by Enrico Buonanno (if you work in the .NET environment), or for more general concepts "Programming in Haskell" by Graham Hutton. Furthermore, research on monadic programming concepts and error handling techniques in functional languages will definitely expand your understanding of these alternatives.
