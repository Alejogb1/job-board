---
title: "How to call asynchronous functions in ASP.NET?"
date: "2025-01-30"
id: "how-to-call-asynchronous-functions-in-aspnet"
---
Directly integrating asynchronous operations is paramount for responsiveness in modern ASP.NET applications, especially when dealing with I/O-bound tasks like database access or external API calls. Blocking a thread while waiting for such operations to complete can severely impact scalability and performance; hence, leveraging `async` and `await` keywords is crucial. This response elucidates how to call asynchronous functions within the ASP.NET context, supported by practical code examples.

The core concept revolves around marking methods as `async`, allowing them to pause execution without blocking the underlying thread while awaiting the completion of an asynchronous task. The `await` keyword then resumes execution once the task yields a result, thus freeing up the thread for other requests. This mechanism significantly enhances the application's ability to handle concurrent requests, providing a smoother user experience. I’ve seen firsthand how transitioning from synchronous to asynchronous operations on legacy systems can dramatically reduce response times and resource consumption, particularly under heavy load.

Here's a breakdown of how to implement this, focusing on the controller context, which is a frequent location for such operations:

**1. Asynchronous Action Methods in Controllers:**

In ASP.NET MVC or Web API controllers, actions can be defined as asynchronous by returning a `Task` or `Task<T>` type. This is fundamental for enabling non-blocking behavior. The controller method is marked with the `async` keyword and, subsequently, asynchronous operations are invoked using `await`. The return type determines whether a value is returned (`Task<T>`) or not (`Task`).

**Example 1: Data Retrieval**

```csharp
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using System.Net.Http;
using Newtonsoft.Json;

namespace AsyncDemo.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class DataController : ControllerBase
    {
        private readonly IHttpClientFactory _clientFactory;

        public DataController(IHttpClientFactory clientFactory)
        {
            _clientFactory = clientFactory;
        }

        [HttpGet("external")]
        public async Task<IActionResult> GetExternalData()
        {
            var client = _clientFactory.CreateClient();
            var response = await client.GetAsync("https://api.example.com/data");

            if (!response.IsSuccessStatusCode)
            {
                return BadRequest("Failed to retrieve external data.");
            }

            var content = await response.Content.ReadAsStringAsync();
            var data = JsonConvert.DeserializeObject(content);

            return Ok(data);
        }
    }
}
```

*   **`async Task<IActionResult> GetExternalData()`:** This declares the method as asynchronous and specifies that it will return an `IActionResult`, which allows for various HTTP response types.
*   **`var client = _clientFactory.CreateClient();`:** An `HttpClient` instance is created, often obtained via dependency injection. The `IHttpClientFactory` promotes the efficient reuse of `HttpClient` instances, avoiding resource exhaustion.
*   **`await client.GetAsync("https://api.example.com/data");`:** This is the asynchronous call. The `await` keyword pauses method execution while waiting for the HTTP request to complete, without blocking the thread. Once the response arrives, execution resumes at this point.
*   **`await response.Content.ReadAsStringAsync();`:** Another asynchronous call awaits reading the response content.
*   **`return Ok(data);`:** The retrieved data is returned as an HTTP 200 (OK) response.

**2. Calling Asynchronous Methods from Other Asynchronous Methods:**

Asynchronous methods can be composed by calling other asynchronous methods using `await`. This promotes clean and sequential execution of asynchronous tasks. When creating these composite workflows, it's critical to consistently use `async` and `await` to propagate the asynchronous nature throughout the call stack. This avoids blocking and maintains the responsiveness of the application. I have found that inconsistencies in asynchronous workflow can easily lead to deadlock issues, which can be a headache to debug.

**Example 2: Multiple Data Sources**

```csharp
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using System.Net.Http;
using Newtonsoft.Json;

namespace AsyncDemo.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class CombinedDataController : ControllerBase
    {
        private readonly IHttpClientFactory _clientFactory;

        public CombinedDataController(IHttpClientFactory clientFactory)
        {
            _clientFactory = clientFactory;
        }

        [HttpGet("combined")]
        public async Task<IActionResult> GetCombinedData()
        {
            var data1Task = GetExternalData("https://api1.example.com/data");
            var data2Task = GetExternalData("https://api2.example.com/data");

            //Wait for both tasks to complete concurrently
            await Task.WhenAll(data1Task, data2Task);
            
            if(data1Task.IsCompletedSuccessfully && data2Task.IsCompletedSuccessfully)
            {
              var data1 = data1Task.Result;
              var data2 = data2Task.Result;
              return Ok(new {data1, data2});
            }

            return BadRequest("Failed to retrieve data from one or more sources.");

        }

        private async Task<object> GetExternalData(string url)
        {
            var client = _clientFactory.CreateClient();
            var response = await client.GetAsync(url);

            if (!response.IsSuccessStatusCode)
            {
                return null; //Or throw exception if required.
            }

            var content = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject(content);

        }
    }
}
```

*   **`Task<object> GetExternalData(string url)`:** This method encapsulates the asynchronous data retrieval logic.
*   **`var data1Task = GetExternalData("https://api1.example.com/data");` and `var data2Task = GetExternalData("https://api2.example.com/data");`:** The `GetExternalData` method is called twice asynchronously, returning `Task` instances without blocking.
*   **`await Task.WhenAll(data1Task, data2Task);`:** The `Task.WhenAll` method allows the two tasks to run in parallel and the method will only proceed when both are finished. This prevents the application from unnecessarily waiting sequentially.
*   **`var data1 = data1Task.Result;` and `var data2 = data2Task.Result;`:** The data is retrieved from the `Task.Result` property after completion.
*   **`return Ok(new {data1, data2});`:** A combined object containing data from both requests is returned.

**3. Asynchronous Operations within Services:**

It's good practice to separate business logic from the controller, often employing services for this purpose. Asynchronous service methods can then be called from asynchronous controller actions. This maintains separation of concerns and improves code maintainability. As I’ve witnessed, adhering to well-defined layers greatly enhances a project’s scalability and reduces the cognitive load required for updates or bug fixes.

**Example 3: Data Processing and Persistence**

```csharp
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using AsyncDemo.Services;
using AsyncDemo.Models;

namespace AsyncDemo.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ProcessDataController : ControllerBase
    {
        private readonly IDataService _dataService;

        public ProcessDataController(IDataService dataService)
        {
            _dataService = dataService;
        }

        [HttpPost("process")]
        public async Task<IActionResult> ProcessIncomingData([FromBody] DataModel input)
        {
            if(input == null || string.IsNullOrEmpty(input.SomeValue)) {
                return BadRequest("Input cannot be null or empty");
            }

            var processedData = await _dataService.ProcessDataAsync(input);

            if(processedData != null) {
                return Ok(processedData);
            }

            return BadRequest("Data processing failed.");

        }
    }
}


//Service Layer
namespace AsyncDemo.Services
{
    public interface IDataService
    {
        Task<DataModel> ProcessDataAsync(DataModel input);
    }

    public class DataService : IDataService
    {
       private readonly IDataStore _dataStore;
        public DataService(IDataStore dataStore){
            _dataStore = dataStore;
        }
        public async Task<DataModel> ProcessDataAsync(DataModel input)
        {
            //Simulated processing delay
            await Task.Delay(500);

             var result = await _dataStore.AddDataAsync(input);
            if (result)
            {
              return input;
            }
            return null;
        }
    }

    public interface IDataStore
    {
        Task<bool> AddDataAsync(DataModel input);
    }

    public class DataStore : IDataStore
    {
         //Simulated data persistence, typically would use a DB context here
        public async Task<bool> AddDataAsync(DataModel input)
        {
            await Task.Delay(250); //Simulate DB write
            return true;
        }
    }

}

namespace AsyncDemo.Models {
    public class DataModel {
        public string SomeValue { get; set; }
    }
}
```

*   **`[HttpPost("process")]`:** Controller action accepts input data via POST request.
*   **`IDataService _dataService`:** The data service is injected via dependency injection.
*   **`await _dataService.ProcessDataAsync(input);`:** The asynchronous method of the service layer is called using await.
*  **`await Task.Delay(500);`:** This simulates some processing by pausing asynchronously.
*  **`await _dataStore.AddDataAsync(input);`:** The data is asynchronously written via the data store layer.
*  **`return input;`:** The input data is returned as a result if the persistence is successful.

**Resource Recommendations:**

To deepen understanding of asynchronous programming in ASP.NET, explore the official Microsoft documentation on `async` and `await`, along with the documentation covering the `Task` and `Task<T>` classes. Investigate best practices for dependency injection and the use of `IHttpClientFactory`. Additionally, examining articles that delve into performance implications and common pitfalls of async programming within ASP.NET can provide invaluable insights. Consulting blog posts by experienced .NET developers can further solidify your knowledge, offering practical approaches and examples that complement the theoretical information provided by the documentation. These resources will provide a strong foundation for effectively leveraging asynchronous operations in your projects.
