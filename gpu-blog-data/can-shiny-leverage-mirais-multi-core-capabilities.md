---
title: "Can Shiny leverage Mirai's multi-core capabilities?"
date: "2025-01-30"
id: "can-shiny-leverage-mirais-multi-core-capabilities"
---
The default R environment, upon which Shiny is built, executes single-threaded processes. This fundamentally limits its ability to fully utilize multi-core processors without additional measures. My experience developing complex Shiny applications for financial analysis has consistently highlighted this bottleneck, especially when handling large datasets or performing resource-intensive computations. However, the advent of packages like `future` and its parallel processing backend, `mirai`, offers a powerful solution to circumvent this inherent limitation, enabling parallel execution within Shiny apps.

Fundamentally, Shiny operates within a single R session. This means that any R code executed within the Shiny server component will, by default, run sequentially. If your application involves computationally heavy functions, long-running queries, or complex data manipulations, the user interface becomes unresponsive as the main thread is blocked. This leads to a poor user experience and severely restricts the scalability of Shiny applications on modern multi-core hardware. The challenge, therefore, lies in offloading these computations to separate R processes running in parallel, effectively unlocking multi-core performance. This is where `mirai`, working in conjunction with the `future` package, proves indispensable.

The `future` package provides an abstraction layer that allows you to specify how code should be evaluated—sequentially, synchronously, asynchronously, or in parallel. `mirai` acts as a specific backend for `future`, enabling parallel processing by leveraging the `mirai` library, which spawns multiple independent R processes. When you tell `future` to evaluate an expression using the `mirai` backend, it is executed within one of these child R processes, allowing the main Shiny R process to remain responsive and execute other tasks. This approach is not without its complexities, as data must be explicitly passed to and from these child processes, and the results then need to be incorporated back into the Shiny application. However, the performance gains are often substantial.

Consider a simplified example. Assume you have a computationally intensive function that takes a few seconds to complete:

```R
# Slow function example
slow_calculation <- function(x) {
  Sys.sleep(2)  # Simulates a long calculation
  return(x * 2)
}
```

If this function is executed directly within the Shiny server context in response to user input, the UI will freeze for two seconds while the calculation completes. Now, let's demonstrate how `mirai` can be used to offload this calculation:

```R
# Example 1: mirai with a simple function call
library(shiny)
library(future)
library(mirai)
plan(mirai) # Configure future to use mirai backend

ui <- fluidPage(
  actionButton("calculate", "Calculate"),
  textOutput("result")
)

server <- function(input, output, session) {
  observeEvent(input$calculate, {
    output$result <- renderText({
        future({ slow_calculation(5) }) %...>% as.character()
    })
  })
}

shinyApp(ui, server)
```

In this code snippet, `plan(mirai)` sets the processing strategy to use `mirai`. The call to `future()` wraps the computationally intensive function `slow_calculation`, which is then executed in a separate R process. The `%...>%` operator waits for the result from the future before converting it to a string via `as.character()` for output to the Shiny application. The UI remains responsive because the slow function does not block the main thread.

For more complex scenarios, especially when dealing with reactive values, `future` allows for asynchronous updates to the Shiny application. Imagine your Shiny app fetches large amounts of data from a database in response to a user selection, or a computationally-heavy filtering or grouping operation is needed. We can parallelize those operations to return updates to the UI with minimal delays:

```R
# Example 2: mirai with reactive values and promises
library(shiny)
library(future)
library(mirai)
plan(mirai)

ui <- fluidPage(
    selectInput("dataset", "Choose a dataset:", choices = c("iris", "mtcars")),
    textOutput("data_summary")
)

server <- function(input, output, session) {

    data_promise <- reactive({
        future({
            Sys.sleep(1) # simulate data fetching/processing
            data_set_name <- input$dataset
            get(data_set_name)
        })
    })

    output$data_summary <- renderText({
      data_promise() %...>% {
          paste("Data shape:", paste(dim(.), collapse = "x"),
                ", Number of Rows:", nrow(.))
      }
    })
}

shinyApp(ui, server)
```

In this second example, the selected dataset is loaded in a separate R process via `future({ ... })` upon a user change in the select box. The `reactive` function stores a `promise` from the future. The `renderText` function uses `data_promise() %...>% { ... }` to wait for the promise resolution before generating the summary output. If the dataset is computationally intensive to load or process, the `mirai` backend ensures it happens off the main UI thread.

Finally, consider a situation where a Shiny application needs to update a reactive output based on results from computations with potentially varying execution times. The use of `future` and `mirai` enables a non-blocking update mechanism by returning the results as soon as they are available:

```R
# Example 3: asynchronous updates with multiple future calls
library(shiny)
library(future)
library(mirai)
plan(mirai)

ui <- fluidPage(
  actionButton("start_tasks", "Start Tasks"),
  textOutput("task1_result"),
  textOutput("task2_result"),
  textOutput("task3_result")
)

server <- function(input, output, session) {
    observeEvent(input$start_tasks, {
        future({Sys.sleep(0.5); "Task 1 complete"}) %...>% {output$task1_result <- renderText(.)}
        future({Sys.sleep(1.0); "Task 2 complete"}) %...>% {output$task2_result <- renderText(.)}
        future({Sys.sleep(1.5); "Task 3 complete"}) %...>% {output$task3_result <- renderText(.)}
    })
}
shinyApp(ui, server)
```

In this example, three independent tasks with different simulated execution times are submitted to the `mirai` backend. The UI updates asynchronously as each task finishes, instead of waiting for all of them to complete.  The `future(...) %...>% {output$task_result <- renderText(.)}` pattern efficiently returns each task result to its respective output text component. This approach highlights the responsiveness gains by not delaying any UI update based on others' execution.

For those interested in delving deeper into parallel processing in R, I recommend exploring the documentation for the `future` package, specifically its `multisession` plan and other backends, as well as focusing on the nuances of managing data flow between the main process and child processes.  The book "Parallel Computing for R" offers comprehensive guidance on parallel programming concepts and related strategies. Finally, examining best practices for managing concurrency in R, as detailed in advanced R programming materials, helps refine the application's robustness and scalability. These resources offer the theoretical background and practical advice required to fully leverage the capabilities offered by `mirai` and `future` in Shiny applications. Properly implemented, `mirai` allows R-based Shiny applications to compete with more scalable platforms for computationally intensive tasks.
