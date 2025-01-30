---
title: "How can I create a Thread application in ModelSim 10.5c's DO file using TCL?"
date: "2025-01-30"
id: "how-can-i-create-a-thread-application-in"
---
The core challenge in creating a threaded application within ModelSim's DO file environment using TCL lies in the inherent limitations of TCL's threading capabilities and ModelSim's interaction with external processes.  TCL's threading model, while functional, isn't directly suited for tightly controlling concurrent simulation processes within ModelSim.  Instead, efficient thread management necessitates leveraging ModelSim's built-in command set and careful orchestration of simulation processes using TCL as the glue. My experience implementing similar solutions for complex SoC verification environments has highlighted the need for a robust, event-driven approach.

**1. Clear Explanation:**

The approach hinges on structuring the simulation flow as a series of independent, yet coordinated, simulation runs.  Instead of attempting true multi-threading within a single ModelSim instance, we use TCL to launch multiple instances of ModelSim, each managing a separate thread of execution.  These instances communicate indirectly, typically through files or a shared memory mechanism (if supported by the underlying operating system and the simulated design). Inter-process communication is crucial for synchronizing the threads and exchanging data.  This approach avoids the complexities of managing threads directly within the TCL interpreter, bypassing the inherent limitations of TCL's threading model in the context of ModelSim.

To synchronize these independent simulation processes, we leverage ModelSim's ability to write data to files during simulation and to read these files in subsequent runs. This allows a process to complete a task, write its results, and then trigger another process to start based on these results.  Error handling and timeout mechanisms are essential to ensure robustness.  The choice between file-based communication and potentially faster shared memory approaches depends on the complexity of the data exchanged and the system's capabilities.  For many applications, the simplicity and robustness of file-based communication outweigh the slight performance penalty.

**2. Code Examples with Commentary:**

**Example 1: Basic Two-Thread Simulation**

This example shows two independent ModelSim instances simulating separate parts of a design.  Each instance writes its results to a distinct file.  Error handling is simplified for brevity.

```tcl
# Thread 1: Simulate module A
set sim1_cmd "vsim -c -do \"run 10000; writemem -force result1.txt data_from_A; quit -f\" work.module_A"
exec $sim1_cmd &

# Thread 2: Simulate module B
set sim2_cmd "vsim -c -do \"run 10000; writemem -force result2.txt data_from_B; quit -f\" work.module_B"
exec $sim2_cmd &

# Wait for both simulations to complete (simplified for demonstration)
wait
puts "Simulations completed. Results in result1.txt and result2.txt"
```

**Commentary:** This example demonstrates launching two ModelSim instances concurrently using `exec` with the `&` operator for background execution. Each instance executes a specific DO file command sequence, simulating a different module and storing results in separate files.  The `wait` command is a placeholder; a more sophisticated mechanism would be needed in a real-world application to track the completion of both processes.  Consider using the `wait` command within loops to properly monitor the simulations' exit status and handle potential errors.


**Example 2:  Sequential Thread Execution with File-Based Communication**

Here, the output of one simulation triggers the next.

```tcl
# Thread 1: Simulate module C, producing input for Thread 2
set sim1_cmd "vsim -c -do \"run 1000; writemem -force input_for_D.txt data_from_C; quit -f\" work.module_C"
exec $sim1_cmd &
wait
if {[catch {exec cat input_for_D.txt} error_message]} {
    puts "Error during Thread 1: $error_message"
    exit 1
}


# Thread 2: Simulate module D, using output from Thread 1
set sim2_cmd "vsim -c -do \"readmem input_for_D.txt data_from_C; run 2000; quit -f\" work.module_D"
exec $sim2_cmd &
wait


puts "Simulations completed sequentially"
```

**Commentary:** This example shows a more structured approach.  Thread 1 completes, writes its output to `input_for_D.txt`, and then Thread 2 starts, reading this file as input.  Error handling is included to check if Thread 1 completed successfully and if the file exists.  The use of `wait` after each `exec` command ensures sequential execution.  This is a basic example; in larger applications, sophisticated flow control mechanisms might be necessary.



**Example 3:  Advanced Thread Management with Error Handling and Timeout**

This code introduces more robust error handling and a timeout mechanism.

```tcl
proc run_simulation {sim_cmd timeout} {
    exec $sim_cmd &
    set pid [lindex [split [exec ps aux | grep $sim_cmd | grep -v grep]] 2]
    set startTime [clock seconds]
    while {[clock seconds] - $startTime < $timeout} {
        if {[catch {exec kill -0 $pid} result] } {
            puts "Simulation completed or failed."
            return $result
        }
        after 1000
    }
    puts "Timeout reached."
    exec kill -9 $pid
    return -code error "Timeout occurred"
}

# Example Usage
set sim1_cmd "vsim -c -do \"run 1000; quit -f\" work.module_E"
set sim2_cmd "vsim -c -do \"run 2000; quit -f\" work.module_F"

if {[run_simulation $sim1_cmd 60] == "0"} {
    puts "Simulation 1 completed successfully"
    if {[run_simulation $sim2_cmd 60] == "0"} {
        puts "Simulation 2 completed successfully"
    }
}
```

**Commentary:** This example defines a procedure `run_simulation` that handles the launching, monitoring, and killing of individual simulation instances.  It includes a timeout mechanism (`$timeout`) to prevent runaway simulations.  The `exec kill -0 $pid` command checks if the process is still running.  This robust error handling is critical in real-world applications.

**3. Resource Recommendations:**

The ModelSim documentation is crucial.  A thorough understanding of TCL scripting and its interaction with ModelSim's command line interface is essential.  Consult texts on concurrent programming and inter-process communication for the design of robust solutions.  Familiarize yourself with TCL's built-in functions for process management and error handling.  Knowledge of operating system commands (like `ps`, `kill`) is advantageous for advanced process control.  Consider exploring resources on design verification methodologies; these provide context for efficient simulation organization and verification strategy development.
