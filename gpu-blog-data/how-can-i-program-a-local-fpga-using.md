---
title: "How can I program a local FPGA using Vivado installed on a server?"
date: "2025-01-30"
id: "how-can-i-program-a-local-fpga-using"
---
Integrating FPGA development into a server-based workflow, while seemingly complex, becomes manageable through careful orchestration of Vivado and remote access protocols. I've personally navigated this setup within a continuous integration (CI) environment, realizing significant benefits in terms of resource management and collaborative development. The key lies in decoupling the hardware connection to the FPGA from the Vivado installation, allowing multiple developers to compile and deploy to a shared device.

The central concept here revolves around employing a dedicated hardware access server, separate from the Vivado installation server, and establishing communication between the two. The Vivado server performs the synthesis, implementation, and bitstream generation; it *does not* directly interact with the target FPGA hardware. The hardware server, typically a machine with a USB-to-JTAG interface or a dedicated debug probe, solely handles programming the FPGA using the bitstream produced by the Vivado server. This two-tier architecture is crucial for scalability and efficient hardware resource sharing.

The Vivado server requires a specific configuration to operate headlessly, meaning without a graphical user interface. The primary mode of interaction will be through the Tcl scripting interface, which allows complete automation of the design flow. Scripting every stage, from project creation to bitstream generation, enables consistent and reproducible builds. Setting up Vivado on a server involves installing the software and configuring licensing through environment variables to avoid manual interaction. The licensing must be valid for command line usage and should ideally use a floating license server for concurrency. A shared network storage location for project files and design sources is essential, providing a central repository for the team. The target FPGA device's part number must be specified correctly in the project to ensure that bitstreams are generated that are compatible with the specific hardware. Finally, server-side scripting will invoke Vivado in batch mode for synthesis, implementation, and bitstream generation.

On the hardware access server, specialized software is required. Xilinx provides the `hw_server` component within the Vivado installation, which allows remote access to a programming cable. Alternatively, open-source tools such as `openocd` can be utilized. In this setup, the Vivado server communicates with the hardware server to program the FPGA using the generated bitstream. The specific commands for this communication will depend on the chosen tool and the network setup. The hardware access server typically requires a static IP address or a method for the Vivado server to locate it. Finally, security is paramount; communication should be limited to a trusted network segment, and the servers should have restricted access.

Consider the following Tcl scripts, which exemplify the automation process:

**Example 1: Project Setup and Synthesis (Vivado Server Script)**

```tcl
# Example: create_project.tcl
# Assuming project directory, part number, and source file are defined elsewhere.

# Create project
create_project -force project_name project_dir -part xc7z020clg400-1

# Add source file(s)
add_files -norecurse [file join source_dir top_level.vhd]
add_files -fileset sim_1 -norecurse [file join source_dir testbench.vhd]

# Set top module
set_property top top_level [current_fileset]
set_property design_mode RTL [current_fileset]

# Run synthesis
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# Exit
exit
```

This script, executed by Vivado in batch mode using `vivado -mode batch -source create_project.tcl`, illustrates the basic project creation and synthesis steps. It first creates the project, specifies the target FPGA part, adds the required source files and specifies the top module. Subsequently, it initiates synthesis using multiple threads (`-jobs 8`) for accelerated computation. The `wait_on_run` command ensures the script pauses until the synthesis completes. The exit statement cleanly terminates Vivado after all operations. This script, combined with others, forms the foundation of a completely automated build pipeline. The `project_dir` and `source_dir` variables, while omitted here, would typically be defined at a higher level to allow project-specific configurations without modifying core scripts.

**Example 2: Implementation and Bitstream Generation (Vivado Server Script)**

```tcl
# Example: implement_and_generate.tcl
# Assuming a synthesized design already exists.

open_project project_dir/project_name.xpr

# Run implementation
launch_runs impl_1 -jobs 8
wait_on_run impl_1

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1_write_bitstream

# Get the path to bitstream file
set bitstream_file [get_property BITSTREAM.FILE [get_runs impl_1_write_bitstream]]

puts "Bitstream file generated: $bitstream_file"

# Exit
exit
```

This script demonstrates the implementation and bitstream generation. It assumes that the project already exists and that synthesis has been completed. It opens the existing project using `open_project`, then initiates implementation. Following successful implementation, it generates the bitstream. The script then retrieves and outputs the path to the generated bitstream file. This bitstream is the crucial file for programming the FPGA. This output would typically be captured by the CI system or a management script, and then transferred to the hardware access server for deployment.

**Example 3: Bitstream Transfer and FPGA Programming (Hardware Server Script - Pseudocode)**

```python
# Example: program_fpga.py
# Python code (pseudocode using a library like fabric for SSH)

import subprocess
import fabric

# Define constants or load from configuration file.
SERVER_IP = "192.168.1.100" # Replace with the Vivado Server IP address
BITSTREAM_PATH_SERVER = "/path/to/bitstream/on/vivado/server/top_level.bit"
BITSTREAM_PATH_LOCAL = "/tmp/top_level.bit" # Temporary location on hardware server.
USERNAME = "user" # Replace with the Vivado Server username
PASSWORD = "password" # Replace with password (use secure key-based auth in production)
HARDWARE_SERVER_IP = "192.168.1.101" # Replace with hardware server IP address
JTAG_CHAIN = 0 # Device position in the JTAG chain


def copy_bitstream():
    c = fabric.Connection(host=SERVER_IP, user=USERNAME, connect_kwargs={"password": PASSWORD})
    c.get(BITSTREAM_PATH_SERVER, BITSTREAM_PATH_LOCAL)

def program_fpga():
    # Option 1: Use Vivado hw_server
    command = f"hw_server -s {HARDWARE_SERVER_IP} && xsdb -batch -source program_script.tcl"
    subprocess.run(command, shell=True, check=True) # Use shell=False to avoid command injection

    # Option 2: Use OpenOCD (requires a suitable configuration file)
    # command = f"openocd -f board_config.cfg -c 'program {BITSTREAM_PATH_LOCAL}' -c shutdown"
    # subprocess.run(command, shell=True, check=True)

def create_program_script():
    with open("program_script.tcl","w") as f:
      f.write(f"connect\n")
      f.write(f"fpga -f {BITSTREAM_PATH_LOCAL}\n")
      f.write(f"exit\n")

if __name__ == "__main__":
    copy_bitstream()
    create_program_script()
    program_fpga()
```

This Python pseudocode illustrates a basic script executing on the hardware server. It uses the Fabric library (installable with `pip install fabric`) for SSH file transfers. The script first copies the generated bitstream file from the Vivado server to a temporary location on the hardware server. It then creates a `program_script.tcl` and executes the `hw_server` command to connect to the local jtag chain and program the FPGA. The `hw_server` tool communicates with the hardware access server to execute programming. The `subprocess.run` function is utilized to execute the command, ensuring proper error handling. Alternatively, an `openocd` approach is commented, demonstrating the potential to leverage open-source tooling. This script highlights the critical element of separating bitstream generation from the actual FPGA programming.

For additional information, I suggest exploring resources focused on: 1) Vivado Tcl scripting guides, offering comprehensive details on the available commands and options; 2) Xilinx documentation on the `hw_server` tool, which provides technical specifications and troubleshooting procedures; 3) open-source community forums discussing best practices for headless FPGA development workflows; 4) tutorials on SSH and remote execution, which provide a foundation for remote management of the servers. These sources will enable the reader to build a robust understanding of each component involved in a server-based FPGA development setup.
