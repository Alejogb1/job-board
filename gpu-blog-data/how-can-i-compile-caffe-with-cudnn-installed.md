---
title: "How can I compile Caffe with cuDNN installed in a user's home directory (without root privileges) when the compiler cannot find it?"
date: "2025-01-30"
id: "how-can-i-compile-caffe-with-cudnn-installed"
---
The challenge of compiling Caffe with a locally installed cuDNN library, without root privileges, frequently arises due to the compiler's default search paths. Specifically, the compiler and linker typically look in standard system locations like `/usr/lib`, `/usr/local/lib`, and `/lib` for necessary shared libraries. When cuDNN resides in a user’s home directory, these locations are absent from the default search path, leading to compile-time and link-time errors.

The root cause is that compilation involves several stages, and each stage needs to be informed about the custom location of cuDNN. The preprocessor, compiler, and linker must all be aware of the library’s location. The preprocessor needs to find the cuDNN header files (typically `.h`), the compiler needs to generate object files containing calls to cuDNN functions, and the linker must resolve these calls by linking against the cuDNN shared library (typically `.so` on Linux). When any of these steps fails to locate cuDNN, compilation will halt.

The solution revolves around explicitly guiding the compiler and linker to the cuDNN files by adjusting search paths and library flags. I've addressed this precise scenario numerous times while working on personal deep learning projects where I opted for non-standard installation locations.

The first step is to inform the preprocessor about the location of the cuDNN header files. Typically, these are located within an `include` subdirectory inside the cuDNN installation path. This is achieved through the `-I` flag, which adds a specific directory to the preprocessor's search path. The following makefile snippet shows how to use it, assuming that cuDNN was extracted to `~/local/cudnn`:

```makefile
INCLUDE_DIRS := -I$(HOME)/local/cudnn/include

%.o: %.cpp
    $(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o $@ $<
```

In this example, `INCLUDE_DIRS` is a variable that holds the `-I` flag and the path to the cuDNN include directory. This variable is then used when compiling the source file, enabling the preprocessor to locate cuDNN headers. If the header files are placed in a location other than a subfolder named "include" inside the cuDNN folder, then the path must match. It's crucial to remember that paths must be defined with absolute paths for reliability, and any relative paths can cause build errors.

Secondly, the linker needs to know where to locate the cuDNN shared library. This is achieved through the `-L` flag, which specifies a search path for the linker, and `-l` flag, which specifies the name of the library to link against (without the `lib` prefix or the `.so` suffix). Additionally, the runtime loader needs to know where to locate the library at runtime, which can be addressed by setting the `LD_LIBRARY_PATH` environment variable.

Consider the following makefile snippet showing these flags:

```makefile
CUDNN_PATH := $(HOME)/local/cudnn
LIBRARY_DIRS := -L$(CUDNN_PATH)/lib64
LIBS := -lcudnn -lcudart

all: caffe

caffe: main.o
	$(CXX) $(CXXFLAGS) $(LIBRARY_DIRS) $(LIBS) -o $@ $^

run: caffe
    LD_LIBRARY_PATH=$(CUDNN_PATH)/lib64 ./caffe
```

Here, `CUDNN_PATH` defines the root directory where cuDNN is installed. `LIBRARY_DIRS` adds the path to the linker’s search path. `LIBS` specifies the cuDNN and CUDA runtime libraries to link against. In the `run` target, the `LD_LIBRARY_PATH` variable is set to include the library directory before the executable runs. The `lib64` extension to the directory name might need to change to `lib` depending on the system architecture and the actual directory containing the `.so` files. Furthermore, note that `cudart` might not be needed by a specific Caffe build but it is often necessary as a pre-requisite and is good practice to be included.

The combination of these compiler and linker flags, and the environment variable setting, typically resolves the “cannot find” issue. However, sometimes Caffe's build system is configured in a way that bypasses these standard flags or relies on a slightly different naming convention. In that case, modifying Caffe's makefile or configuration script directly is necessary.

Here's a conceptual adjustment of Caffe's `Makefile.config`, assuming the user is modifying the makefile directly. I've seen similar problems within the `Makefile.config` in Caffe itself, and have addressed them similarly:

```makefile
# Example of a section in Makefile.config
CUDA_DIR ?= /usr/local/cuda # Original path
# Modified section
CUDA_DIR := /usr/local/cuda
CUSTOM_CUDNN_DIR := $(HOME)/local/cudnn # Custom cuDNN path
INCLUDE_DIRS += -I$(CUSTOM_CUDNN_DIR)/include
LIBRARY_DIRS += -L$(CUSTOM_CUDNN_DIR)/lib64
# Ensure the library flag includes the location from the above path
LIBRARIES += -lcudnn -lcudart 
```

In this conceptual example, I've defined a new variable `CUSTOM_CUDNN_DIR` pointing to the user’s custom cuDNN location. Then, I explicitly added the required paths to the existing `INCLUDE_DIRS` and `LIBRARY_DIRS` variables used by Caffe's makefile. Lastly, the library flags are added to the `LIBRARIES` variable, making the explicit linking of the cudnn libraries. Note that the exact names of the variables used by Caffe in `Makefile.config` might differ slightly and need to be reviewed and adapted to the user's Caffe distribution. The usage of a `+=` operator, rather than an `=` ensures that the new paths are added to the existent ones, and are not overwritten.

Finally, it is important to ensure that the installed CUDA toolkit version and the cuDNN version are compatible. Version mismatch issues can lead to runtime errors that are difficult to debug. Furthermore, it is good practice to double-check that the architecture of the cuDNN library is correct for the system on which the Caffe project is being built.

Regarding recommendations for further resources, the official documentation for the C++ compiler being used (e.g. g++) is crucial. The documentation regarding makefiles is also very useful, specifically the rules for variables, targets, and dependencies. Additionally, the NVIDIA cuDNN documentation provides essential details regarding the library paths and usage. Finally, it can often be helpful to read the Caffe documentation and example makefiles to understand the build process in greater detail. Reviewing these resources helps ensure proper configuration and troubleshooting of compilation issues.
