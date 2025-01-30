---
title: "How can I create a Python C extension without distutils?"
date: "2025-01-30"
id: "how-can-i-create-a-python-c-extension"
---
Directly crafting Python C extensions without relying on `distutils` or its successor `setuptools` requires a deeper understanding of the Python C API and the underlying compilation process. I've had to resort to this approach in resource-constrained environments where relying on full build systems was impractical, or when specific custom build steps were needed that `distutils` made difficult. This technique, while more involved, offers unparalleled control over the build process and results in a very lightweight extension module.

The core challenge lies in manually compiling the C code and then linking it into a shared object (.so on Linux/macOS, .pyd on Windows) with the necessary Python library. This process involves several key steps: defining the Python module interface in C using the Python C API, compiling the C source into an object file, and then creating a shared library by linking the object file against the Python interpreter’s shared library. Finally, we ensure the resulting shared object is correctly importable by Python.

Here’s how I would accomplish this, broken into the necessary steps with corresponding code examples:

**Step 1: Define the Python module interface in C using the Python C API**

The Python C API is extensive, but our focus will be on functions and structures essential for defining a simple module and function.

```c
// example_module.c
#include <Python.h>

static PyObject* example_function(PyObject* self, PyObject* args) {
  int arg1, arg2;

  if (!PyArg_ParseTuple(args, "ii", &arg1, &arg2)) {
    return NULL; // Signal parsing error
  }

  int result = arg1 + arg2;
  return PyLong_FromLong(result);
}

static PyMethodDef example_methods[] = {
    {"example_function",  example_function, METH_VARARGS, "Adds two integers."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef example_module = {
    PyModuleDef_HEAD_INIT,
    "example_module",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    example_methods
};

PyMODINIT_FUNC PyInit_example_module(void) {
  return PyModule_Create(&example_module);
}
```
**Commentary:**

*   `#include <Python.h>` is essential; this header file exposes the Python C API.
*   `example_function`: This is the C function that will be accessible from Python. It takes `self` (module object) and `args` (arguments passed from Python) as inputs. `PyArg_ParseTuple` parses arguments using format strings. "ii" denotes two integers. It returns `NULL` on error and returns a `PyLong` object containing the sum.
*   `example_methods`: This array maps a Python function name ("example\_function") to its C implementation (`example_function`). `METH_VARARGS` indicates the function takes variable positional arguments, which it receives as a Python tuple.
*   `example_module`: This `PyModuleDef` structure contains the metadata for the module, including the name (`example_module`), method mappings (`example_methods`), and other fields.
*   `PyInit_example_module`: This initialization function is mandatory. It gets called when Python first imports the module, creating the module object from `PyModule_Create` with the metadata defined in `example_module`.

**Step 2: Compile the C source into an object file**

We must compile this C file into an object file (.o for Linux/macOS and .obj for Windows), and ensure we include the necessary include path to the Python headers. I typically use `gcc` for Linux/macOS or the equivalent on Windows (`cl.exe`), invoked directly from the shell.

```bash
# Linux/macOS example:
gcc -I/usr/include/python3.9 -c example_module.c -o example_module.o

# Adjust python include directory for your installed version:
# -I/usr/include/python3.10
# -I/usr/local/include/python3.11
# Example for Windows (using cl.exe from the Visual Studio command prompt):
# cl /I"C:\Path\To\Python\include" /c example_module.c /Foexample_module.obj
```

**Commentary:**
*  `-I/usr/include/python3.9` specifies the include path to where Python's header files are located. This path will vary based on your Python installation. Use `python3 -m site --user-site` for your Python install to determine site packages, and adjust accordingly. Ensure the major and minor Python version matches. On Windows, you may have to look inside the extracted Python installation, or obtain one from Windows installer.
* `-c`: This option tells the compiler to compile the C source code into an object file but not link it yet.
* `-o example_module.o`:  specifies the output file name for the compiled object file,

**Step 3: Create a shared library by linking the object file against the Python interpreter’s shared library**

The final step is linking the object file with Python's dynamic library (usually `libpython` or `pythonXY.dll`, where XY is Python’s major and minor version numbers). The exact library and linking method vary across platforms.

```bash
# Linux/macOS example:
gcc -shared example_module.o -lpython3.9 -o example_module.so

# Windows example (using cl.exe and link.exe):
# link /DLL example_module.obj /OUT:example_module.pyd /LIBPATH:"C:\Path\To\Python\libs" python39.lib

```

**Commentary:**

*   `-shared`:  instructs the compiler/linker to produce a shared library.
*   `-lpython3.9`:  tells the linker to include the Python library when creating the shared library. The name is platform specific.
*   `-o example_module.so`: specifies the output shared library file, ensuring the .so (Linux/macOS) or .pyd (Windows) extension is correct for Python import.
*   On Windows,  `/LIBPATH` points to the directory where `python39.lib` (or similar for your version) resides, and we link using `python39.lib`, which is a stub that facilitates linking to the actual DLL.

**Step 4: Ensure the resulting shared object is importable by Python**

After this build process, `example_module.so` (or `example_module.pyd` on Windows) should be in the current directory or a path recognized by Python’s module search paths. Then it is imported with a simple `import example_module`.

```python
# example.py
import example_module

result = example_module.example_function(5, 3)
print(f"The result is: {result}") # Output: The result is: 8
```

**Commentary:**

*   Importing the module will dynamically load the generated shared library.
*   Calling the function `example_function` demonstrates the module's interaction with Python.

**Resource Recommendations:**

Several resources can deepen one's understanding of this process. The official Python documentation, particularly the C API section, is a mandatory starting point. A well-regarded textbook on systems programming that includes sections on creating shared libraries is invaluable. Online documentation of `gcc` (or the platform-specific compiler/linker) is also helpful. Finally, searching for older documentation on Python C extension development before `distutils` became popular can provide additional insights, especially for understanding the raw mechanics of the process. While this older material may be less discoverable, its lack of reliance on higher level tools makes it very useful for someone who needs precise control over compilation and linking.

In summary, creating Python C extensions without `distutils` demands a careful, manual process. Understanding the interplay between C code, the Python C API, compiler options, and linker behavior is critical. While this approach is more complex, it's crucial in scenarios where fine-grained control over the build process or a lean build is necessary, mirroring my experiences with embedded systems and customized toolchains.
