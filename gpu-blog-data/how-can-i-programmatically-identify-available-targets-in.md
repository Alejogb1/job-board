---
title: "How can I programmatically identify available targets in a CUDA cubin?"
date: "2025-01-30"
id: "how-can-i-programmatically-identify-available-targets-in"
---
The structure of a CUDA cubin file, particularly its metadata section, allows for programmatic identification of available kernel entry points. This process relies on parsing the embedded ELF (Executable and Linkable Format) data and extracting specific symbols that denote kernel functions. My experience in optimizing kernel dispatch mechanisms has consistently involved direct manipulation of cubin files, and this technique is invaluable for building dynamic dispatch systems.

The cubin file, essentially a binary object, contains both the compiled PTX (Parallel Thread Execution) instructions for the GPU and metadata crucial for runtime interaction. Specifically, the ELF format within the cubin provides a standardized structure for this data. Key sections, such as `.symtab` (symbol table) and `.strtab` (string table), provide the basis for locating kernel entry points. The symbol table contains records, each referencing a specific symbol (function, variable, etc.) and its associated address within the cubin. The string table houses the human-readable names corresponding to these symbols. By processing these sections, it's possible to discern which functions are designated as kernels.

Kernel function symbols in a cubin file are typically identifiable by a combination of their type, binding, and naming convention. A kernel function will have a type flag that indicates it is a function, a binding flag that specifies global visibility (allowing it to be called from the host), and a name that typically adheres to specific conventions which indicate that the function is a kernel (e.g., often prefixed by `_Z7` followed by a mangled name representing the return type and function arguments). Additionally, CUDA driver API allows for introspection of loaded modules, but performing a manual parse of a cubin file gives you complete control and eliminates any dependence on driver API calls.

The process involves several key steps, starting with the cubin file being read into memory. We then locate the ELF header, which provides information about the overall structure of the file, including the locations of the section headers. We iterate through the section headers, identifying the symbol table and string table sections using their respective names (`.symtab` and `.strtab`). Subsequently, we parse the symbol table entries, analyzing each entry to determine if it represents a function and if it has a global binding. The name of the symbol is then retrieved from the string table using an offset stored in the symbol table entry. Finally, kernel candidates are filtered using naming conventions or additional metadata. Below I provide three Python code examples utilizing `struct` and `io` modules to demonstrate this process. Note, these examples assume a specific cubin format (64-bit ELF) and would require adjustment for other architectures or older CUDA versions. Further, this does not include robust error handling for clarity.

**Example 1: Basic ELF Header Parsing**

This first example illustrates parsing the ELF header to locate the section header offset and the section header size.

```python
import struct
import io

def parse_elf_header(cubin_data):
    # Parse ELF header (64-bit)
    e_ident = cubin_data.read(16)
    if e_ident[0:4] != b'\x7fELF':
        raise ValueError("Not an ELF file")
    if e_ident[4] != 2: # Check for 64-bit
        raise ValueError("Not a 64-bit ELF file")
    cubin_data.seek(16) # skip e_ident
    e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack("<HHIIQQIIHHHH", cubin_data.read(52))
    print(f"Section header offset: {e_shoff}")
    print(f"Section header entry size: {e_shentsize}")
    return e_shoff, e_shentsize, e_shnum

with open("my_kernel.cubin", "rb") as f:
    cubin_bytes = f.read()
    cubin_stream = io.BytesIO(cubin_bytes)
    section_offset, section_size, num_sections = parse_elf_header(cubin_stream)
```

This example utilizes the `struct` module to unpack the ELF header. The ELF header is the initial block of bytes defining the file structure. First, we verify the magic number for an ELF file and then for a 64-bit architecture. It extracts, prints, and returns key parameters for navigating the section headers (the offset, the size of each section entry, and the total number of sections).

**Example 2: Locating the Symbol Table and String Table**

Building on the prior example, this snippet searches for the symbol and string tables.

```python
def find_symtab_strtab(cubin_data, section_offset, section_size, num_sections):
    cubin_data.seek(section_offset)
    symtab_offset = 0
    symtab_size = 0
    strtab_offset = 0
    strtab_size = 0
    for i in range(num_sections):
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack("<IIQQQQIIQQ", cubin_data.read(64))

        if sh_type == 2 and sh_name == 1: # SHT_SYMTAB, this relies on specific cubin section name string table indexes.
                symtab_offset = sh_offset
                symtab_size = sh_size
        elif sh_type == 3 and sh_name == 2: # SHT_STRTAB
                strtab_offset = sh_offset
                strtab_size = sh_size
    print(f"Symbol table offset: {symtab_offset}, size: {symtab_size}")
    print(f"String table offset: {strtab_offset}, size: {strtab_size}")
    return symtab_offset, symtab_size, strtab_offset, strtab_size

with open("my_kernel.cubin", "rb") as f:
    cubin_bytes = f.read()
    cubin_stream = io.BytesIO(cubin_bytes)
    section_offset, section_size, num_sections = parse_elf_header(cubin_stream)
    symtab_offset, symtab_size, strtab_offset, strtab_size = find_symtab_strtab(cubin_stream, section_offset, section_size, num_sections)
```

In this code, the script iterates through each section header, identified through the offset read earlier.  The script checks the section type and name. `sh_name` values are highly dependent on how the cubin is built. Generally, name indexes 1 and 2 often correspond to `.symtab` and `.strtab` respectively in default cubin linking, though this is not guaranteed and robust implementations should retrieve this name to check. When the `SHT_SYMTAB` or `SHT_STRTAB` section types are encountered (identified by type numbers 2 and 3), the script extracts their corresponding offset and size. Again the values of 1 and 2 for name indexes are assumptions on the standard structure and are not guaranteed. The example then prints and returns these offsets and sizes for subsequent parsing.

**Example 3: Extracting Kernel Names**

Finally, this completes the process to pull kernel names out of the identified tables.

```python
def extract_kernel_names(cubin_data, symtab_offset, symtab_size, strtab_offset, strtab_size):
    cubin_data.seek(symtab_offset)
    kernel_names = []
    entry_size = 24 # Size of ELF symbol table entry (64-bit)

    for _ in range(symtab_size // entry_size):
        st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack("<IIIBQQ", cubin_data.read(entry_size))
        st_type = st_info & 0x0f  # Extract symbol type
        st_bind = st_info >> 4   # Extract symbol binding

        if st_type == 2 and st_bind == 1: # STT_FUNC, STB_GLOBAL
           cubin_data.seek(strtab_offset + st_name) # Seek to the name in the strtab
           name_bytes = b""
           while True:
                byte = cubin_data.read(1)
                if byte == b"\x00":
                   break
                name_bytes += byte
           name = name_bytes.decode("utf-8")
           if name.startswith("_Z7"): # This is not a strict check
            kernel_names.append(name)
    print(f"Kernel names: {kernel_names}")
    return kernel_names


with open("my_kernel.cubin", "rb") as f:
    cubin_bytes = f.read()
    cubin_stream = io.BytesIO(cubin_bytes)
    section_offset, section_size, num_sections = parse_elf_header(cubin_stream)
    symtab_offset, symtab_size, strtab_offset, strtab_size = find_symtab_strtab(cubin_stream, section_offset, section_size, num_sections)
    kernel_names = extract_kernel_names(cubin_stream, symtab_offset, symtab_size, strtab_offset, strtab_size)
```

This code processes each symbol table entry. It first unpacks the symbol information. Then it checks if the symbol corresponds to a function (`STT_FUNC`, value 2) and has global visibility (`STB_GLOBAL`, value 1). If both conditions are met, the code fetches the symbol name from the string table using the `st_name` offset, reading until a null terminator is encountered.  Finally, any kernel matching a naming convention is added to a list. The example concludes by printing and returning the list of identified kernel names. Note this name check using `_Z7` is not a robust method, and more thorough mangled name parsing should be done in practice for full reliability.

To further solidify my understanding and refine these techniques, I have relied on several resources. The ELF specification document provides the definitive description of the file format. I recommend consulting the documents on ELF file format and the System V ABI for detailed information about symbol table structures. While not targeted directly for cubin parsing, the documentation on the CUDA driver API, specifically related to module loading and inspection, gives complementary insights. Research papers dealing with binary analysis and reverse engineering are also helpful for understanding the broader context of this method. This combined approach of direct analysis with a solid foundation in the related specifications provides a comprehensive way to programmatically extract kernel entry points from CUDA cubin files.
