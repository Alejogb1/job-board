---
title: "How to resolve 'NotADirectoryError: 'Errno 20' Not a directory' when loading a zip file in Google Colab?"
date: "2025-01-30"
id: "how-to-resolve-notadirectoryerror-errno-20-not-a"
---
The "NotADirectoryError: [Errno 20] Not a directory" when attempting to load a ZIP file in Google Colab typically indicates that the path provided to the ZIP handling function does not point to a directory, but instead to a file, or possibly does not exist at all. This error often arises because users mistakenly treat the ZIP archive itself as the target directory for extraction or other operations, rather than the directory *where* the archive is located. Over years of debugging similar file system issues, I've found that this mistake is rooted in a misunderstanding of file paths and how libraries interact with them.

The core issue is that functions like `zipfile.ZipFile()` or `shutil.unpack_archive()` require the path to the ZIP file itself, not a hypothetical "contents" directory. Subsequently, functions such as `os.listdir()` or `glob.glob()` that are designed to operate on directories will naturally raise this error if given the path to the ZIP archive instead. The error essentially means the system is being asked to view a file as a container that holds other files, instead of the storage unit itself. Resolving it involves correctly identifying and utilizing the actual path to the ZIP file and the intended destination for any extraction.

Let's consider a few scenarios and code examples to illustrate common pitfalls and proper solutions.

**Scenario 1: Incorrect Path for ZIP File Extraction**

A common mistake is assuming that the path provided to `zipfile.ZipFile` is where the contents should be extracted to rather than *where* the zip file exists. Consider this erroneous code:

```python
import zipfile

zip_path = '/content/my_archive.zip'  # Incorrectly assumed target directory
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path)  # This is the error!
except FileNotFoundError:
    print("Zip file not found at specified path")
except NotADirectoryError:
    print("NotADirectoryError raised. The path points to a file, not a directory.")
```

In this example, `/content/my_archive.zip` is indeed the path to the ZIP file. However, it is also being provided as the extraction destination to `zip_ref.extractall()`. The function interprets this as trying to create a subdirectory with the same name as the ZIP file. Since it is not a directory, the `NotADirectoryError` occurs.

The fix lies in providing a distinct *directory* path for extraction:

```python
import zipfile
import os

zip_path = '/content/my_archive.zip'
extract_path = '/content/extracted_archive' # Correct extraction directory
try:
    os.makedirs(extract_path, exist_ok=True) # Ensure the output folder exists
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path) # Correct path usage
    print(f"Zip file extracted to: {extract_path}")
except FileNotFoundError:
    print("Zip file not found at specified path")
except NotADirectoryError:
    print("NotADirectoryError raised. Check provided output path is valid.")
```

Here, `/content/extracted_archive` is explicitly created as the target directory for extraction. This separates the path to the ZIP archive itself and the desired output location, resolving the original error. `os.makedirs(extract_path, exist_ok=True)` ensures that the output folder exists before attempting to write anything to it and that it won't throw an error if the folder already exists. This is a good practice.

**Scenario 2: Mistaking ZIP File Path for Directory Content**

Another common mistake occurs when users assume the ZIP file's path can be used directly to list its contents or find specific files *inside* the archive. Consider this erroneous approach using `glob.glob`:

```python
import glob

zip_path = '/content/my_archive.zip'
try:
    files = glob.glob(f"{zip_path}/*") # Incorrect use of ZIP path
    print(files)
except NotADirectoryError:
    print("NotADirectoryError: The provided path is not a directory.")
```

`glob.glob()` is designed for working with files and directories *on the file system*, not for interacting directly with files inside a ZIP archive. The above code will treat the ZIP file as a directory, which is incorrect. It will raise a `NotADirectoryError` as a result.

To solve this, we must extract the contents of the zip file, which we have already seen in our previous example, and then we can use `glob.glob()`:

```python
import glob
import zipfile
import os

zip_path = '/content/my_archive.zip'
extract_path = '/content/extracted_archive'

try:
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    files = glob.glob(f"{extract_path}/*") # Correct use of extracted path
    print(files)
except FileNotFoundError:
    print("Zip file not found at specified path")
except NotADirectoryError:
        print("NotADirectoryError raised. Check provided output path is valid.")
```

Here, the ZIP file is first extracted into the `/content/extracted_archive` directory. We can then use `glob.glob` on that directory, which allows for finding specific files. This approach ensures that the file system is used correctly for directory operations, and the archive's contents are extracted so they are accessible.

**Scenario 3: Misusing `shutil.unpack_archive` with Incorrect Paths**

The `shutil.unpack_archive()` function, while convenient, can also lead to errors if used improperly, much like the previous example with `zipfile.ZipFile`. Consider this incorrect usage:

```python
import shutil

zip_path = '/content/my_archive.zip'
try:
  shutil.unpack_archive(zip_path, zip_path)  # Incorrectly providing zip path as output
except FileNotFoundError:
  print("Zip file not found at specified path")
except NotADirectoryError:
  print("NotADirectoryError. Check provided paths are valid")
```

Similar to scenario 1, this attempts to use the ZIP file path itself as the extraction location, triggering the `NotADirectoryError`.

To correct this, provide a distinct directory as the extraction target:

```python
import shutil
import os

zip_path = '/content/my_archive.zip'
extract_path = '/content/extracted_archive'

try:
    os.makedirs(extract_path, exist_ok=True)
    shutil.unpack_archive(zip_path, extract_path) # Correct extraction path provided
    print(f"Zip file extracted to: {extract_path}")
except FileNotFoundError:
  print("Zip file not found at specified path")
except NotADirectoryError:
  print("NotADirectoryError. Check provided paths are valid")
```

By explicitly setting `extract_path` as the destination, the error is resolved. The `unpack_archive` function correctly extracts the ZIP's content to a valid directory.

In summary, resolving "NotADirectoryError" during ZIP file handling on Google Colab is achieved by correctly distinguishing between the ZIP file's path and the desired extraction directory. One should always ensure to use separate paths for the ZIP file itself and the directory where its content should be extracted, using functions like `os.makedirs` to ensure that these directories exist and avoid other errors. Careful consideration of how libraries interact with file paths and the file system in general, is crucial to avoid such mistakes.

For further exploration, review the documentation for Python's `zipfile` module, the `shutil` module, and the `os` module. Additionally, examining the documentation on file paths and directory structures in any Linux system will offer further clarity. Finally, gaining a better understanding of file permissions may reveal errors where user access permissions can throw unexpected file-system errors.
