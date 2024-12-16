---
title: "How to save and access pickle/hdf5 files in Azure ML Studio?"
date: "2024-12-16"
id: "how-to-save-and-access-picklehdf5-files-in-azure-ml-studio"
---

Alright, let's tackle this pickle/hdf5 storage and retrieval issue within Azure ml studio. It's a common hurdle, and I’ve certainly spent my share of time navigating it in past projects, particularly when dealing with large model objects or complex datasets. The core challenge lies in efficiently managing data persistence across training runs and between different execution environments, and Azure ml studio has some specific considerations.

Essentially, you're working within a managed environment where you often need to persist data to a durable store that can be accessed by subsequent steps or different pipeline stages. This isn't a simple matter of saving files locally to the compute; those files vanish when the compute instance terminates. We need to leverage Azure blob storage or data stores provided by Azure ml studio. I'll outline the common approaches and provide code examples.

First, let's talk about the storage options. Azure ml studio provides several mechanisms, but for pickling and hdf5 files, the primary choices are either using a registered datastore linked to an azure blob storage container or directly utilizing azure blob storage. I've generally favored the datastore approach for its simplicity in managing connectivity details within the context of azure ml experiments. This method avoids hardcoding storage credentials within your scripts which is very good practice from a security point of view.

Here’s how I’d approach saving a pickled object using a registered datastore:

```python
import pickle
import os
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.run import Run

# Fetch the run context to gain access to the workspace and run
run = Run.get_context()
ws = run.experiment.workspace

# Assuming you have a datastore named 'myblobstore' registered
datastore = Datastore.get(ws, 'myblobstore')

# Dummy object to pickle
data_to_pickle = {'model_parameters': [1,2,3], 'training_results': {'accuracy': 0.95, 'loss': 0.05}}

# Path within the blob storage to save the pickle file
pickle_path = 'my_pickled_data/model_object.pkl'

# Create the directory if it doesn't exist. This will create a pseudo-directory within the blob.
os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

# Upload the pickled data using the datastore
with open("temp_pickle.pkl","wb") as f:
    pickle.dump(data_to_pickle, f)
datastore.upload_files(files=['temp_pickle.pkl'], target_path = pickle_path, overwrite=True)
os.remove("temp_pickle.pkl") # Clean up temp file
print(f'Pickle data saved to {pickle_path} in datastore {datastore.name}')

```

In the example above, I’m first obtaining a reference to my workspace and then retrieving my previously registered datastore. I create a simple dictionary as an example and then I dump the object to a temporary file before using `datastore.upload_files` method, to upload the pickled object to the blob. The `target_path` parameter is a crucial element, as it allows you to establish a logical folder structure within your blob storage. After saving the pickled file to the blob storage, the temporary file is cleaned up.

Now, let's look at retrieving that pickled object in another script or pipeline step. This is equally straightforward:

```python
import pickle
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.run import Run

# Fetch the run context
run = Run.get_context()
ws = run.experiment.workspace

# Get the same datastore we used to save the file
datastore = Datastore.get(ws, 'myblobstore')

# Path to the pickle file. Must match the path used when saving.
pickle_path = 'my_pickled_data/model_object.pkl'


# Download the pickle file and load the object
download_path = './'
datastore.download(target_path=download_path, prefix=pickle_path, overwrite=True)

with open(os.path.join(download_path, pickle_path.split('/')[-1]),"rb") as f:
    loaded_data = pickle.load(f)
print('Successfully loaded data')
print(loaded_data)

```

Here, I’m using the `datastore.download` method to retrieve the previously saved file to the current directory. The `prefix` argument specifies which files or directories should be downloaded, and again, consistency in these paths is absolutely vital. Once downloaded, it is loaded using the pickle library. Note that I am creating a local download path, and then specifying to download a specific subfolder within the blob. The file will be available inside the local download path with the filename specified within that subdirectory. After the download is complete, the content can be loaded from the file.

Now, regarding hdf5 files, the process is very similar. Hdf5 files are often used with models that involve numpy arrays or other structured data. Here is an example of how to save a hdf5 file:

```python
import h5py
import numpy as np
import os
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.run import Run

# Fetch the run context
run = Run.get_context()
ws = run.experiment.workspace

# Get the datastore
datastore = Datastore.get(ws, 'myblobstore')

# Sample hdf5 data
data = np.random.rand(100, 100)

# Path to the hdf5 file
hdf5_path = 'my_hdf5_data/my_dataset.hdf5'
# Create the directory if it doesn't exist.
os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

# Saving hdf5 to temporary file.
with h5py.File('temp_hdf5.hdf5', 'w') as hf:
    hf.create_dataset('dataset_1', data=data)
# Upload the hdf5 file to the datastore
datastore.upload_files(files=['temp_hdf5.hdf5'], target_path = hdf5_path, overwrite=True)
os.remove("temp_hdf5.hdf5")
print(f'HDF5 data saved to {hdf5_path} in datastore {datastore.name}')
```

And here’s how you would retrieve it.

```python
import h5py
import numpy as np
import os
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.run import Run

# Fetch the run context
run = Run.get_context()
ws = run.experiment.workspace

# Get the datastore
datastore = Datastore.get(ws, 'myblobstore')

# Path to the hdf5 file, must be the same as the saved path.
hdf5_path = 'my_hdf5_data/my_dataset.hdf5'

# Download the hdf5 file from the datastore
download_path = './'
datastore.download(target_path=download_path, prefix=hdf5_path, overwrite=True)

# Load the hdf5 data
with h5py.File(os.path.join(download_path,hdf5_path.split('/')[-1]), 'r') as hf:
    loaded_data = hf['dataset_1'][:]
print("HDF5 data loaded successfully")
print(loaded_data)
```

The code follows the same principles as with pickling: we save the hdf5 file to a temporary path, upload it to the datastore, and then download it when needed in a subsequent step. Again, the paths must match.

A few important tips to keep in mind:

*   **Datastore Registration:** Make sure you've registered your blob storage as a datastore in your Azure ml workspace before running these codes.

*   **Path Consistency:** Be rigorous about path management. One typo, and you're not going to find your persisted data.

*   **Error Handling:** Always add proper error handling around your file operations, downloads, and uploads.

*   **Large Data:** If you are dealing with very large files, consider leveraging more optimized methods within the azure SDK, such as the `upload` and `download` methods of the `BlobClient`, which allow for streaming the upload to avoid out of memory issues.

*   **Version Control:** Azure ML also offers built-in versioning for datasets, which could be considered if you want a more integrated approach to tracking changes to your saved data. Explore Azure ML datasets for potentially more efficient dataset management and versioning.

For more detailed information on datastore management, I'd highly recommend reviewing the official Azure ML documentation, particularly the sections on `azureml.core.Datastore` and `azureml.core.Dataset`. Additionally, “Programming Machine Learning: From Data to Deployable Models” by Paolo Perrotta provides a good foundation on general data persistence strategies within machine learning workflows. Also, the documentation for the python packages used here are key, including the *pickle* package and the *h5py* package.

These examples should get you started on saving and loading pickle and hdf5 files reliably within Azure ml studio. The key is meticulous path management and proper understanding of the storage mechanisms provided by the platform. It's not that difficult, once you’ve wrapped your head around these concepts.
