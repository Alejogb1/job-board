---
title: "How to prevent fastai fastbook from requesting access to Google Drive when running in Google Colab?"
date: "2024-12-15"
id: "how-to-prevent-fastai-fastbook-from-requesting-access-to-google-drive-when-running-in-google-colab"
---

alright, so you're having that annoying fastai fastbook google drive prompt pop up in colab, got it. yeah, i've been there, that dance with google's authorization flow. it’s something that can get old real quick, especially when you're just trying to iterate on a model or follow along with the fastbook. i’ve spent more hours than i care to remember messing around with permissions and mount points, so i've had a couple of go-to solutions that usually do the trick.

the core issue here is that fastbook, by default, tries to cache datasets and other files in your google drive. it makes sense for persistent storage between sessions, but it's not ideal if you want to avoid the authorization step every single time. it’s a convenience thing for the book authors but ends up as an inconvenience for people in colab, it kind of seems backwards to me. i think in the first few times i went through it i kept clicking allow and it felt like a chore.

the first thing i usually try is to explicitly tell fastai to use the colab instance's local storage instead. this sidesteps the whole google drive thing entirely. how can we achieve that? well it’s fairly straightforward. instead of letting fastai default to some place in drive just define our own local data path.

here's the code snippet:

```python
import os
from fastbook import *

# set a data path to the /content directory
path = Path('/content/')
# fastbook has some built-in data download and path handling, lets make it work in our local folder
URLs.path = path/'data'

# check if the data path already exists, create if it does not
if not os.path.exists(URLs.path):
  os.makedirs(URLs.path)

# from now on, fastai will store datasets and data in our local directory
```

what's happening here? we are importing the necessary packages, defining a local path, setting the fastai's url path to it and checking if it exists, creating it if it does not. simple enough. this way all the data fastbook tries to download and the places that expect some datasets are going to be within colab's local /content path. now, important caveat: this means that every time you reset the colab runtime (or it gets reset automatically), you'll lose the downloaded datasets. for most fastbook learning, this is not a big deal, and it will download those datasets again as you go through the code. you can always save your trained model in google drive using a custom code and load it later.

the second approach i usually use, which kind of works as a more 'fire and forget' is to use the `untar_data` function with the `force_download=True` parameter, if you don't want to change default directories globally. this forces fastai to download the data again even if it thinks it already has it somewhere (usually on a mounted google drive). this might sound a bit counter-intuitive because it re-downloads stuff but the positive side of it is that it does not require any change on global configurations and if it was not downloaded before, it also downloads it locally to the /content folder.

here is how it looks in code:

```python
from fastbook import *
from fastai.data.external import *

# the imagenette dataset is common in fastbook
# this downloads it locally to the /content directory
path = untar_data(URLs.IMAGENETTE, force_download=True)

# you can then proceed to use it as usual
dls = ImageDataLoaders.from_folder(path, item_tfms=Resize(224), bs=64)

```

here, the `force_download=True` tells the `untar_data` function to download the data even if it thinks it's already available. so again, local storage, no more google drive prompts. in the process of downloading, if you notice carefully, fastai is downloading to the default place but when the `force_download` is set to true it's also cached locally in colab. i used the imagenette dataset as an example because it's used in many places of the fastbook.

the last trick i use sometimes, if the two options above don't work properly (happens sometimes, very rare) is to explicitly download the datasets to a local folder using the `download_url` function from fastai and then refer to the files locally. this is a bit more manual but gives you complete control over the process. it’s like a last resort but i have to say, it’s bulletproof. this approach comes in handy when you're not using the standard fastai datasets or if you are doing something that isn't from the book that involves fetching a dataset online.

here’s the example:

```python
from fastbook import *
from fastai.data.external import *
import os

# define your data path
data_path = Path('/content/my_data')
if not os.path.exists(data_path):
    os.makedirs(data_path)

# urls are usually saved in URLs class in fastai
# you can also point to your own custom urls for datasets
data_url = URLs.IMAGENETTE

file_name = data_url.split('/')[-1]
local_file_path = data_path/file_name

if not os.path.exists(local_file_path):
  #download the data into a local path
  download_url(data_url, local_file_path)
  # this also includes extracting the zip file if it is a zip
  untar_data(local_file_path, dest=data_path)

# your dataset is now in the /content/my_data directory
# you can proceed to use it with fastai's utilities
dls = ImageDataLoaders.from_folder(data_path, item_tfms=Resize(224), bs=64)
```

in this code, we're creating a directory `/content/my_data`, downloading the dataset to a local path and extracting it if needed and then passing it to the `ImageDataLoaders`. that way, you never mount google drive and do everything from scratch in colab. as i mentioned, this is like the nuclear option. you really don’t need it most of the time. but it's good to have in the toolbox.

now, for those interested in diving deeper, i’d recommend having a look into how fastai handles `URLs` and download paths, specifically in their source code. it's usually in the `fastai.data.external` module. understanding the mechanism of where fastai expects the datasets to be will be helpful for troubleshooting any unexpected problems. the fastai documentation pages also sometimes are helpful to understand the nuances. for example the `untar_data` or `download_url`. but i find their code easier to understand. there's a book called “deep learning for coders with fastai and pytorch” that goes into the guts of the library, but it is more of a practical guide than a deep dive on internals. you can also check out the official pytorch documentation, particularly the datasets api, which might be useful in case you want to extend to a custom dataset. also, take a look at colab’s documentation itself for more information about using the storage in colab runtimes.

i've been using fastai for a long time now, and i've found that these three approaches cover almost every scenario. and that includes those times when i was testing different versions of the library and all sorts of experiments that forced me to understand better how everything worked internally (and not internally). and, by the way, did you hear about the data scientist who couldn't find their keys? they were always in the cloud. he he he. that joke was terrible i know.

hope this helps you avoid that google drive authorization popup. if you have other issues feel free to ask.
