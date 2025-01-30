---
title: "Why does collectstatic re-include deleted files in my Wagtail project?"
date: "2025-01-30"
id: "why-does-collectstatic-re-include-deleted-files-in-my"
---
In my experience managing deployments of Wagtail-based applications, encountering the re-inclusion of deleted files by `collectstatic` is a common, and often frustrating, issue. This behavior arises from the interplay between Django's static file handling, the storage mechanisms used for static files, and the nature of `collectstatic` itself. `collectstatic` is fundamentally designed to gather *all* static files referenced within your project, regardless of their presence in the current filesystem. It’s not a differential sync tool; rather, it’s an aggregator.

The root cause isn't an issue within Wagtail itself, but rather the way Django handles static file locations and the process by which `collectstatic` locates and copies them. Specifically, `collectstatic` searches across all installed apps and configured static file directories as defined in your Django settings (primarily `STATICFILES_FINDERS`, `STATIC_ROOT`, and `STATICFILES_DIRS`). If a file, even one previously deleted from your project's source code directory, still exists within any location specified in the static file finders, it will be collected and copied to your `STATIC_ROOT` destination directory.

This can manifest unexpectedly when dealing with version control. For instance, consider a scenario where you remove an asset, say an image file named `old_image.jpg`, from your `assets` directory within a Django app. You commit and push this change, and then, during deployment, you run `collectstatic`. To your surprise, `old_image.jpg` reappears in the static directory. This happens because, potentially, a previous `collectstatic` operation copied the old file to your `STATIC_ROOT` (which might be a deployment-specific directory, like `/var/www/myproject/static`), and that directory is not routinely cleared before each deployment or `collectstatic` run.

`collectstatic` effectively uses a *copy-over* mechanism, not a synchronized delta update. It goes to all declared locations, checks for files, and if present, copies them to `STATIC_ROOT`. It doesn't track whether files were removed from the source; it just sees what's present in the locations it's configured to check. The consequence is that files previously copied to the `STATIC_ROOT` persist there until specifically deleted, and each execution of `collectstatic` essentially adds any files found in the static locations, including the existing ones, unless specifically handled.

Let’s illustrate this with code examples and commentary. First, consider this basic configuration in a `settings.py` file:

```python
# settings.py

STATIC_URL = '/static/'
STATIC_ROOT = '/var/www/myproject/static'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'assets'),
]
```

In this instance, `STATIC_ROOT` points to a directory outside the project's source code. `STATICFILES_DIRS` includes a relative path to an `assets` folder. If `old_image.jpg` existed in this `assets` directory once, it would be copied to `/var/www/myproject/static` by `collectstatic`. Later, even if removed from `assets`, it would remain in the `/var/www/myproject/static` directory, and, if the directory isn't cleared before running `collectstatic` again, it would continue to exist. The behavior of `collectstatic` here is to effectively "refresh" the state of files it finds; it doesn't remove files.

Now, consider an example with multiple apps:

```python
# settings.py (continued)

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',
    'wagtail.core',
    'wagtail.admin',
    'wagtail.documents',
    'wagtail.images',

]
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
]

```

`AppDirectoriesFinder` will look for `static` directories inside the `myapp` application and every app listed under `INSTALLED_APPS`. Consider if the `myapp` application also had the file `old_image.jpg` present at some point in its `/myapp/static/images/` directory; it would be added to `STATIC_ROOT`. Even if `old_image.jpg` was deleted from that `/myapp/static/images/` directory it can still reside in `/var/www/myproject/static` and persist through subsequent `collectstatic` operations. This is true for all installed applications.

Finally, let's look at an example involving third-party libraries or external directories. This is often an overlooked area:

```python
# settings.py (continued)
import os

THIRD_PARTY_STATIC_DIR = os.path.join('/path/to/third/party/statics')

STATICFILES_DIRS.append(THIRD_PARTY_STATIC_DIR)
```

Here, even if `old_image.jpg` never resided in our project or in application `static` directories, but existed in the `/path/to/third/party/statics` at some point, it could end up in `/var/www/myproject/static`. If later removed from `/path/to/third/party/statics`, a subsequent run of `collectstatic` would not remove it from the destination, `/var/www/myproject/static` directory. This is crucial, as third-party libraries or manually added locations are also sources that `collectstatic` examines.

To manage this behavior, I have found several effective strategies. The primary one is to explicitly clear the `STATIC_ROOT` directory *before* running `collectstatic` during the deployment process. This ensures only the currently available static assets are added, removing stale files. This clearing step should not be done within the project itself; it is something you perform in your deployment environment. Tools such as Ansible, Fabric, or specific deployment scripts can handle this effectively. A basic example might involve a simple `rm -rf /var/www/myproject/static/*` before calling `python manage.py collectstatic`.

Another technique involves using a different `STATIC_ROOT` directory for development versus production. This can help isolate issues introduced during development that may carry over. In my development workflow, the `STATIC_ROOT` can point to the `static` folder within the project, so changes are immediately reflected. In the production deployment workflow, it would point to the desired production directory. This keeps the environments distinct.

Also, carefully reviewing `STATICFILES_DIRS` and ensuring no extraneous or deprecated locations are included is essential. Over time, static file configurations may get added, and may no longer be necessary. Maintaining a clean and minimal `STATICFILES_DIRS` configuration can mitigate this issue. Furthermore, using versioned static files, often accomplished through tools like Django Compressor, can also alleviate the problem by ensuring browsers and CDNs request the latest version of the files.

For more in-depth information, I recommend consulting the official Django documentation regarding static file handling, especially sections on `STATIC_ROOT`, `STATIC_URL`, `STATICFILES_DIRS`, and `STATICFILES_FINDERS`. Additionally, exploring documentation on using versioned static files in Django and deployment patterns involving static files is beneficial. This understanding, combined with careful file management, should mitigate the re-inclusion of deleted static assets. Resources on deployment automation using Ansible or similar tools also offer guidance in automating the pre-collectstatic cleanup process. These sources have proven invaluable in my work with Wagtail deployments. Understanding that `collectstatic` is an aggregator, and thus needs the `STATIC_ROOT` to be pre-emptively cleaned is key to avoiding this problem.
