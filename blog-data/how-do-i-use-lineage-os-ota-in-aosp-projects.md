---
title: "How do I use Lineage OS OTA in AOSP projects?"
date: "2024-12-23"
id: "how-do-i-use-lineage-os-ota-in-aosp-projects"
---

Alright, let's tackle the intricacies of integrating LineageOS' over-the-air (OTA) update mechanism into an AOSP (Android Open Source Project) based project. It’s a subject I've spent quite some time with, particularly back when we were customizing a rather unique set-top box distribution a few years ago. The need for seamless updates on devices deployed to, shall we say, less than tech-savvy users quickly pushed it to the forefront. It wasn’t a walk in the park, but it certainly yielded a robust system once properly implemented.

The core idea is to leverage LineageOS' delta-based updates. Essentially, instead of downloading a complete new system image, we transmit only the *changes* between the current and target versions. This saves bandwidth and significantly reduces the time taken for an update, especially beneficial in resource-constrained environments. The whole architecture rests on a few key components: the `updater` binary, the `recovery` environment, and the update packages themselves, generally zips containing the patch information. The method isn't exactly plug-and-play and involves quite a bit of configuration. Let me walk you through the necessary steps, drawing from my prior experience.

First, we need to make sure our AOSP build includes the correct `updater` binary, which is typically located in `system/bin/updater`. This is often tied to the `recovery` environment as well, and both rely on a specific set of tools and functionalities for patching the system. Usually, if you're using a recent AOSP source, this shouldn't be a major concern, as the basic updater tools will be there, but it's best to verify. If you find yourself needing a more up-to-date version or a specific lineageOS patch, you’ll need to pull in the relevant parts from the LineageOS GitHub repos.

The real customization comes in the construction of the OTA package. The LineageOS OTA update process relies on a combination of `ota_from_target_files` (a python script) and specific configuration files located within the AOSP build tree. The script is found inside AOSP, specifically under `build/tools/releasetools/ota_from_target_files`. This tool takes two target files (produced when you build a device ROM), one for your previous and one for your new version. They contain essentially the compiled image. The script then compares them and generates a zip package with the necessary patch information.

Let's solidify this with a basic example. Assume I've successfully built two versions of my AOSP based system image: `target_files_old.zip` and `target_files_new.zip`.

Here's a snippet showing how to use `ota_from_target_files` :

```bash
#!/bin/bash

# location of the releasetools
RELEASET_TOOLS="$AOSP_ROOT/build/tools/releasetools"

# old target files zip
OLD_TARGET_ZIP="target_files_old.zip"

# new target files zip
NEW_TARGET_ZIP="target_files_new.zip"

# OTA zip output
OTA_OUTPUT="update.zip"

# call the releasetools script
python3 "${RELEASET_TOOLS}/ota_from_target_files" \
    -k test-keys  \
    "${OLD_TARGET_ZIP}" "${NEW_TARGET_ZIP}" "${OTA_OUTPUT}"
```

In this script, `AOSP_ROOT` represents your AOSP source directory path. `-k test-keys` indicates the testing keys used for signing the update package. For release builds, you’d need to use proper signing keys. `target_files_old.zip` represents your existing OS, and `target_files_new.zip` is the updated version. The result is `update.zip`, ready to deploy.

Beyond this, the `build/make/tools/releasetools/ota_from_target_files` script has many options and you'll need to use the `--block` or `-i` flag if the device doesn't support a direct upgrade. This was a crucial distinction in a system that had a non-standard partition layout, which we also faced.

Now, let's dive into how this integrates with the device. The AOSP system has a system app that checks for updates and downloads them. This is typically handled by `frameworks/base/services/java/com/android/server/pm/OtaManagerService.java`. The downloaded file is staged, and on a reboot, the device enters recovery and applies the patch. The `recovery` partition plays a very important role and has its own set of tools and configuration. The update itself is triggered through `recovery/recovery.c`. This part is generally device-specific and will require some modification to match the device's storage arrangement and firmware setup. In my experience, we needed to modify the `recovery` ramdisk to accommodate our specific partitioning and bootloader logic.

To illustrate the modification part, let's look at how you might change the updater configuration in `recovery` to support a non-standard partition layout. The below code snipped represents a highly abstracted version to demonstrate the concept, and would need to be adjusted per specific device.

```c
// This is within the recovery.c file within the recovery environment

// Simplified version; in real implementation, error checks are crucial
int install_package(const char* path) {
    // ... other logic

    // Assume the partition definition is defined in struct PartitionInfo
   struct PartitionInfo partitions[] = {
       { .name = "system", .device_path = "/dev/mmcblk0p1", .mount_point = "/system" },
       { .name = "vendor", .device_path = "/dev/mmcblk0p2", .mount_point = "/vendor" },
       { .name = "userdata", .device_path = "/dev/mmcblk0p3", .mount_point = "/data" }
       // ... Other custom partitions
   };

   int num_partitions = sizeof(partitions) / sizeof(partitions[0]);


    // Instead of the hardcoded partition name we try to look it up from defined information
    for (int i = 0; i < num_partitions; i++) {
       if (strcmp(partitions[i].name, "system") == 0) {
            // Mounts our system partition based on information found in array
           mount(partitions[i].device_path, partitions[i].mount_point, "ext4", 0, 0);
           break;
       }
   }


    // Apply update package logic
    apply_update(path);

    // Unmount the partition (simplified)
    umount("/system");

    //...
}

```
This c code snippet attempts to illustrate a crucial point; hardcoded partition locations can be brittle. Instead, you should strive to dynamically obtain this information. The `recovery` environment is a limited environment, so this process might be device specific and more involved. This is where deep knowledge of your specific platform comes into play.

Finally, the OTA system uses a manifest to describe the various components and their checksums and partition locations, usually inside the update package itself. These manifests (often in text format, e.g. `updater-script`) are crucial for the update process and have to be generated correctly using the AOSP tools.

Here's a glimpse of what an `updater-script` might look like:

```
ui_print("Starting OTA Update");
show_progress(0.1, 0);
package_extract_dir("vendor", "/vendor");
package_extract_file("system.new.dat", "/tmp/system.new.dat");
package_extract_file("system.patch.dat", "/tmp/system.patch.dat");
apply_patch("/system", "/tmp/system.new.dat", "/tmp/system.patch.dat");
show_progress(0.6, 0);
package_extract_file("vendor.new.dat", "/tmp/vendor.new.dat");
package_extract_file("vendor.patch.dat", "/tmp/vendor.patch.dat");
apply_patch("/vendor", "/tmp/vendor.new.dat", "/tmp/vendor.patch.dat");
show_progress(0.9, 0);
ui_print("Update Completed!");

```

This script is a simplified example, and in practice would contain logic to verify data integrity and device specific steps. The core point here is that these scripts, generated by `ota_from_target_files`, instruct `updater` on how to apply the updates. Notice the extraction and patch commands; this is how the diff-based upgrade is achieved.

For deeper understanding, I’d recommend examining the source code of AOSP's `build/tools/releasetools` directory, in particular, the `ota_from_target_files.py` script and the `applypatch` tool. The book “Embedded Android” by Karim Yaghmour also offers a solid theoretical foundation and practical examples on various aspects of AOSP. Lastly, the source code of LineageOS, particularly under the `device` and `system` directories, provides a practical view of how all these pieces fit together in a real-world implementation. Studying these resources in detail helped us immensely back on that set-top box project. There's no single magic wand, it's more about understanding the tools and how to effectively use them for your specific needs. It can be a complex process but one that pays off once mastered.
