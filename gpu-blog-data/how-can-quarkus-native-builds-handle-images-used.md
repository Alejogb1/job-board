---
title: "How can Quarkus native builds handle images used in Qute templates?"
date: "2025-01-30"
id: "how-can-quarkus-native-builds-handle-images-used"
---
Quarkus native image compilation, relying on GraalVM, presents a unique challenge when dealing with resources referenced in Qute templates, specifically images. The default behavior of GraalVM doesn't automatically package these resources unless explicitly instructed, leading to runtime errors where images are not found. This necessitates a specific approach to ensure images embedded in Qute templates are properly incorporated into the native executable.

The core issue arises from GraalVM’s static analysis. During native image generation, it attempts to determine which resources are required by the application at runtime. Dynamic resource loading, frequently used in web application frameworks such as Quarkus, can mislead GraalVM if it isn't explicitly told about these assets. Consequently, images referenced within Qute templates, which are often loaded via dynamic path resolution, may be overlooked by the native image building process, resulting in a binary lacking the necessary image files. The `io.quarkus.runtime.configuration.ResourceLocation` class is involved in this dynamic resource discovery, and therefore, becomes relevant to our solution. The resolution often entails configuring resource inclusion patterns for the build.

To address this, we must inform GraalVM about the location of our image resources. I’ve found two effective methods, each with distinct advantages and use cases: explicit inclusion via the `application.properties` file, and using a programmatic `BuildStep` to register resources. My experience has shown that the former is suitable for common project layouts, while the latter provides finer-grained control, particularly in complex applications or plugin development.

First, let's examine explicit inclusion using the `application.properties` file. This approach utilizes the `quarkus.native.resources.includes` property to define glob-style patterns for inclusion of the relevant files. I often use this approach for projects following the default Maven or Gradle layout, where images are generally within the `src/main/resources` directory, or a subfolder within that directory. I might have an `images` folder containing my visual assets within `src/main/resources`. To include all images, regardless of the filename extension (e.g., `.png`, `.jpg`, `.svg`), the following `application.properties` entry would suffice:

```properties
quarkus.native.resources.includes=images/**/*
```

This configuration instructs GraalVM to include all files and subdirectories under `src/main/resources/images` within the native image. If your images are in a more specific folder within your resources directory, update the pattern to reflect the accurate path. The `*` acts as a wildcard for any character, and `**` matches any subdirectory, providing the flexibility I need for different project arrangements. This is often the quickest and most straightforward way to handle image resources in basic projects. The following code snippet provides a Qute template that renders an image that is available via the `src/main/resources/images/my_image.png` file:

```html
<!-- src/main/resources/templates/image.html -->
<html>
 <head>
  <title>Image Template</title>
 </head>
 <body>
  <img src="/images/my_image.png" alt="My Image" />
 </body>
</html>
```

And a simple Quarkus resource class (e.g., `ImageResource.java`) might look like this, showing how this template can be returned to a client request:

```java
import io.quarkus.qute.Template;
import io.quarkus.qute.TemplateInstance;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

@Path("/image")
public class ImageResource {

    @Inject
    Template image;

    @GET
    @Produces(MediaType.TEXT_HTML)
    public TemplateInstance get() {
        return image.instance();
    }
}
```

In my experience, once the `application.properties` configuration is in place and a native executable is built, the image will be displayed correctly within the Qute template at runtime.

The second method involves using a programmatic `BuildStep`, a more complex solution suited to scenarios demanding a higher degree of control. I've used this method when I need to dynamically determine what resources should be included based on application logic, or when handling custom resource locations that fall outside the default `src/main/resources` folder.  For instance, if the location of the images is determined dynamically during runtime, or when working with complex dependencies where it is difficult to specify the resources with wildcard patterns. To achieve this, you must define a Quarkus extension or custom `BuildStep`. For this method, one would create a Java class annotated with `@BuildStep` that has methods to perform the resource registration within the build lifecycle.

Below is a code sample showing how to programmatically register a resource using `io.quarkus.deployment.builditem.nativeimage.NativeImageResourceBuildItem`. The `BuildStep` method below adds resources relative to the `src/main/resources` folder.

```java
import io.quarkus.deployment.annotations.BuildStep;
import io.quarkus.deployment.builditem.nativeimage.NativeImageResourceBuildItem;
import java.util.Arrays;
import java.util.List;

public class ImageResourceProcessor {

    @BuildStep
    NativeImageResourceBuildItem registerImages() {
        List<String> resources = Arrays.asList(
                "images/my_image.png",
                "images/another_image.jpg"
        );
        return new NativeImageResourceBuildItem(resources);
    }
}
```

This `BuildStep` explicitly registers "images/my_image.png" and "images/another_image.jpg" for inclusion in the native image, bypassing the need for wildcard expressions within `application.properties`. This approach proves invaluable when dealing with resources that do not conform to a simple path or naming convention and allows for dynamic inclusion based on the state of application or user configuration. The method `registerImages` will be called during the build process, and the declared files included into the native image. To enable the BuildStep, an implementation of a Quarkus extension should be configured in the project, usually by adding a file under `src/main/resources/META-INF/services` named `io.quarkus.deployment.spi.builditem.BuildStep` and entering the full name of the processor class on a new line (e.g. `com.example.ImageResourceProcessor`).

The final code example uses a different approach within the BuildStep to resolve resources. This is useful if you store your resources in other folders, such as within the same package as your other Java source files:

```java
import io.quarkus.deployment.annotations.BuildStep;
import io.quarkus.deployment.builditem.nativeimage.NativeImageResourceBuildItem;
import io.quarkus.deployment.pkg.steps.NativeBuild;
import org.jboss.jandex.Index;
import org.jboss.jandex.Indexer;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class ImageResourceProcessor {

    @BuildStep
    NativeImageResourceBuildItem registerClasspathImages(NativeBuild.NativeBuildEnabledBuildItem nativeBuildEnabled) throws IOException {
        if (!nativeBuildEnabled.isNativeBuild()) {
          return null;
        }
        List<String> resources = new ArrayList<>();
        Indexer indexer = new Indexer();

        try (InputStream resource = ImageResourceProcessor.class.getResourceAsStream("/templates/images/my_image.png")) {
            if (resource != null) {
                resources.add("templates/images/my_image.png");
            }
        } catch (IOException e) {
            throw new IOException("Unable to read resource", e);
        }


        try (InputStream resource = ImageResourceProcessor.class.getResourceAsStream("/templates/images/another_image.jpg")) {
            if (resource != null) {
                resources.add("templates/images/another_image.jpg");
            }
        } catch (IOException e) {
           throw new IOException("Unable to read resource", e);
        }
      return new NativeImageResourceBuildItem(resources);
    }
}

```

This approach employs the `getResourceAsStream` method of the `Class` object to load the resources. The method checks that the provided path exists and if it does the resource is then added to the list to be processed and included in the final native executable. In this case, it demonstrates the loading of images located within the `templates/images` folder inside the same Java package as the class, in the same way Java application loads resources such as configuration files.

In conclusion, I have found the combination of declarative configurations and programmatic resource registration provides an effective solution for managing image resources within Quarkus native image builds. For typical projects, the `quarkus.native.resources.includes` property in the `application.properties` will likely be sufficient. For more complex scenarios, or dynamic resource registration, utilizing a `BuildStep` proves to be the better approach. I recommend consulting the official Quarkus documentation sections regarding native image building and resource handling to gain an even more comprehensive understanding of the underlying mechanics. Specifically the Quarkus Native guide, and the section on build time extensions are essential reference points when you need additional information on native image generation and building Quarkus extensions.
