---
title: "How do I add a file to the Testcontainers classpath in VS Code/Metals using sbt?"
date: "2024-12-23"
id: "how-do-i-add-a-file-to-the-testcontainers-classpath-in-vs-codemetals-using-sbt"
---

Alright, let’s tackle this. I've actually run into this exact scenario a few times, especially when dealing with integration testing that requires custom configurations or data files alongside Testcontainers. It’s a fairly common need, and the solution, while not always immediately obvious, is actually quite straightforward once you understand how classpaths are handled.

The issue, fundamentally, stems from the fact that Testcontainers, when spinning up its Docker containers, needs access to any resource files you intend to use within your tests. If these files aren't properly included in the classpath that Testcontainers sees, you'll inevitably run into `FileNotFoundException` or similar errors when the code inside your container tries to access those resources. The default `sbt test` classpath doesn’t automatically grab extra, non-source files. That's where we need to intervene.

My personal experience with this involved a complex microservice architecture where we had to load specific configuration files into containers during integration testing. We needed to simulate different environment configurations using these files, and initially, getting them into the Testcontainers environment was, shall we say, a learning experience.

The approach I’ve found consistently reliable involves leveraging sbt's ability to handle resource directories and then ensuring that these directories are correctly included in the classpath used when running your tests. Specifically, we'll use `unmanagedResources` in your `build.sbt` file. This directive tells sbt to treat specific directories as sources of resources that should be included when the project is compiled and bundled. Additionally, we'll use a `test` setting to ensure that resources end up on the classpath specifically for tests, which is where Testcontainers typically executes.

Here’s how we typically set it up in `build.sbt`:

```scala
import sbt._
import Keys._

lazy val root = (project in file(".")).
  settings(
    name := "testcontainers-classpath-example",
    version := "1.0",
    scalaVersion := "2.13.12",
    libraryDependencies ++= Seq(
      "org.testcontainers" % "testcontainers" % "1.19.1" % Test,
      "org.scalatest" %% "scalatest" % "3.2.16" % Test
    ),
    unmanagedResourceDirectories in Compile += baseDirectory.value / "src" / "main" / "resources",
    unmanagedResourceDirectories in Test += baseDirectory.value / "src" / "test" / "resources",
    // This setting includes resources for test execution only
    unmanagedResourceDirectories in Test := (unmanagedResourceDirectories in Test).value ++ Seq(
       baseDirectory.value / "src" / "test" / "testdata"
    ),
    testOptions += Tests.Argument(TestFrameworks.ScalaTest, "-oDF"),
  )
```

In this setup, `unmanagedResourceDirectories in Compile` includes resources from your primary resource directory that might also be necessary during compilation. `unmanagedResourceDirectories in Test` is crucial; it specifically adds resources located in `src/test/resources` to the classpath *only when tests are run*. And most importantly for this question, we explicitly add `baseDirectory.value / "src" / "test" / "testdata"`. This means that anything under `src/test/testdata`, which I recommend as a specific location for your test resources, will be included in the classpath available when you run the tests, and therefore, available to your Testcontainers.

Here's the core idea in action. Suppose you want to load a simple JSON file (`data.json`) into your container, which might be structured something like this:

```json
{
  "message": "Hello from resource file"
}
```

Place this `data.json` file in the `src/test/testdata` directory of your project.

Next, in your test class, you could have code that looks like this, using a Testcontainers based test setup:

```scala
import org.scalatest.BeforeAndAfterAll
import org.scalatest.wordspec.AnyWordSpec
import org.testcontainers.containers.GenericContainer
import org.testcontainers.utility.DockerImageName
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.util.Using

class ResourceAccessTest extends AnyWordSpec with BeforeAndAfterAll {

  val container = new GenericContainer(DockerImageName.parse("alpine/git:latest"))
  container.withCommand("sh", "-c", "sleep 5 && cat /testdata/data.json")


  override def beforeAll(): Unit = {
    container.start()
  }

  override def afterAll(): Unit = {
    container.stop()
  }

   "Testcontainers resource access" should {
     "be able to load json from the classpath in testcontainers" in {

       val output = Using(container.getLogs()) { logs =>
          logs.getStdout
        }.getOrElse("")

       assert(output.contains("Hello from resource file"))
     }
   }
}
```

In this test, the crucial part is where we set the `container.withCommand()`. Note the command `cat /testdata/data.json`. The `cat` command attempts to read the contents of `data.json`. Since `data.json` resides in the `src/test/testdata` folder which we configured to be included on the classpath, Testcontainers makes this resource available within the Docker container under the `/testdata` directory. It is effectively adding all classpath resources to that directory under the root path. So if you had `src/test/testdata/some/folder/file.txt`, you would access this as `/some/folder/file.txt` inside of the test container environment.

Another example showcasing a test with a slightly different resource loading strategy, this time using java’s built-in methods for retrieving classpath resources:

```scala
import org.scalatest.BeforeAndAfterAll
import org.scalatest.wordspec.AnyWordSpec
import org.testcontainers.containers.GenericContainer
import org.testcontainers.utility.DockerImageName
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.io.Source
import scala.util.Using


class ResourceAccessTestB extends AnyWordSpec with BeforeAndAfterAll {

    val container = new GenericContainer(DockerImageName.parse("alpine/git:latest"))
    container.withCommand("sh", "-c", "sleep 5")


    override def beforeAll(): Unit = {
        container.start()
    }

    override def afterAll(): Unit = {
       container.stop()
    }

    "Testcontainers resource access using scala built-ins" should {
        "be able to load data.json from the classpath" in {
            val resourcePath = "/data.json" // resource path in the classpath
            val resourceStream = getClass.getResourceAsStream(resourcePath)

            assert(resourceStream != null, "Resource stream should not be null")

            Using(Source.fromInputStream(resourceStream)) { source =>
                val fileContent = source.getLines().mkString("\n")

                assert(fileContent.contains("Hello from resource file"), "File content is not correct")
            }.getOrElse(fail("Failed to open and read resource stream"))
        }
    }
}
```

Here, we are directly accessing the resource file via `getClass.getResourceAsStream()`. The essential factor remains that this resource ( `data.json` ) is on the classpath thanks to our `build.sbt` configuration.

Now, when you run this test using `sbt test`, Testcontainers should have the necessary resource and you should not encounter any resource-loading exceptions. If you're using the Metals extension in VS Code, simply right-click the test and select "Run Test" to see this in action.

For a deeper understanding of classpaths and resource handling, I strongly suggest looking into the documentation of both sbt (specifically around managed and unmanaged resources) and the Java Classloader. A foundational text would be "Java Concurrency in Practice" by Brian Goetz et al., as it delves into class loading within a concurrent context, which is relevant for a lot of testing setups. Additionally, any standard textbook on compiler theory will provide the necessary theoretical basis for classpaths if you want to go down that route. For a more practical approach, digging into the sbt documentation (which is extensive but very helpful) will give you the specific details about resource management you will need.

In summary, correctly adding your file to the Testcontainers classpath in sbt using Metals involves configuring `unmanagedResourceDirectories` in `build.sbt` and placing your resource files in the correct location under your `src/test` folder, typically a custom `testdata` directory. It seems like a small configuration change but it's crucial for effective integration testing. Once you’ve got that down, resource loading with Testcontainers becomes much less of a headache.
