---
title: "How does processing multiple targets from a single source impact Nant's processing time?"
date: "2025-01-30"
id: "how-does-processing-multiple-targets-from-a-single"
---
The efficiency of Nant, a .NET build tool, is significantly influenced by how it handles multiple target dependencies originating from a single source file or set of files. Specifically, processing multiple targets that share the same input often leads to redundant processing if not managed carefully, primarily because Nant’s default execution is target-oriented rather than dependency-oriented. This means that for each target which lists that source, Nant potentially rebuilds the input every time. This behavior is not unique to Nant; it is a common issue in many build systems.

I’ve encountered this exact problem several times in my years developing build processes, most recently while streamlining our deployment process for a large microservices project. The common mistake occurs when build files are structured with multiple independent targets, each performing a separate action (compiling different project components, executing tests, or generating specific outputs), but all deriving from the same initial source (e.g., source code). In this situation, if the first target which uses the source has not completed, each successive target attempts to build the source again. If the build process is lengthy or the source compilation is resource intensive, the resulting increase in build time becomes substantial and unacceptable for any large-scale project.

The core of the problem lies in the lack of inherent dependency tracking at the resource level in standard Nant usage. Nant primarily understands dependencies at the target level, meaning if a target ‘B’ depends on target ‘A’, it executes target ‘A’ first. However, it doesn't natively track which target *consumes* which *resources* and ensure they are only built once. This leads to scenarios where the same source code is repeatedly compiled, or the same data is repeatedly processed as each target using that source requests it be rebuilt.

To manage this issue, we need to explicitly introduce dependency management of resources by implementing strategies that prevent this redundant processing. This typically involves two approaches: using the `<property>` task to store the output path of an intermediate build step and using the `<if>` or `<unless>` tasks in conjunction with file or directory existence checks to determine whether a particular operation has already been performed. Alternatively, if only certain files have changed, it can sometimes be more efficient to perform incremental builds using a combination of timestamps and appropriate compilers.

Let’s illustrate this with code examples. The first scenario, an inefficient implementation, showcases the problem:

```xml
<project name="MultipleTargets" default="buildall">

  <target name="compile_project1">
    <echo message="Compiling Project 1"/>
    <exec program="csc.exe" commandline="/target:library /out:project1.dll src/project1.cs"/>
  </target>

  <target name="compile_project2">
   <echo message="Compiling Project 2"/>
    <exec program="csc.exe" commandline="/target:library /out:project2.dll src/project2.cs"/>
   </target>

  <target name="run_tests1" depends="compile_project1">
    <echo message="Running tests for Project 1"/>
    <exec program="nunit-console.exe" commandline="project1.dll" />
  </target>

  <target name="run_tests2" depends="compile_project2">
    <echo message="Running tests for Project 2"/>
    <exec program="nunit-console.exe" commandline="project2.dll" />
  </target>


  <target name="buildall" depends="run_tests1,run_tests2" />

</project>
```

In this simplistic example, `compile_project1` and `compile_project2` each compiles its respective project. `run_tests1` and `run_tests2` depend on these targets and run corresponding unit tests. If all targets need to execute, which would be the case in a clean build, the compiler would run twice and we are assuming here that compilation is more than just instantaneous, it could involve resource-intensive tasks. When we run the `buildall` target, this naive implementation will lead to `project1.dll` being compiled regardless of whether project 2's compilation or project 1’s tests have been performed. This is because Nant's dependence logic works on targets not specific resources or file inputs.

The second example demonstrates a better approach using property storage and file existence checks:

```xml
<project name="MultipleTargets" default="buildall">

  <property name="project1.dll" value="bin/project1.dll" />
   <property name="project2.dll" value="bin/project2.dll" />

  <target name="compile_project1" >
    <unless condition="${file::exists(project1.dll)}">
      <echo message="Compiling Project 1"/>
      <mkdir dir="bin" />
      <exec program="csc.exe" commandline="/target:library /out:${project1.dll} src/project1.cs"/>
    </unless>
     <echo message="Project 1 Already Built" />
  </target>

  <target name="compile_project2" >
      <unless condition="${file::exists(project2.dll)}">
          <echo message="Compiling Project 2"/>
           <mkdir dir="bin" />
        <exec program="csc.exe" commandline="/target:library /out:${project2.dll} src/project2.cs"/>
      </unless>
        <echo message="Project 2 Already Built" />
  </target>

   <target name="run_tests1" depends="compile_project1">
    <echo message="Running tests for Project 1"/>
    <exec program="nunit-console.exe" commandline="${project1.dll}" />
  </target>

  <target name="run_tests2" depends="compile_project2">
    <echo message="Running tests for Project 2"/>
    <exec program="nunit-console.exe" commandline="${project2.dll}" />
  </target>


  <target name="buildall" depends="run_tests1,run_tests2" />

</project>
```

Here, we use `<property>` to store the paths of the build artifacts, and we use `unless` in conjunction with the `file::exists()` function to only execute the compile action if the artifact doesn’t exist already. Therefore, if subsequent targets execute, and they are all dependent on the same build source, the compiler is only executed once. Note that this relies on us having a deterministic output path for our dll files, and creating the `bin` output directory first, if it doesn’t exist. The `<mkdir>` task handles this step. This significantly reduces compilation time in repeated build cycles.

Finally, a third more complex example utilizes a timestamp approach. This is useful when only certain files have changed, which is often the case when changes to source files occur over time. This approach would require more complex logic and may not be required unless the previous method does not address a specific need. This is particularly important when dealing with very large projects where recompiling everything is costly:

```xml
<project name="IncrementalBuild" default="buildall">

  <property name="project1.dll" value="bin/project1.dll" />
  <property name="project1.src" value="src/project1.cs" />

 <target name="compile_project1" >
    <if test="${file::exists(project1.dll) and file::getlastwritetime(project1.dll) > file::getlastwritetime(project1.src)}">
        <echo message="Project 1 Up-to-Date" />
    </if>
     <else>
      <echo message="Compiling Project 1"/>
       <mkdir dir="bin" />
      <exec program="csc.exe" commandline="/target:library /out:${project1.dll} ${project1.src}" />
      </else>
  </target>

    <target name="run_tests1" depends="compile_project1">
    <echo message="Running tests for Project 1"/>
    <exec program="nunit-console.exe" commandline="${project1.dll}" />
  </target>


  <target name="buildall" depends="run_tests1" />
</project>
```

In this example, we are comparing the last modified time of the source file and of the resulting compiled library. If the compilation artifact’s timestamp is later than the source timestamp, we assume the compilation is up to date. This is a more granular dependency management than merely checking existence. While more verbose, this mechanism is a step closer to what modern build tools can do with internal build graphs. Note that this will have different behavior if you delete the output artifacts yourself, whereas the previous method won’t.

Several excellent resources provide further information on build system optimization and best practices. “Continuous Delivery” by Jez Humble and David Farley details how to structure efficient delivery pipelines. Although not Nant-specific, the principles are universal.  “Working Effectively with Legacy Code” by Michael Feathers, while primarily addressing unit testing, also provides valuable insight into structuring code in a manner that is more easily processed by build systems and promotes more effective incremental compilation. Finally, any good book on software engineering, such as "Code Complete" by Steve McConnell, will cover the basics of building code effectively and the pitfalls to watch out for when designing a build process. These sources should provide a more comprehensive understanding of the principles involved and allow for a deeper exploration into best practices.
