---
title: "Are mermaid diagrams supported in GitHub .rst files?"
date: "2024-12-23"
id: "are-mermaid-diagrams-supported-in-github-rst-files"
---

,  It's a question I've personally encountered on several occasions, notably during a large documentation refactor we did a couple of years back. The core issue revolves around the rendering capabilities of reStructuredText (.rst) files within the GitHub ecosystem, specifically when it comes to complex diagramming tools like mermaid.

The short answer is: *natively*, no, GitHub does not directly render mermaid diagrams embedded within .rst files. GitHub's rendering pipeline primarily focuses on processing the .rst syntax itself, transforming it into HTML. While .rst supports directives and roles, it doesn't have a built-in understanding of mermaid's syntax. Trying to insert a mermaid code block directly, thinking it will just automagically work, is a common pitfall for developers unfamiliar with the intricacies of documentation pipelines.

The longer explanation involves considering what GitHub *does* support, and how we can leverage that to achieve the desired result: embedding mermaid diagrams in our documentation. GitHub primarily relies on Sphinx, a documentation generator, for rendering .rst files in its code repository's pages feature. And while Sphinx is incredibly powerful, it still requires extensions or specific configurations to understand and process mermaid code.

The challenge lies in the fact that raw .rst files displayed directly on GitHub’s interface won't render a mermaid code block. This means that while your source .rst file might *contain* the mermaid code, the rendered output, which is what users see in the repository's file browser, will treat that code as, well, code; it won't be interpreted and transformed into a visual diagram. This led to some very confusing moments during our previous project, I recall vividly. People were expecting charts, but getting verbatim mermaid code instead.

To address this, we can take one of several approaches. The most robust and maintainable method is to use Sphinx with a mermaid extension and generate the HTML using Sphinx. We can then host the HTML on the GitHub pages feature, or link them from the rendered GitHub page.

Here's how it practically plays out in a couple of approaches I've used, combined with some fictional experience from past projects.

**Approach 1: Using sphinx with the sphinxcontrib-mermaid Extension (Recommended)**

This is the more standard approach for embedding complex diagrams in a technical documentation. It requires a bit more setup but provides a more sustainable solution over the long term.

*   **Step 1: Install Sphinx and sphinxcontrib-mermaid:** This is usually done via pip.

    ```bash
    pip install sphinx sphinxcontrib-mermaid
    ```

*   **Step 2: Configure `conf.py`:** You need to enable the extension within your Sphinx configuration file.

    ```python
    # conf.py
    extensions = [
        'sphinx.ext.duration',
        'sphinx.ext.doctest',
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.intersphinx',
        'sphinx.ext.mathjax',
        'sphinx.ext.ifconfig',
        'sphinx.ext.viewcode',
        'sphinx.ext.githubpages',
        'sphinxcontrib.mermaid',
    ]
    ```

*   **Step 3: Embed Mermaid in your .rst file using a directive:** This is where you directly put the mermaid code.

    ```rst
    .. mermaid::

        graph LR
            A[Start] --> B{Decision}
            B -- Yes --> C[Process]
            B -- No --> D[End]
            C --> D
    ```

*   **Step 4: Generate HTML:** Run Sphinx to generate the documentation from your .rst files.

    ```bash
    sphinx-build -b html ./source ./build
    ```

    This creates the folder `./build/html` containing your site, which would contain fully rendered mermaid diagrams.

**Approach 2: Embedding SVG Directly (Less Ideal but Useful in Certain Cases)**

Sometimes, you might need to have a single or a small number of mermaid diagrams that, for whatever reason, you don’t want to run through a whole sphinx build process. In these less frequent situations, we can leverage the capability of mermaid to output SVGs.

*   **Step 1: Generate SVG from Mermaid code:** We’ll use the mermaid CLI tool or an online tool for this. Assuming you have the command-line tools:

    ```bash
    echo 'graph LR; A --> B;' | npx mermaid > diagram.svg
    ```

    This generates the file `diagram.svg`.

*   **Step 2: Embed SVG directly in .rst:** You will need to use an image directive here.

    ```rst
    .. image:: diagram.svg
       :alt: Alternative text for the diagram
       :align: center
       :width: 400px
    ```

*   **Step 3: Note:** the svg needs to be in the same folder as the `rst` or linked correctly relative to that location. The `diagram.svg` needs to be checked in to source control.

This technique worked reasonably well when I was handling a limited number of static process diagrams that weren’t changing too often. It's not great for more complex projects, because you need to manually generate and embed the SVGs.

**Approach 3: Javascript Embedding (Not Recommended)**

You *can* technically embed mermaid diagrams using Javascript directly in the HTML output of your Sphinx build if, for example, you do not want to depend on the mermaid sphinx extension. This approach, while technically viable, adds unnecessary complexity and is generally not recommended for maintainability reasons. You’ll need to inject Javascript and make sure your HTML file can trigger the rendering after the page loads, which can introduce synchronization issues. Also, this requires advanced knowledge of sphinx and HTML manipulation, so I am not providing code for this approach.

**Final Remarks**

To truly understand the nuances of managing complex documentation with tools like Sphinx and mermaid, I would strongly advise reading the official Sphinx documentation. It is a well-written, extremely comprehensive source of knowledge for dealing with challenges such as the one presented here. Furthermore, diving into the source code of `sphinxcontrib-mermaid` on its GitHub repository, can prove very beneficial to understand its internal mechanics if required. For a deeper understanding of graph visualization in general, “Graph Drawing: Algorithms for the Visualization of Graphs” by Giuseppe Di Battista, Peter Eades, Roberto Tamassia, and Ioannis G. Tollis, is a very good choice. It covers many concepts related to graph drawing, which helps understand how tools like mermaid function. While it doesn't deal directly with mermaid specifically, its concepts are very applicable.

So, to sum it up: no, GitHub does not *directly* process mermaid inside `.rst` files on its rendering interface. However, by utilizing the power of sphinx and relevant extensions, we can embed these complex diagrams into our documentation, ensuring they are correctly rendered. This solution offers far greater flexibility than, for example, attempting to pre-render images or injecting javascript into the HTML. It is essential to make sure your documentation setup is future proof and maintainable in the long run.
