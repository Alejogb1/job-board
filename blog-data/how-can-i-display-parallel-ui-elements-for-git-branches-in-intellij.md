---
title: "How can I display parallel UI elements for Git branches in IntelliJ?"
date: "2024-12-23"
id: "how-can-i-display-parallel-ui-elements-for-git-branches-in-intellij"
---

Alright,  I remember back when my team was migrating to a heavily branch-based workflow – it became a real challenge keeping track of everything visually within IntelliJ. Displaying parallel UI elements for git branches isn't a default configuration, so it requires a bit of setup and understanding of the available tools. We stumbled around for a bit, but eventually, we landed on a setup that worked quite effectively for us. The key here isn't one single magic setting but a combination of configuration and intelligent plugin usage. I'll walk you through how we solved this, starting with the basics and then touching on some more advanced options.

First off, understand that IntelliJ's primary Git integration centers around displaying the currently checked-out branch in the bottom right corner, along with the ability to switch between branches through that dropdown. That's fine for simple workflows, but not nearly sufficient when you need to visualize the broader context of multiple branches simultaneously. What we're after is a method to see branches, their commit histories, and their relationships side-by-side.

The initial setup, which is often overlooked, lies in judicious use of the *Version Control Tool Window*. It's not enough to just glance at the bottom corner. We need to actively use this window, typically accessible via alt + 9 (or cmd + 9 on MacOS). In the 'Log' view within this window, you'll see your commit history. But here's the crucial part: you can filter this log by branch. So, rather than just seeing everything jumbled together, you can select a particular branch from the dropdown near the filter input and see only its commits. This allows you to compare commit timelines from different branches when you manually alternate. While it doesn't display them all simultaneously, it’s the cornerstone for understanding what’s happening on each branch. It's critical to use this to get a good grasp of the commit structure of each branch before exploring visualisations, so let’s explore how to filter these logs with examples first.

Here's a basic snippet showcasing how you might programmatically check for the currently active branch and then use this for filtering in the Version Control Tool window. While this doesn't directly *display* parallel branches, it highlights the underpinnings of how IntelliJ and Git interplay:

```java
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vcs.changes.ChangeListManager;
import com.intellij.openapi.vfs.VirtualFile;
import git4idea.GitUtil;
import git4idea.commands.Git;
import git4idea.repo.GitRepository;
import git4idea.repo.GitRepositoryManager;
import java.util.Optional;

public class GitBranchUtility {

    public static Optional<String> getCurrentBranchName(Project project) {
        VirtualFile projectDir = project.getBaseDir();
        GitRepositoryManager repositoryManager = GitRepositoryManager.getInstance(project);
        GitRepository repository = repositoryManager.getRepositoryForFile(projectDir);

        if (repository == null) {
            return Optional.empty(); // No git repository found
        }

        return Optional.ofNullable(repository.getCurrentBranchName());
    }


    public static void displayBranchInLog(Project project, String branchName) {
        // In reality, you'd need to trigger the log view manually
        // or programmatically select a specific branch filter.
        // This is a *conceptual* representation, as the direct manipulation
        // of UI elements would require more low-level introspection which isn't
        // generally recommended and can break on platform upgrades.
        System.out.println("Simulating filtering the Version Control Log for branch: " + branchName);
    }

    public static void main(String[] args) {
        // The Project object is a dependency injection or IDE resource in context of a plugin.
        // For simulation, assume project object is available and you have the correct VCS setup.
        Project project = null; // Typically obtained in IntelliJ plugin context
        Optional<String> currentBranch = getCurrentBranchName(project);
        currentBranch.ifPresent(branch -> displayBranchInLog(project, branch));

    }

}

```

This example isn't going to magically change your IntelliJ UI, but it helps illustrate a critical point: IntelliJ *knows* your current branch, and you can programmatically access and utilise this information, though generally not for direct UI manipulation. The `displayBranchInLog` is a conceptual step. IntelliJ uses this data internally to filter the commit log, which is what you would interact with through the `Version Control` panel.

The second technique, and where we started to see real gains, is using the *Git Branch Graph* provided by an IntelliJ plugin. I suggest looking into plugins like "Git Graph" or something similar from the plugin marketplace. They're generally well-maintained, and it is essential to understand that third party tools can, from time to time, break during IntelliJ version updates. You can achieve similar results without plugins, but their convenience outweighs the complexity. What these plugins often do is visualize the branch structures and commit relationships into a graph view, much like gitk or other external graph tools. With this, you are seeing different branches in a visual timeline and can understand how they relate to each other. The advantage here is immediate visual parallel representation of branches.

Here’s a conceptual snippet of what the core logic within such a plugin might look like—again, *simplified* and not directly runnable, but it’s a good analogy for what these tools do under the hood:

```python
import networkx as nx
import matplotlib.pyplot as plt
import subprocess
import re

def get_git_log(branch):
    try:
        result = subprocess.run(['git', 'log', '--pretty=format:%h %P %ad %s', '--date=short', branch], capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error running git log: {e}")
        return []


def parse_git_log(log_lines):
    commits = []
    for line in log_lines:
        if line:
            parts = re.split(r'\s+', line, maxsplit=4)
            if len(parts) == 4 or len(parts) == 5:
               hash = parts[0]
               parents = parts[1].split() if len(parts)>4 else []
               date= parts[-2]
               message=parts[-1]
               commits.append({'hash':hash, 'parents':parents, 'date': date, 'message': message})
    return commits


def create_commit_graph(branches):
  graph = nx.DiGraph()
  for branch in branches:
    log = get_git_log(branch)
    commits = parse_git_log(log)
    for commit in commits:
        graph.add_node(commit['hash'], message=commit['message'], date= commit['date'], branch = branch)
        for parent in commit['parents']:
            if parent:
                graph.add_edge(parent,commit['hash'])
  return graph;

def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=8, edge_color='gray', arrowsize=10)
    labels = nx.get_node_attributes(graph, 'message')
    nx.draw_networkx_labels(graph, pos, labels, font_size=6)
    plt.show()

if __name__ == "__main__":
  branches_to_display = ['main', 'develop', 'feature/my-new-feature']
  git_graph = create_commit_graph(branches_to_display)
  visualize_graph(git_graph)
```

This Python example shows how a simplified git commit graph could be built. In the real world, of course, the plugins are much more performant, work in native Java, and are integrated into the IntelliJ platform. It captures the key aspect: fetching the commit logs, parsing the tree, and displaying the relationships visually.

Finally, consider utilizing multiple IntelliJ windows or splits. It sounds simple, but opening separate project windows or splitting your view to show multiple instances of the 'Version Control' log view, with each showing different branches filtered, can achieve a form of parallel display. It's a bit more manual to set up, but it avoids reliance on plugins and offers a granular view if that suits your workflow. This may be less efficient for complex visualisations compared to the graph view, but it's a valid approach that requires no external tooling. Here’s a small java snippet on how we could conceptually achieve multiple log views, again, without attempting to modify the actual IntelliJ UI, as direct manipulation isn’t encouraged:

```java
import java.util.ArrayList;
import java.util.List;

public class MultipleLogView {

    public static void simulateLogDisplay(List<String> branches) {
       //  In a plugin this would involve creating multiple instances
       // of the commit log display and filtering appropriately, which is far beyond
       // the scope of this simple java snippet.

        for (String branch : branches) {
            System.out.println("Simulating Log Display for Branch: " + branch);
            System.out.println("------------------------------------");
            // In reality, the commit log for this branch would be displayed
            // In a window/tab in IntelliJ.
        }
    }
    public static void main(String[] args) {
      List<String> branches = new ArrayList<>();
      branches.add("main");
      branches.add("develop");
      branches.add("feature/my-new-feature");
      simulateLogDisplay(branches);
    }
}
```

This example indicates how one might approach the idea of multiple windows, with each providing a filter on a different branch, simulating what one might achieve manually with the IntelliJ interface or through a more complex plugin.

In conclusion, there’s no single “display parallel branches” button within IntelliJ. It requires a thoughtful combination of tools: actively using the `Version Control` log window, installing a branch graph visualization plugin, and optionally utilizing multiple IntelliJ window views. This multi-faceted approach, backed by an understanding of how git data is accessed by IntelliJ, has served me well in the past for dealing with complex, branch-heavy projects. For further reading, I’d recommend looking into "Pro Git" by Scott Chacon and Ben Straub for understanding Git internals and how those are exposed programmatically. Furthermore, exploring the IntelliJ Platform SDK documentation will give you an insight into plugin development, should you want to create a custom solution. Good luck!
