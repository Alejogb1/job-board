---
title: "How can I list MODX articles sharing a common tag?"
date: "2024-12-23"
id: "how-can-i-list-modx-articles-sharing-a-common-tag"
---

Let's dive straight into this, shall we? I recall a particularly tricky project back in 2017. We were building a knowledge base for a medical device company using MODX Revolution, and they absolutely needed a robust way to filter articles by topic – essentially, tags. It was surprisingly complex to get it just *right*, avoiding performance bottlenecks and maintaining a clean user experience. Listing articles by a common tag in MODX, while seemingly straightforward, actually requires a thoughtful approach to get it working effectively, especially when you start dealing with larger datasets.

The core challenge here is efficiently querying the MODX database, given that relationships between resources (articles) and tags aren't natively stored in a direct manner. Instead, tags are usually managed through a third-party extra or custom implementation. Let's assume, for the sake of clarity, that you're using a common approach: a custom template variable (TV) to store the comma-separated list of tags for each article. While not the most elegant solution – dedicated tagging extras like TagLister or more recent ones, which I'll recommend later, offer better performance and management features – it's quite often encountered, so we'll begin there.

The basic idea is to fetch all resources, parse their tag TV, and then filter the list down to those containing the tag of interest. Now, this can quickly become incredibly slow and inefficient, particularly on large MODX sites. Doing this directly within a standard snippet using getResources or pdoResources will result in significant performance issues. So, we'll focus on methods that minimize this performance hit.

Here's how I tackled it, and here’s how I recommend you approach it:

**Step 1: Utilizing a Basic Snippet for Tag Matching (Inefficient, but illustrative)**

Let's start with a simplistic approach, mostly for demonstration. It will highlight the problems of brute-force filtering, but is useful to understand the basic mechanics. This snippet uses a pdoResources call, fetches all resources, and filters client-side within the php snippet:

```php
<?php
$tagToMatch = $modx->getOption('tag', $scriptProperties, '');
if (empty($tagToMatch)) return '';

$resources = $modx->runSnippet('pdoResources', [
    'parents' => '0', // Assuming top-level articles
    'includeTVs' => 'your_tag_tv_name', // Replace 'your_tag_tv_name' with your actual TV name
    'limit' => '0', // Get all resources
    'processTVs' => '1', // Ensure TVs are processed.
    'tpl' => '@INLINE [[+pagetitle]]<br>',
    'outputSeparator' => '' // Prevents extra separator between resource entries
]);

if(empty($resources)) return 'No articles found.';
$output = '';
foreach (explode('<br>', $resources) as $line) {
    if(!empty($line)) {
        $resource = $modx->getObject('modResource', ['pagetitle' => trim($line)]);
    if($resource){
     $tags = isset($resource->get('tv.your_tag_tv_name')) ? explode(',', $resource->get('tv.your_tag_tv_name')) : []; //fetch and explode tags from tv
     if (in_array($tagToMatch, array_map('trim', $tags))) {
         $output .= $line . "<br>\n"; // Show the resource title if the tag matches
    }
    }

    }
}

return $output;
```
This is essentially iterating through every resource, grabbing its tag TV, and checking for a match. This **won't scale** and will perform poorly even on mid-sized sites.

**Step 2: Enhanced Snippet Using a where Clause with LIKE for better efficiency**

The next step is to improve the search within the database level using a `where` clause. While still not perfect, it's a much better starting point for larger datasets than processing every record in PHP. This relies on the database to perform the filtering, which will often be considerably faster.

```php
<?php
$tagToMatch = $modx->getOption('tag', $scriptProperties, '');
if (empty($tagToMatch)) return '';


$output = $modx->runSnippet('pdoResources', [
    'parents' => '0', // Assuming top-level articles
    'includeTVs' => 'your_tag_tv_name', // Replace 'your_tag_tv_name' with your actual TV name
    'limit' => '0', // Get all resources
    'processTVs' => '1', // Ensure TVs are processed.
        'where' => ["`tv.your_tag_tv_name` LIKE '%". trim($tagToMatch) . "%'"],
        'tpl' => '@INLINE [[+pagetitle]]<br>', // Use an inline tpl for a quick return
        'outputSeparator' => '' // Prevents extra separator between resource entries
]);

if(empty($output)) return 'No articles found';

return $output;
```
This snippet now utilizes a `LIKE` clause in the `where` parameter, which is passed down to the database to efficiently retrieve matching resources. This is a substantial improvement over the previous PHP-based filter and much faster, especially when many resources are involved. The use of trim helps with matching even if spaces exist within the tv string values.

**Step 3: Leveraging a Dedicated Tagging Extra (The Recommended Approach)**

Frankly, the previous approaches, although more performant than the first, are suboptimal. The ideal approach is to utilize a dedicated tagging extra, which will handle the relationship between tags and articles at the database level efficiently. The best solution will also manage tag hierarchies and relationships more effectively, and will not cause the performance issues of a custom tv approach.

I've found that TagLister (a slightly older but very robust extra) and the newer, more actively developed *taggit* extra, are excellent options. I'll focus on describing the basic concept with taggit, as it's the more actively supported of the two. These extras typically create a dedicated database table that stores the associations between resources and tags, allowing for efficient and direct database queries that would not be possible with the custom tv approach. You would interact with *taggit* through its own API. Here is a conceptual way you might use a call via a snippet to get tagged results:

```php
<?php
$tagToMatch = $modx->getOption('tag', $scriptProperties, '');
if (empty($tagToMatch)) return '';

$taggit = $modx->getService('taggit', 'taggit', MODX_CORE_PATH . 'components/taggit/model/');

if (!($taggit instanceof taggit)) {
    return 'taggit component not installed';
}

$matchingResources = $taggit->getArticlesByTags(trim($tagToMatch));

if (empty($matchingResources)) {
  return 'No resources found with that tag.';
}

$output = '';

foreach ($matchingResources as $resource) {
    $output .= $resource->get('pagetitle') . '<br>';
}

return $output;
```
In this approach, `taggit` manages tag associations in a database, and the `getArticlesByTags` method leverages these associations directly, optimizing database queries significantly. It's a much cleaner, more efficient method that does not rely on inefficient LIKE clauses, and is much better for a large number of resources.

**Key Points to Consider**

*   **Database Optimization:** While using a LIKE clause in a `where` clause is an improvement over client-side PHP filtering, it still can lead to performance issues. A dedicated tagging extra is much more efficient.
*   **Caching:** Whichever approach you use, implementing caching of the results, even for short durations, will vastly improve the user experience. MODX's caching system can help.
*   **User Experience:** Ensure your output formats tags and links cleanly. The snippet output examples provided are barebones. In production, you should use template chunks and properly formatted output.
*   **Documentation:** Please reference the documentation for the specific tag extra that you decide to use. They are often the most detailed and current.
*   **Modularity:** Aim for well-structured code that's easily maintainable. Using multiple snippets for different parts of the process is generally helpful.

**Recommended Resources**

*   **The MODX documentation itself** should be your first port of call. The section on 'extras' and on custom template variables is fundamental.
*   **MODX Community forums and Slack channel:** These resources provide great peer-to-peer support and examples of real-world implementations.
*   For a deeper understanding of database querying and optimization, I recommend the book "High Performance MySQL" by Baron Schwartz, Peter Zaitsev, and Vadim Tkachenko.
*   The taggit's github page will also provide additional details specific to the plugin.

In closing, while it's possible to get articles by tag without an external extra, for any site of reasonable scale, a dedicated tagging extra such as TagLister or *taggit* will be essential. Remember to test thoroughly and optimize your code for better performance, and also always reference the official documentation for the extra you are using. The examples provided here are illustrative, but hopefully they offer a good starting point for creating robust and performant tag listings within MODX.
