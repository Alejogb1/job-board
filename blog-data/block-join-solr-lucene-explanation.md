---
title: "block join solr lucene explanation?"
date: "2024-12-13"
id: "block-join-solr-lucene-explanation"
---

 so you're asking about block join in Solr and Lucene eh I get it I've been there done that bought the t-shirt and probably even spilled coffee on it

Let me break this down for you from a perspective of someone who's actually wrestled with this beast not just read the documentation although you should totally do that too I'm not your mom

Block join is essentially a way to index and then query documents that have a parent-child relationship it's not about blocking some network requests or other kind of technical block its about the nested structure of your documents Think of it like family trees or product reviews belonging to a specific product that is the parent document or a whole bunch of messages all connected to one thread this is what we are talking about

Now when I first encountered this I was working on this project for a big e-commerce site they had products and each product had variations like color size etc we were trying to get fast searches for the variations based on product attributes and that's where block join saved our skin The naive approach would have been to denormalize everything into one giant document it worked but man oh man performance and storage quickly went down the drain This is the sort of situation where you know you are on a bad path and that a better solution exists you just have to find it

The core concept is simple you index parent and child documents together in the same index with a special marker saying which one is which this is all in the same index and the trick relies on lucene's ability to efficiently do queries between the block or better said the documents that are related

Lucene handles this by grouping these related documents together in an indexed fashion that allows you to query the parent based on the child attributes or the child based on parent attributes that is the magic of block join It allows you to find what you need without resorting to expensive joins after you already got the documents this is a key benefit

Let's get our hands dirty and start with a small code example

**Example 1: Indexing data**

Imagine a very basic setup a blog post which is the parent and comments which are the children we're using SolrJ a simple way to programmatically interact with Solr

```java
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.impl.HttpSolrClient;
import org.apache.solr.common.SolrInputDocument;
import java.util.Arrays;
import java.util.List;

public class BlockJoinExample {

    public static void main(String[] args) throws Exception {

        String solrUrl = "http://localhost:8983/solr/mycore";
        SolrClient solr = new HttpSolrClient.Builder(solrUrl).build();

        // Create Parent Document Post
        SolrInputDocument parentDoc = new SolrInputDocument();
        parentDoc.addField("id", "post1");
        parentDoc.addField("type_s", "post");
        parentDoc.addField("title_t", "My Awesome Blog Post");
        parentDoc.addField("content_t", "This is the content of my blog post...");

        // Create Child Documents Comments
        List<SolrInputDocument> childDocs = Arrays.asList(
                new SolrInputDocument() {{
                    addField("id", "comment1");
                    addField("type_s", "comment");
                    addField("commenter_s", "User1");
                    addField("text_t", "Great post!");
                }},
                new SolrInputDocument() {{
                    addField("id", "comment2");
                    addField("type_s", "comment");
                    addField("commenter_s", "User2");
                    addField("text_t", "I totally agree!");
                }}
        );


        List<SolrInputDocument> allDocs = new java.util.ArrayList<>();
        allDocs.add(parentDoc);
        allDocs.addAll(childDocs);

       solr.add(allDocs);
       solr.commit();
       System.out.println("Documents Indexed");
        solr.close();

    }
}

```

This java code uses a simple loop to add the parent document and the child documents all together to the index this is the magic it allows lucene to see all these documents together and allow to query using block join and also see how there is a type field to discriminate the parent and the children documents

Now you might be thinking " so where do we tell solr that they are connected in parent-child fashion?" well my friend you tell Solr using the schema.xml file Here is a snippet of the relevant parts

```xml
 <schema name="example" version="1.6">
  <fields>
   <field name="id" type="string" indexed="true" stored="true" required="true" />
   <field name="type_s" type="string" indexed="true" stored="true"/>
   <field name="title_t" type="text_general" indexed="true" stored="true"/>
   <field name="content_t" type="text_general" indexed="true" stored="true"/>
   <field name="commenter_s" type="string" indexed="true" stored="true"/>
   <field name="text_t" type="text_general" indexed="true" stored="true"/>
 </fields>
 <uniqueKey>id</uniqueKey>
 <types>
  <fieldType name="string" class="solr.StrField" sortMissingLast="true" />
    <fieldType name="text_general" class="solr.TextField" positionIncrementGap="100">
      <analyzer type="index">
       <tokenizer class="solr.StandardTokenizerFactory"/>
       <filter class="solr.StopFilterFactory" words="stopwords.txt" ignoreCase="true"/>
       <filter class="solr.LowerCaseFilterFactory"/>
      </analyzer>
      <analyzer type="query">
       <tokenizer class="solr.StandardTokenizerFactory"/>
       <filter class="solr.StopFilterFactory" words="stopwords.txt" ignoreCase="true"/>
       <filter class="solr.SynonymGraphFilterFactory" synonyms="synonyms.txt" ignoreCase="true" expand="true"/>
       <filter class="solr.LowerCaseFilterFactory"/>
      </analyzer>
     </fieldType>
  </types>
  </schema>
```

This is a bare minimum setup you can customize the fields and types but the important part is that you need a type field this is what I used to denote between parent and child documents and the trick with the schema is that you don't have to explicitly define any block join related configuration the trick is in the query language as you will see below.

**Example 2: Querying using block join**

Now the fun part let's say you want to find all posts that have comments containing the word "great" here is the query

```java
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.impl.HttpSolrClient;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;


public class BlockJoinQuery {

    public static void main(String[] args) throws Exception {
        String solrUrl = "http://localhost:8983/solr/mycore";
        SolrClient solr = new HttpSolrClient.Builder(solrUrl).build();

        SolrQuery query = new SolrQuery();
        query.setQuery("{!parent which=\"type_s:post\"}text_t:great");
        QueryResponse response = solr.query(query);
        SolrDocumentList results = response.getResults();

        System.out.println("Found " + results.size() + " results:");
        for (SolrDocument doc : results) {
            System.out.println("Post ID: " + doc.getFieldValue("id") + " Title: " + doc.getFieldValue("title_t"));
        }
        solr.close();
    }
}
```

 so what’s happening here? We’re using the `{!parent}` query parser it’s like saying “Hey Solr give me the parent documents that have these child characteristics” `which="type_s:post"` tells Solr to look for the parent type documents and `text_t:great` looks for the child documents with this text

The key part is that it’s not matching the posts that contain “great” it's matching the posts that have *children* that contain the word “great” that’s the power of block join. It lets you traverse the relation between parents and child documents to create complex search logic

I also ran into some crazy situation once when I was debugging I had a bug in the code and it was giving me results that were unexpected and it was when I noticed I had not put the query correctly I am telling you these block joins can cause nightmares if you are not careful but that's part of the charm of working with these complex systems isn't it?

**Example 3: Querying child documents with parent filter**

Now let's flip the table how about getting all the comments for a specific post?

```java
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.impl.HttpSolrClient;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;


public class ChildQueryExample {

 public static void main(String[] args) throws Exception {
   String solrUrl = "http://localhost:8983/solr/mycore";
   SolrClient solr = new HttpSolrClient.Builder(solrUrl).build();


   SolrQuery query = new SolrQuery();
   query.setQuery("{!child of=\"type_s:post id:post1\"}type_s:comment");
    QueryResponse response = solr.query(query);
    SolrDocumentList results = response.getResults();


   System.out.println("Found " + results.size() + " comments for post post1:");
   for (SolrDocument doc : results) {
     System.out.println("Comment ID: " + doc.getFieldValue("id") + " Commenter: " + doc.getFieldValue("commenter_s"));
   }
   solr.close();
 }
}
```

See the difference here? We're using the `{!child}` query parser and the `of` parameter which is referencing a parent that matches `type_s:post` and `id:post1` and then we ask for all the child documents `type_s:comment`. This gives you the power to narrow down to specific children associated to a parent

Now some words of wisdom the performance of these queries really depends on the complexity of the queries and how much you have in your index and your hardware but for large datasets block join is a massive win compared to trying to do application-level joins after the fact

If you want to dive deeper into the technical details I would recommend checking some research papers on information retrieval specifically papers on nested document retrieval or hierarchical data indexing that’s where you’ll find the underpinnings of the techniques used by Lucene. Also look for the Apache Lucene documentation it’s not always the most fun read but it's the bible for this stuff Also look for the book Lucene in Action it is a great resource

And before I forget one last thing debugging these queries can be a pain I’ve had moments where I’ve stared at the query for hours before realizing I had a typo a simple typo can lead you to a rabbithole of despair so be very very careful or maybe I just need more coffee

So there you have it block join in a nutshell a simple but super powerful tool for handling relational data in a search context
