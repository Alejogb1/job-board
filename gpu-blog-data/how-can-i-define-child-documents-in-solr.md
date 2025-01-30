---
title: "How can I define child documents in SOLR using a DataImportHandler with a URLDataSource?"
date: "2025-01-30"
id: "how-can-i-define-child-documents-in-solr"
---
The nuanced challenge of indexing hierarchical data structures, such as parent-child relationships, within Solr using the DataImportHandler (DIH) and a URLDataSource necessitates a precise understanding of the handler's configuration and data transformation capabilities. My experience working on a large-scale document management system revealed that merely pointing DIH to a nested JSON or XML structure is insufficient; explicit instructions must be provided to guide the indexing process.

**Conceptual Understanding**

Solr does not inherently recognize hierarchical structures. Child documents, in Solr's perspective, are independent documents that are linked to a parent document via a common field. To establish this relationship, the DIH must be configured to process each child document separately, while ensuring they are associated with their correct parent. This often involves transforming the data into a format that the DIH can interpret as distinct Solr documents, each with a reference to its parent’s unique identifier. The core idea lies in instructing the DIH how to identify a parent document, how to identify a child document and how to associate a child document with a parent document.

**Configuration Strategy**

The critical piece is leveraging the `entity` tag within the `dataConfig.xml` file, and, specifically, the `transformer` and `childEntity` constructs. We will focus on how these can be applied with a JSON structure obtained from a URLDataSource. Assuming a JSON structure resembling this:

```json
{
   "documents":[
      {
         "id":"parent_1",
         "title":"Parent Document One",
         "children":[
            {
               "id":"child_1a",
               "content":"Child Content A"
            },
             {
               "id":"child_1b",
               "content":"Child Content B"
            }
         ]
      },
      {
         "id":"parent_2",
         "title":"Parent Document Two",
          "children":[
            {
               "id":"child_2a",
               "content":"Child Content C"
            }
        ]
      }
   ]
}
```

The primary `entity` element will process each parent document, and then the `childEntity` elements will be responsible for indexing each child. A key aspect is adding a field to the child document containing a link to the parent documents id which enables searching related documents.

**Code Examples with Commentary**

I will demonstrate the configuration using three code examples. The first will show the most basic setup to get hierarchical documents indexed. The second and third examples will then add functionality, first by specifying childEntity fields, and then by leveraging a more complex transform.

**Example 1: Basic Hierarchical Indexing**

```xml
<dataConfig>
  <dataSource type="URLDataSource" url="http://example.com/data.json"/>
  <document>
      <entity name="parent" 
              processor="XPathEntityProcessor" 
              root="/documents" 
              forEach="/documents/document"
              transformer="JsonPathTransformer" >

        <field column="id" jsonPath="$.id" name="id"/>
        <field column="title" jsonPath="$.title" name="title"/>
        
          <entity name="child" 
                    processor="XPathEntityProcessor" 
                    root="/documents/document/children" 
                    forEach="/documents/document/children/item"
                    transformer="JsonPathTransformer">
              <field column="id" jsonPath="$.id" name="id"/>
              <field column="content" jsonPath="$.content" name="content"/>
             <field column="parent_id" jsonPath="$.parent().id" name="parent_id"/>
          </entity>
    </entity>
  </document>
</dataConfig>
```

In the above code, the top-level entity, `parent`, iterates over the documents, extracting the parent ID and title. The `childEntity`, `child`,  iterates over the `children` arrays. The critical line is the `field` declaration for the `parent_id`. The JsonPathTransformer function `parent()` allows access to the parent element's data. This ensures the correct parent ID is stored with each child document in the "parent\_id" field, establishing the link. This basic configuration will index the parent documents with fields "id" and "title", while the child documents will be indexed with fields "id", "content", and "parent\_id".

**Example 2: Specifying Fields in `childEntity`**

```xml
<dataConfig>
  <dataSource type="URLDataSource" url="http://example.com/data.json"/>
  <document>
      <entity name="parent" 
              processor="XPathEntityProcessor" 
              root="/documents" 
              forEach="/documents/document"
              transformer="JsonPathTransformer" >

        <field column="id" jsonPath="$.id" name="id"/>
        <field column="title" jsonPath="$.title" name="title"/>
        
          <entity name="child" 
                    processor="XPathEntityProcessor" 
                    root="/documents/document/children" 
                    forEach="/documents/document/children/item"
                    transformer="JsonPathTransformer">
               <field column="id" jsonPath="$.id" name="child_id"/>
               <field column="content" jsonPath="$.content" name="child_content"/>
                <field column="parent_id" jsonPath="$.parent().id" name="parent_id"/>
             </entity>
    </entity>
  </document>
</dataConfig>
```

This example expands upon the first example by explicitly naming the fields indexed for the `childEntity`. Here, instead of a generic "id" and "content" for child documents, we now have `child_id` and `child_content` which provides better context when reviewing your indexed schema. This highlights the flexibility in controlling field names during the indexing process. The parent documents will remain the same as the first example, but the child documents will now be indexed with fields "child\_id", "child\_content", and "parent\_id".

**Example 3: Leveraging a More Complex Transform**

```xml
<dataConfig>
  <dataSource type="URLDataSource" url="http://example.com/data.json"/>
  <document>
      <entity name="parent" 
              processor="XPathEntityProcessor" 
              root="/documents" 
              forEach="/documents/document"
              transformer="JsonPathTransformer" >

        <field column="id" jsonPath="$.id" name="id"/>
        <field column="title" jsonPath="$.title" name="title"/>
        
          <entity name="child" 
                    processor="XPathEntityProcessor" 
                    root="/documents/document/children" 
                    forEach="/documents/document/children/item"
                    transformer="ScriptTransformer">
               <script><![CDATA[
                  function transformRow(row) {
                    row.put('child_id', row.get('id'));
                    row.put('child_content', row.get('content').toUpperCase());
                    row.put('parent_id', row.parent().get('id'));
                    return row;
                  }
                 ]]>
               </script>
              <field column="child_id"  name="child_id"/>
              <field column="child_content" name="child_content"/>
              <field column="parent_id" name="parent_id"/>
            </entity>
    </entity>
  </document>
</dataConfig>
```

This final example demonstrates how to use the `ScriptTransformer` for more intricate transformations. We now use the ScriptTransformer to first, populate the 'child\_id' and 'parent\_id' as in the previous examples, but also to transform the content for every child document using javascript, specifically converting the content to uppercase. This is useful when performing complex transformations, and the `ScriptTransformer` provides a way to accomplish this using javascript. The parent documents will remain the same as the previous examples, but the child documents will be indexed with fields "child\_id", "child\_content" (which will be uppercase), and "parent\_id".

**Resource Recommendations**

To further investigate the DataImportHandler, consult the official Solr documentation pages on DIH configuration and the available transformers, such as XPathEntityProcessor and JsonPathTransformer. Furthermore, consider reviewing documentation for Solr's document relationships, often referred to as "block join queries," as this allows you to use the parent\_id field to search across these relationships. Resources on using Solr's query syntax is also highly recommended. Specific books about Solr could also be informative.

**Conclusion**

Defining child documents using a DataImportHandler and a URLDataSource in Solr requires a careful approach to configuring the DIH. Using `entity` and `childEntity` elements allows you to recursively process a JSON or XML structure and explicitly specify how each parent and child document should be indexed. It is important to understand how the different transformers can be used to extract and transform the data into Solr fields. By applying these principles and understanding the structure of your data, you can effectively index hierarchical relationships and make them searchable in your Solr implementation.
