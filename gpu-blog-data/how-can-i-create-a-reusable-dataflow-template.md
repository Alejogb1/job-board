---
title: "How can I create a reusable Dataflow template for job orchestration?"
date: "2025-01-30"
id: "how-can-i-create-a-reusable-dataflow-template"
---
Dataflow templates, while simplifying repetitive job deployments, necessitate careful design to maximize reusability. My experience building a large-scale data ingestion pipeline, processing thousands of files daily from varied sources, underscored this. A truly reusable template requires a parameterizable pipeline structure, decoupling processing logic from specific configurations.

The core principle involves structuring your Dataflow pipeline around modular, well-defined Transforms, each accepting runtime parameters. This approach avoids hardcoding source locations, schemas, or processing logic within the pipeline's core structure. Instead, parameters passed to the Dataflow job at launch determine how these transforms operate. This flexibility transforms a one-off job definition into a versatile tool for various orchestration scenarios.

Consider an ingestion pattern: often, we're fetching data from multiple file locations, potentially with different schemas and target destinations. A non-reusable approach might embed the specifics of a single source-to-sink mapping. However, if we abstract this into a pipeline accepting file paths, schemas, and BigQuery table locations as runtime parameters, we can use the same template for a multitude of sources. The pipeline's structure, involving file reading, schema application, and BigQuery write, remains unchanged. The *specifics* are parameterized.

Implementing this parameterization largely relies on two core Dataflow concepts: the `--parameters` flag at job launch and the use of `ValueProvider` to access those parameters within your pipeline definition. The `--parameters` flag, available through `gcloud dataflow jobs run`, enables you to pass key-value pairs during job execution. Within your pipeline, you wrap each parameter needed using `ValueProvider`, effectively deferring the parameter value resolution until runtime.

For instance, instead of directly specifying a Google Cloud Storage path, the pipeline defines a `ValueProvider<String>` for it. This allows you to define parameters such as `--inputPath=gs://your-bucket/input/` at launch. The actual path is resolved only when the Dataflow job executes, effectively making your pipeline source-agnostic.

Let's illustrate this with code examples, focusing on key components of a reusable template. First, we will demonstrate reading data from a flexible GCS location based on a parameter.

```java
  public static class ReadFromGCS implements SerializableFunction<ValueProvider<String>, PCollection<String>>{

    @Override
    public PCollection<String> apply(ValueProvider<String> inputPath) {
       return pipeline.apply("Read From GCS", TextIO.read().from(inputPath));
    }
 }

 // Main pipeline code
 pipeline.apply(new ReadFromGCS().apply(options.getInputPath())); // options would contain parameter from commandline.
```

Here, `ReadFromGCS` is a serializable function encapsulating the logic of reading data from GCS using `TextIO.read()`. The critical detail is that `from()` takes a `ValueProvider<String>`, obtained from the pipeline's command-line options. The actual path is not defined in the pipeline code; instead it is read during the runtime execution using options. This demonstrates parameterization of source locations.

Next, consider transforming and writing data to a dynamic BigQuery destination.

```java
  public static class WriteToBigQuery implements SerializableFunction<PCollection<TableRow>, PCollection<TableRow>> {
      private final ValueProvider<String> bqTable;

       WriteToBigQuery(ValueProvider<String> bqTable){
           this.bqTable = bqTable;
       }

        @Override
        public PCollection<TableRow> apply(PCollection<TableRow> tableRows) {
           tableRows.apply(BigQueryIO.writeTableRows()
                  .to(bqTable)
                  .withSchema(tableSchema)
                 .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED)
                 .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_APPEND));

        return tableRows;
       }
  }

  // Main pipeline code
  pipeline.apply(new TransformData()).apply(new WriteToBigQuery(options.getBqTable())); // options contain command line parameter
```

The `WriteToBigQuery` class demonstrates a similar pattern. It accepts a `ValueProvider<String>` for the BigQuery table reference, along with a schema object for the table rows. The destination table is not hardcoded, allowing the pipeline to write results into different BigQuery tables based on the parameters provided at launch.

Finally, let's consider a transformation, in this case simple string manipulation, which also takes runtime parameters.

```java

public static class TransformData implements SerializableFunction<PCollection<String>, PCollection<TableRow>> {
    private final ValueProvider<String> delimiter;
    private final ValueProvider<String> columnNames;

    TransformData(ValueProvider<String> delimiter, ValueProvider<String> columnNames){
           this.delimiter = delimiter;
           this.columnNames = columnNames;
    }

    @Override
    public PCollection<TableRow> apply(PCollection<String> inputRows) {
       return inputRows.apply(
         MapElements.via(new SimpleFunction<String, TableRow>() {
            @Override
            public TableRow apply(String row) {
                TableRow tableRow = new TableRow();
                String[] elements = row.split(delimiter.get());
                String[] columns = columnNames.get().split(",");

               for(int i = 0; i < elements.length; i++){
                   tableRow.set(columns[i], elements[i]);
               }
                 return tableRow;
               }
            })
       );
    }
}

//Main pipeline code
pipeline.apply(new TransformData(options.getDelimiter(), options.getColumnNames())); // options contain command line parameters
```

Here, `TransformData` also takes the delimiter, and column names as parameters which can be read from the command line using the options. This allows for dynamic behavior during transformation. The specific delimiter for splitting the incoming string, as well as column headers, can be passed in as run time parameters. The transformation logic itself remains static; however, it operates differently based on input parameters, demonstrating flexibility in data processing steps.

These examples showcase the core principles of building reusable Dataflow templates. By employing `ValueProvider` and structuring our pipeline around parameterized transforms, we can achieve a level of flexibility that enables us to reuse our pipeline across multiple job scenarios by defining configuration parameters at run-time. This means, the actual logic for reading, transforming and writing is separated from the configuration.

Effective reuse also necessitates meticulous template documentation. Each parameter, with its expected data type and purpose, needs to be documented clearly for downstream users. Version control of your templates is also critical, especially as your pipeline evolves. You might also consider developing standardized input parameter formats and output formats for improved reusability and interoperability.

For further learning about structuring reusable Dataflow pipelines and templating, resources such as official Google Cloud Dataflow documentation, and community contributions on data engineering best practices are invaluable. Specific focus should be placed on `ValueProvider` usage and design patterns around encapsulating complex transforms into reusable functions. Examining public code repositories containing Dataflow templates provides practical insight. Consider a book on Advanced Dataflow Techniques, or an in-depth video course on data engineering patterns. These, along with official documentation, offer comprehensive details on the nuances of reusable template design and best practices.
