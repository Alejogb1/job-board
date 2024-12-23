---
title: "large pdf sample file generation?"
date: "2024-12-13"
id: "large-pdf-sample-file-generation"
---

so you're asking about large PDF generation specifically sample files I've wrestled with this beast before it's a classic case of "sounds simple until you actually try it" believe me I know a thing or two

Let's get down to the gritty details here generating a PDF isn't inherently complex the problem comes when you're talking about large files either huge page counts or lots of complex content or a combination of both that's where things can get slow and memory intensive fast

Back in my days at a now defunct startup we needed to generate product catalogs thousands of them with variable layouts images tables the whole nine yards we started with a basic PDF library something common like iText or similar in Java and we immediately hit a wall our server kept crashing or going into like a coma we were generating 10-20 page catalogs and even those were taking like 30 seconds each that's just brutal in the real world.

First the naive approach we did something very elementary like looping through a list and create a page or an object of data on each page add content and then save this and it worked but very very slow and consumed a lot of memory something that we were not ready for and we quickly realized we were approaching this wrong

Here is what we initially had something very basic

```java
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Paragraph;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class NaivePdfGenerator {

  public static void main(String[] args) throws FileNotFoundException {
    List<String> data = generateDummyData(5000); // Let’s simulate a list of data

    PdfWriter writer = new PdfWriter("naive_pdf.pdf");
    PdfDocument pdf = new PdfDocument(writer);
    Document document = new Document(pdf);

    for (String item : data) {
        document.add(new Paragraph(item));
    }

    document.close();
    System.out.println("PDF with Naive approach generated");
  }
  private static List<String> generateDummyData(int count) {
    List<String> data = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      data.add("Item " + i + ": Some text content for the PDF.");
    }
    return data;
  }
}
```

The problem is that we were loading everything into memory and building the document all at once which when you think about is stupid you're dealing with a large amount of data and throwing it all at the PDF generator in one go if you do this you will crash or at least take a lot of time and resources from your machine.

The next logical step for us was thinking how to not load all this in memory at once so we started with the concept of streaming which seems obvious now but it wasn't at the time or at least not to my team we used a technique of creating the PDF stream as we went that would be great for big files.

Here’s a snippet of what we did using the same library iText

```java
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Paragraph;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.io.FileOutputStream;
import java.io.IOException;

public class StreamingPdfGenerator {

    public static void main(String[] args) throws IOException {
        List<String> data = generateDummyData(5000); // Simulate data again
        try (FileOutputStream fos = new FileOutputStream("streaming_pdf.pdf");
             PdfWriter writer = new PdfWriter(fos);
             PdfDocument pdf = new PdfDocument(writer);
             Document document = new Document(pdf)) {

            for (String item : data) {
                document.add(new Paragraph(item));
            }
           System.out.println("PDF using a streaming approach generated");
        }
    }
    private static List<String> generateDummyData(int count) {
        List<String> data = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            data.add("Item " + i + ": Some text content for the PDF.");
        }
        return data;
    }

}
```
This approach was much better we could deal with larger files and we were not running out of memory as fast but the generation was still slow very slow specially when the document size increased and the content complexity was more than a simple paragraph so we needed something else to make it even faster.

So we started diving more into the libraries and looking at how they structured the documents and one key insight was understanding that you don't need to create the objects for each element if they are very repetitive and similar like we were doing for those product catalog the main issue was that we were creating a paragraph object for every single entry and that was very wasteful

So we leveraged what PDF libraries called "Templates" or "Patterns" this involved creating a base PDF structure with placeholders and then populating the placeholders with the actual data that sped things up considerably because you pre-define a lot of the layout elements and reuse them this avoids re-parsing and re-creating the basic objects for each row or element in the data and is something you should seriously consider when working with huge data in your pdf.

Here’s some code that represents what i mean it's still very much simplified but it gives you the gist

```java
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Paragraph;
import com.itextpdf.layout.element.Table;
import com.itextpdf.layout.element.Cell;
import com.itextpdf.kernel.pdf.PdfPage;
import com.itextpdf.layout.properties.UnitValue;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TemplatePdfGenerator {

    public static void main(String[] args) throws IOException {
        List<List<String>> data = generateTableData(5000, 3);

        try (FileOutputStream fos = new FileOutputStream("template_pdf.pdf");
             PdfWriter writer = new PdfWriter(fos);
             PdfDocument pdf = new PdfDocument(writer);
             Document document = new Document(pdf)) {
            // Create table structure
            Table table = new Table(UnitValue.createPercentArray(3));
            table.setWidth(UnitValue.createPercentValue(100));

            // Add header row
            table.addHeaderCell("Column 1");
            table.addHeaderCell("Column 2");
            table.addHeaderCell("Column 3");

            for (List<String> row : data) {
                table.addCell(new Cell().add(new Paragraph(row.get(0))));
                table.addCell(new Cell().add(new Paragraph(row.get(1))));
                table.addCell(new Cell().add(new Paragraph(row.get(2))));
            }

             document.add(table);

            System.out.println("PDF with a template style generated");
        }
    }
  private static List<List<String>> generateTableData(int rows, int cols) {
        List<List<String>> data = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<String> row = new ArrayList<>();
            for (int j = 0; j < cols; j++) {
                row.add("Row: " + i + " Col: " + j);
            }
            data.add(row);
        }
        return data;
    }
}
```

Notice that in this last approach we are using a table with headers and rows which is a common pattern that can be reused instead of creating simple paragraphs and adding them one by one this shows how templates can help

Beyond the code itself other things I learned that are key for generating large PDF's include:

*   **Memory Management**: Be mindful of object creation and make sure you dispose of resources like images font handlers etc as soon as you're done with them this can be tricky specially when working with complex code so a good code review session can help.
*   **Font Caching**: Preload your fonts its something that can save you time later if you keep reloading the fonts for each object creation this can add up to a big time penalty.
*   **Image Optimization**: If you're dealing with a lot of images make sure they're properly resized and compressed you don't need a 5MB image when a 200kb version will do the job and nobody will notice the difference this also helps with network transfer times if you're fetching the images from somewhere.
*   **Content Pre-processing**: Generate any text content you can beforehand avoid string manipulation inside the PDF generation loop and this also allows you to cache results.
*   **Parallelization**: If you have the resources consider generating different pages or sections of the PDF in parallel and then merging them at the end if you can this is a common pattern in this area and will reduce time by processing faster.
*   **Logging and monitoring**: Always log errors and performance and metrics and keep track of memory usage if you have a production setting for the generation this will help you identify bottlenecks and other problems in real time.

I should add that there's no silver bullet here you will need to experiment and profile and see what works best for your specific case you need to do your own profiling and testing to see where the main bottlenecks are for your case.

Also one more thing remember to keep your PDF generation logic outside of your web requests don't generate PDF's on your web thread it will block them and create a bad user experience.

For resources I recommend the iText documentation itself and if you want a deep understanding of PDF internals get the "PDF Reference Sixth Edition" from Adobe. Also take a look at “PDF Explained" by John Whitington its very good for more theoretical understanding of the format. You'll learn a lot and become a pro pdf generator in no time believe me I know.

And that's pretty much it folks you asked about generating large PDF samples and I hope this helped you I've learned this the hard way so you don't have to the more you know the more you understand and the more you code the more you suffer wait did I say that out loud
