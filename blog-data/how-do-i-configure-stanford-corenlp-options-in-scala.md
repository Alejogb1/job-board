---
title: "How do I configure Stanford CoreNLP Options in Scala?"
date: "2024-12-15"
id: "how-do-i-configure-stanford-corenlp-options-in-scala"
---

alright, so you’re looking to tweak stanford corenlp’s options when using it within scala. been there, done that, got the t-shirt, which by the way is stained with coffee from those late nights trying to get it to work perfectly. believe me, the default settings, while useful, don't always cut it for every task. let me share some hard-won wisdom from my past projects.

first, forget about any magical incantations. it’s all about understanding the underlying java api which stanford corenlp exposes, and then translating that into scala idiom. corenlp is essentially a collection of java classes, and scala allows us to interact with them quite seamlessly. the magic is that you are using the java version inside scala, and that will affect how you configure the options.

when i first started messing with corenlp in scala, i remember spending hours scratching my head. i was trying to get it to recognize specific entity types in my domain and the default ners were just not cutting it. i had this text corpus of old scientific papers, with all sorts of unusual names and terms. the out-of-the-box solution just marked everything as “misc”, which was pretty useless.

anyway, the core of your configuration lies in the properties object you pass when creating a `stanfordcorenlp` pipeline. this properties object is a java `java.util.properties` instance. we can create one in scala and then feed it in.

here’s a basic example:

```scala
import java.util.Properties
import edu.stanford.nlp.pipeline.{stanfordcorenlp, annotation}

object corenlpconfig extends app {

  val props = new properties()
  props.setproperty("annotators", "tokenize,ssplit,pos,lemma,ner")
  props.setproperty("ner.model", "edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz")
  val pipeline = new stanfordcorenlp(props)

  val text = "the quick brown fox jumps over the lazy dog."
  val document = new annotation(text)
  pipeline.annotate(document)

  // do whatever you need to with the document and its annotations

}
```

in this snippet, we're setting the basic annotators: tokenization, sentence splitting, part-of-speech tagging, lemmatization, and named entity recognition. we're also explicitly specifying the ner model file. the interesting thing to notice is the absolute path, which needs to exist inside your project structure or be accessible in your classpath. i had my fair share of classpath issues so, always double check that. that's one issue i dealt with quite a while when i first started. if not this will cause errors.

now, let’s say you wanted to customize the ner model further or add a custom one. this is where things get a little more involved but are definitely not impossible. i remember having to train a custom model for chemical named entity recognition, which was surprisingly fun. i used the `stanford ner trainer` for that task, which is documented in the corenlp manual (you need to train it, which takes time, and this should be outside of your scala application, but you should load the trained model).

here's an example of how you can set a custom ner model if you have one:

```scala
import java.util.Properties
import edu.stanford.nlp.pipeline.{stanfordcorenlp, annotation}

object customner extends app {

  val props = new properties()
  props.setproperty("annotators", "tokenize,ssplit,pos,lemma,ner")
  props.setproperty("ner.model", "/path/to/your/custom/model.ser.gz")
    // replace with your actual trained model

  val pipeline = new stanfordcorenlp(props)

  val text = "the new compound, xyz-123, was observed"
  val document = new annotation(text)
  pipeline.annotate(document)

 // now you can access your custom entity types
}
```

remember to replace `/path/to/your/custom/model.ser.gz` with the actual path to your trained model file. this path is relative to where you run your scala program or it can be an absolute path like the previous example. and again, ensure your classpath is correctly configured so that corenlp can find the file. getting classpaths working and running smoothly can feel like you are trying to catch smoke with a net, but once you understand the principle, it's a bit easier.

let me tell you about a particularly annoying issue i once had. i was trying to use corenlp to process text from a database that sometimes contained very messy formatting – lots of extra whitespace, random line breaks, and even some garbled characters. i was getting inconsistent tokenization and weird pos tags and i realized that stanford corenlp was processing the input literally as is. this issue was due to the default `tokenize` annotator which didn't handle my text's format that well. so, i had to add a preprocessing step and some more advanced options using a custom regex splitter (again more undocumented advanced options) to fine-tune the tokenizer.

here's a snippet showing how you might use a custom tokenizer if you really needed to, or you need more than the default one, and that would require reading more corenlp docs about the `tokenize` annotator:

```scala
import java.util.Properties
import edu.stanford.nlp.pipeline.{stanfordcorenlp, annotation}

object customtokenizer extends app {

  val props = new properties()
    props.setproperty("annotators", "tokenize,ssplit,pos,lemma,ner")
    props.setproperty("tokenize.options", "splitOnHyphens=false,normalizeFractions=false")
    // this example just disable some default tokenization options
    props.setproperty("ner.model", "edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz")


  val pipeline = new stanfordcorenlp(props)

  val text = "this-is a-sample text 1/2, but not so weird!"
    val document = new annotation(text)
  pipeline.annotate(document)

//analyze document
}

```

this example disables some default options of the stanford corenlp `tokenize`, like the split on hyphens or the normalization of fractions. if you have really messy text you might need to explore even more advanced custom options. these are not always obvious from the default guides so you might have to read the internal code of corenlp to find out the undocumented options. some are just not easy to find out.

another thing i remember is that, depending on the kind of text you’re processing, sometimes the default english model might not be the best choice. for example, if you're dealing with legal documents, the vocabulary might be very different from the general-purpose corpora used to train default models. you would then explore other pretrained models provided by the stanford nlp group or try to train your own. i know i once used a model that was trained specifically for biomedical text, which made a considerable difference in my task, so always be ready to try other available models or build your own if needed. and that is, a lot of work, trust me.

now, if you’re really serious about natural language processing, i’d strongly recommend taking a look at the stanford corenlp manual, that is the bible for it. also, “speech and language processing” by jurafsky and martin is an invaluable resource, it's not specific to corenlp, but will give you a great theoretical background. also a good resource is "natural language processing with python", by bird, klein and loper. even if you use scala, there's a lot of good theory that can be applied to your code (and the code examples are useful too).

debugging your stanford corenlp setups can be tricky. sometimes the error messages are as clear as a glass of mud. when you get those bizarre exceptions, check the classpath. double-check your properties and the paths, especially the ones to custom models. use a debugger to explore the `properties` object after you’ve created it. printing the properties object to the console is also a good idea, to make sure everything has the values you expect. sometimes a typo in the property key is all it takes to break it. it’s a good practice to write tests to verify the expected outcome when you configure corenlp and you change options (i know i didn’t do it always and then suffered with broken code). and last but not least, try not to pull all nighters when the code does not work, otherwise you might end up like me, with coffee stains in your t-shirt.

oh, and did you hear about the programmer who got stuck in the shower? he couldn't figure out how to get out, because the instructions on the shampoo bottle said "lather, rinse, repeat."
