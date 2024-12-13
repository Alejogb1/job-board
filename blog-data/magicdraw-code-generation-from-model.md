---
title: "magicdraw code generation from model?"
date: "2024-12-13"
id: "magicdraw-code-generation-from-model"
---

Alright so you're asking about generating code from MagicDraw models yeah I’ve been there done that got the t-shirt and probably a few stress-induced grey hairs along the way Let’s just say MagicDraw and code generation it’s a journey not a destination okay

So first off I understand you're probably thinking "hey I've got this beautiful model in MagicDraw surely there's a button that just spits out perfect code" Yeah no not really That's the dream but reality is usually a little more hands-on more like a full blown coding project in itself actually

From what I understand you are looking to auto generate code and I bet your model includes classes attributes methods relationships all that good stuff You're hoping to get that model magically transformed into working code probably Java C++ Python or something right I'm just guessing here no mind reading involved

Okay so back in my early days circa late 2000s I had this client this big corporation type place they had a ridiculously huge MagicDraw model Like seriously the thing was a monstrosity I mean nested packages dependencies all over the place It looked more like a bowl of spaghetti than a well-structured system They needed code generated from it you know for a project I guess and my team was tasked with figuring it out the hard way We were young we were naive and we thought we could just click a button and get gold we were wrong oh so wrong

Our first mistake was thinking MagicDraw's built-in code generation was the answer We tried it oh boy did we try it It spat out code yeah but it was garbage like literally uncompilable garbage It was like someone had taken the model and randomly rearranged it and sprinkled it with syntax errors It was basically unusable I still have nightmares about the compile errors I had to fix by hand I remember one day spent debugging curly braces mismatches It was a character counting exercise I was having a bad day okay

So after much weeping and gnashing of teeth we realized the out-of-the-box solution wasn't going to cut it We had to go full custom so we moved to magicdraw apis and had to write some code for code generation

Now there are a couple of ways to skin this code generation cat and most are painful but some are less painful than others So let's talk about some methods we used back in the day that could be somewhat relevant even today but with a different spin

The first method is using MagicDraw's Open API This is your bread and butter if you want real control You’re basically writing code that interfaces with MagicDraw's internal model and then generates code based on what it finds This involves some serious Java programming and a deep understanding of the MagicDraw API But it gives you ultimate flexibility so for complex model structures it's kinda the only real game in town and this is the most widely used method.

Here is a very simple Java example of how to get class information

```java
import com.nomagic.magicdraw.core.Application;
import com.nomagic.magicdraw.core.Project;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Class;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Package;
import com.nomagic.uml2.ext.magicdraw.model.ModelQuery;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.NamedElement;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Classifier;

import java.util.Collection;

public class MagicDrawClassExtractor {

    public static void main(String[] args) {
        Project project = Application.getInstance().getProject();
        if (project == null) {
            System.out.println("No project loaded.");
            return;
        }

        // Assuming the model root is a package
        Package rootPackage = project.getModel();

        if (rootPackage != null) {
            findClasses(rootPackage);
        }
    }

    private static void findClasses(Package parentPackage) {
          Collection<NamedElement> ownedElements = parentPackage.getOwnedElement();
         for (NamedElement element : ownedElements) {

            if (element instanceof Class)
            {
                Class cls = (Class) element;
                System.out.println("Class: " + cls.getName());

            } else if (element instanceof Package){
                findClasses((Package) element);
            }


        }
    }
}
```
**Note**: You’ll need the MagicDraw API libraries in your classpath to get this to run. Don't try to run this without doing the proper setup first.

The second option more practical for simple model structures involves using templates MagicDraw allows you to define templates using some kind of templating language like Apache Velocity This is generally easier to get started with than the API approach because you don't have to be so knee deep in Java programming Instead you write templates that get populated with data from the model It's kind of like a mail merge but for code.

Here's an example of a simple Velocity template for a Java class:

```velocity
#set( $className = $element.name )
#set( $attributes = $element.attribute )

public class $className {

#foreach( $attr in $attributes )
    private $attr.type.name $attr.name;
#end

    public $className() {

    }

#foreach( $attr in $attributes )
    public $attr.type.name get$attr.name(){
        return this.$attr.name;
    }

    public void set$attr.name($attr.type.name $attr.name){
        this.$attr.name = $attr.name;
    }
#end

}
```

In reality this velocity template is not going to work directly but its a good starting point it needs some additional logic for example it wont handle package or imports or some other more advanced UML elements but its a good foundation.

You’ll need a MagicDraw plugin or some other mechanism to actually process this template and feed it the model data. If you want more information on Apache Velocity templates read its documentation it's a powerful tool but its own set of caveats.

Now a third approach could be a mix of both using an external library or tool that knows how to read MagicDraw's XMI output and then generate code That involves exporting your MagicDraw model to XMI then using that information to build up a code structure it may require creating a new model structure which adds a lot of extra steps to the process so its generally a second choice to the first two options

Here is an example of getting XMI using python

```python
import xml.etree.ElementTree as ET

def parse_xmi_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Define XML namespace (if needed)
    ns = {'xmi': 'http://www.omg.org/XMI'} # Example namespace, adjust if needed

    # Example: Find all class elements
    for class_element in root.findall(".//{"+ns['xmi']+"}Class"):
        class_name = class_element.get("name")
        if class_name:
            print("Class Name:", class_name)
        
        for attribute_element in class_element.findall(".//{"+ns['xmi']+"}Property"):
            attr_name = attribute_element.get("name")
            attr_type = attribute_element.get('type') # this may require more advanced logic to resolve
            if attr_name:
                print("Attribute Name:",attr_name," Type:",attr_type)
        print("\n")


# Example Usage:
xmi_file = "model.xmi" #Replace your_file.xmi
parse_xmi_file(xmi_file)
```

This python example also requires you to define your namespace which may vary depending on the format of the XMI file. And this code is also simplistic but it shows the basic idea of parsing XMI files. Note this script does not generate anything, it reads from file, you will still need to generate the code yourself using this information.

Now all of this code is basic but the concepts and the idea are there. In my experience most real-world code generation projects require a combination of all those approaches with lots of custom logic for specific model properties code annotations dependency resolution dealing with the many many exceptions and particularities in every single model there is always something different. It’s never straightforward

So what resources can help you on this quest? Well first I would recommend "UML 2.5 for Dummies" its a good starting point if you want to brush on UML concepts the same goes for the "UML 2 and the Unified Process" book.

For magicdraw itself you will need to go over its documentation pages if you want to use its internal API and there are multiple books for it but generally the documentation will be enough you can also go over the community forums if you are stuck. For velocity template there is no substitute but the original velocity documentation but be aware that it has several versions so stick to one.

Finally for XMI files you will need to go over the original OMG specifications that can get dense pretty fast but there is no substitute if you want the full picture.

This process is painful it's iterative and there are always corner cases you haven’t thought about I swear one time we found a special character in a class name that was breaking everything It was almost the same as when my car stereo started speaking in Portuguese after an update We had to sanitize inputs the model before generating code.

Anyway I hope this helps you in your code generation endeavor And if you have more questions feel free to ask but be prepared because I've probably seen it all with code gen I’ve seen things you wouldn’t believe. And hopefully, you won't have to debug curly braces for days because of badly generated code. Good luck out there.
