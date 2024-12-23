---
title: "it-145 project 2 java code?"
date: "2024-12-13"
id: "it-145-project-2-java-code"
---

 so you're looking at IT-145 project 2 java code right Been there done that Let me tell you it usually involves a bunch of specific java concepts mashed together and everyone seems to get stuck on the same points I remember the first time I saw that project my initial thought was this can't be THAT hard boy was I wrong

Its probably something about class structures inheritance maybe some basic data structures and perhaps input output streams Seems like a typical intro to OOP project So you're probably feeling a little lost and that's completely normal I have been there Believe me

Let's talk about this first thing first Usually these types of projects try to test your understanding of object oriented principles So you'll likely have to create classes and define their methods and then perhaps set up some inheritance hierarchies For example maybe you have a base class called `Item` and then a bunch of subclasses like `Book` `DVD` or `Software` each with their own attributes and behaviors This sounds familiar right? I bet this part of it has caught you

Here's a sample code that you might find useful as a start point This snippet demonstrates the basic structure of an `Item` class with some simple attributes and methods This way you can understand the base structure and then expand it with other specific class you have to implement:

```java
public class Item {
    private String title;
    private String id;
    private double price;


    public Item(String title, String id, double price) {
        this.title = title;
        this.id = id;
        this.price = price;
    }


    public String getTitle() {
        return title;
    }


    public void setTitle(String title) {
        this.title = title;
    }


    public String getId() {
        return id;
    }


    public void setId(String id) {
        this.id = id;
    }


    public double getPrice() {
        return price;
    }


    public void setPrice(double price) {
        this.price = price;
    }


    @Override
    public String toString() {
        return "Item{" +
               "title='" + title + '\'' +
               ", id='" + id + '\'' +
               ", price=" + price +
               '}';
    }
}
```

Now you probably also have to implement some kind of data storage or data structure to hold these objects Maybe an `ArrayList` or some other collection that you can manipulate I remember when i was doing my first project like this I went through so many implementations of `ArrayLists` and `HashMaps` that I could probably write the javadoc myself from memory the pain still resonates in my coding memory

The whole process was frustrating especially when you find a weird bug that made no sense in theory but in the actual code the problem came from a misplaced pointer That was an interesting one and this happens way too often I hope that this project does not bring you such despair although I cannot promise that

 here's an example of how to use an `ArrayList` to store `Item` objects and add some functionalities like add remove and display items The other important functionality might involve searching by item id or title but that depends on the project requirements and you should implement that by yourself

```java
import java.util.ArrayList;
import java.util.List;


public class ItemManager {
    private List<Item> items;


    public ItemManager() {
        this.items = new ArrayList<>();
    }


    public void addItem(Item item) {
        items.add(item);
    }


    public void removeItem(String itemId) {
        items.removeIf(item -> item.getId().equals(itemId));
    }


    public void displayItems() {
        for (Item item : items) {
            System.out.println(item);
        }
    }


    public static void main(String[] args) {
        ItemManager manager = new ItemManager();
        manager.addItem(new Item("The Java Book", "BK101", 29.99));
        manager.addItem(new Item("Java for Dummies", "BK102", 19.99));
        manager.displayItems();
        manager.removeItem("BK101");
        System.out.println("After removal:");
        manager.displayItems();
    }
}
```

Now input output is another beast I think this is the fun part because its where you start reading from files writing data to other files and make your app interact with data In one project I remember we had to deal with CSV files It was a nightmare of splitting by commas and escaping all sorts of quotes It felt like the code was fighting back but in retrospect it was a fun challenge (not at the time though)
Here's an example of reading items from a simple text file and creating `Item` objects and add it to a list this is useful to load data from external files This is something that will likely be requested by the project so let's implement something useful for that:

```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class ItemLoader {


    public static List<Item> loadItemsFromFile(String filePath) {
        List<Item> items = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length == 3) {
                    String title = parts[0].trim();
                    String id = parts[1].trim();
                    double price = Double.parseDouble(parts[2].trim());
                    items.add(new Item(title, id, price));
                } else {
                    System.err.println("Skipping invalid line: " + line);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
        return items;
    }


    public static void main(String[] args) {
        // Assuming items.txt is in the same directory with data separated by commas
        // example: The Java Book,BK101,29.99
        String filePath = "items.txt";
        List<Item> loadedItems = ItemLoader.loadItemsFromFile(filePath);
        loadedItems.forEach(System.out::println);
    }
}

```
Remember the devil is in the details Pay close attention to how the classes interact with each other Make sure the class structures reflect the business requirements of the project and test every functionality It's not fun to spend hours debugging after submitting and having to fix it all again It's a waste of time that can be solved by proper planning and testing. That said lets be honest that's probably not gonna happen on this project because its a common pitfall when dealing with these assignments and I know it

Some good resources to study from are books like "Head First Java" it's a good introduction if you need to improve your Java knowledge Also "Effective Java" by Joshua Bloch is like the bible for any java developer is a must-read once you know the basics and "Clean Code" by Robert C. Martin is also a resource to create maintainable code

 let me throw in a little joke: why do java programmers wear glasses? Because they dont see sharp! HA!  terrible joke my bad I am not a comedian I am a programmer

Back to the main topic. I know that the project might seem hard but take it one step at a time Break the problem into smaller pieces implement one functionality at a time and test it thoroughly It will be worth the time once you see it working correctly

So to recap make sure you get your object oriented concepts in check define your class structures correctly use collections to store your data and make sure you are able to use input and output streams to interact with files Its all about proper design and incremental implementations

I hope this helps let me know if you get stuck somewhere maybe I can help but I can't promise to solve the whole assignment because learning the process is as important as the result itself but yeah feel free to ask if anything is unclear
