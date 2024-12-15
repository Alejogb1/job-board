---
title: "Why is Extending Parcelable with a new data field without breaking compatibility with old version?"
date: "2024-12-15"
id: "why-is-extending-parcelable-with-a-new-data-field-without-breaking-compatibility-with-old-version"
---

hey there,

so, you're diving into the wonderful world of android's `parcelable`, specifically the challenge of extending it without causing your app to crash and burn for users who haven't updated yet. i get it, i've been there, pulling my hair out over seemingly minor changes that cause major headaches. let me share some of the pain and lessons i've gathered over the years on this.

the core of the issue lies in how `parcelable` works under the hood. when you serialize a parcelable object, it's essentially a flattened version of your data structure written to a byte stream. this stream is then handed off, usually between components, activities, services, or even processes. the critical thing is the order and types of the data written. if you change that order, or add a new piece of data in a place where the old version doesn't expect it, boom, you get class cast exceptions or parcel exceptions. it's like expecting a five-course meal and receiving only four courses; the fifth fork would be useless, and a mess.

i'll tell you about this one time back in the early days of my android development, before i really had a handle on the importance of versioning parcelables. i was building a music player app and i needed to pass playlist data between activities. i had a `playlist` class implementing `parcelable` with `id`, `title`, and `artist`. life was good, everything was working. then came the 'brilliant' idea to add `artworkurl` to the `playlist` model. thinking it was just a simple add, i did it, deployed the new version, and went to lunch. i came back to a pile of crash reports. it turns out that users on the older version of the app, when they tried to receive the `playlist` parcel, got `parcelable` exceptions. since the older app expected three values to be read, after reading that it would expect the parcel to end and throw exception if found more.

my lesson was simple, but brutal: the serialization format of `parcelable` is extremely strict. it's all about the order and the number of things written. that's why your question on extending it *without* breaking compatibility is so important.

now, there are techniques to tackle this gracefully. the key is to be very deliberate about how you read and write data in your `parcelable` implementation. don't assume that everything will be there or that your old apps will be intelligent enough to skip unknown fields.

the first, and most fundamental, method is null checking: when reading values back from the parcel, be sure to use the correct order of reading the parcel in the read method. after every read method, check that the parcel isn't out of values, and if it is stop reading it.

here's how i usually set up reading the parcel in my models to avoid such issues:

```java
public class myparcelable implements parcelable {
    private int id;
    private String name;
    private String imageurl;
    private String description;

    // other methods and constructors

    protected myparcelable(parcel in) {
        id = in.readint();
        name = in.readstring();
        if(in.dataavailable()>0)
        {
          imageurl = in.readstring();
        }
        if(in.dataavailable()>0)
        {
           description = in.readstring();
        }


    }

    @override
    public void writetoparcel(parcel dest, int flags) {
        dest.writeint(id);
        dest.writestring(name);
        dest.writestring(imageurl);
        dest.writestring(description);
    }

    // ... parcelable methods
}
```
the `in.dataavailable()>0` check is crucial, if you remove it, old versions that don't expect the `imageurl` or `description` fields will throw exceptions, because the reading will try to read from a position that is not available in the parcel, leading to crashes.

another approach, particularly useful when you're dealing with optional fields, is to use `parcel.readint()` as a sort of "type indicator". if we have optional value in the model we write an integer which will indicate the presence of that value, if its 1 then the value is present and we read the value, otherwise we skip it. this is particularly useful for fields which you might add later on, but are not crucial to the functioning of the old app, and is a way to read only values if present on the parcel.

here’s an example of that:

```java
public class myparcelable implements parcelable {
   private int id;
   private String name;
   private String imageurl;
   private String description;


    // other methods and constructors

    protected myparcelable(parcel in) {
        id = in.readint();
        name = in.readstring();

       if(in.readint()==1)
       {
         imageurl=in.readstring();
       }

      if(in.readint()==1)
      {
         description = in.readstring();
      }
    }

    @override
    public void writetoparcel(parcel dest, int flags) {
         dest.writeint(id);
         dest.writestring(name);
          if(imageurl!=null)
          {
            dest.writeint(1);
            dest.writestring(imageurl);
          }else
            dest.writeint(0);
        if(description!=null)
          {
             dest.writeint(1);
             dest.writestring(description);
          }else
            dest.writeint(0);
    }

  // ... parcelable methods
}
```

in this example, we write a `1` before the image and description if it is available, and if it is not, we write a `0`. when reading the parcel, we first read the int and check if it is `1` before reading the string. this prevents parcel from trying to read values that are not available and also give the possibility to add and remove new values without breaking the old implementations of the models.

there are more sophisticated approaches, such as writing the class version inside the parcel and handle different versions of your model by implementing a factory pattern, but, these techniques often increase complexity and code and might not be a good fit for simple models. i try to keep it as simple as possible. there are times, when a simple solution works. this is the best approach so far that i have learned.

however, there is one thing that is absolutely important which we all tend to forget when dealing with parcelables, and it's about unit testing: it can sometimes be tempting to avoid writing tests for these classes because they seem so simple, but *don’t*. every single time you create a parcelable implementation or extend an existing one create a test for it. it might seem over the top but i have saved a lot of headaches on production by implementing this approach. the tests should read and write parcels with different scenarios like all values available, not all values available and null values. this will create a safety net on your changes, and guarantee that you did not forget to handle some situations.

here is an example of a simple test that reads and writes to a parcel:
```java
@test
public void testparcelable() {
     myparcelable original = new myparcelable(1,"name", "image", "description");
     parcel parcel = parcel.obtain();
     original.writetoparcel(parcel, 0);
     parcel.setDataposition(0);
     myparcelable restored = new myparcelable(parcel);
     assertthat(restored.getid(),is(original.getid()));
     assertthat(restored.getname(),is(original.getname()));
     assertthat(restored.getimageurl(),is(original.getimageurl()));
     assertthat(restored.getdescription(),is(original.getdescription()));

      myparcelable original2 = new myparcelable(1,"name", null, null);
      parcel parcel2 = parcel.obtain();
     original2.writetoparcel(parcel2, 0);
     parcel2.setDataposition(0);
      myparcelable restored2 = new myparcelable(parcel2);
      assertthat(restored2.getid(),is(original2.getid()));
     assertthat(restored2.getname(),is(original2.getname()));
     assertthat(restored2.getimageurl(),is(original2.getimageurl()));
     assertthat(restored2.getdescription(),is(original2.getdescription()));
}

```
note that the test example uses hamcrest matchers, you could use another approach for unit testing, this is just a small example of what can be tested for a parcelable implementation. in this case, it tests the full and partial values for the model, which covers a good portion of the possible issues with incorrect implementations.

as for further reading, i'd recommend looking into "effective java" by joshua bloch. it might not talk directly about parcelables, but the principles of versioning and interface design it covers are critical for understanding why things like this happen. the section on serialization is especially useful, even though `parcelable` is not the same as java serialization, the problems are the same. also look at the android source code in `parcel.java` in your android sdk path, it is very interesting to understand what happens under the hood on each parcel function.

remember, `parcelable` is a powerful tool for performance, but it demands rigor. a small mistake in its implementation can ripple through your user base. be sure to implement checks on all the values, implement unit tests for them and think on the best way to add or remove values while keeping compatibility with older versions.

one last joke before i go: why did the developer quit his job at the parcel delivery company? because he couldn't handle the constant stack overflow.

hope this helps, and may your parcelables always be bug-free.
