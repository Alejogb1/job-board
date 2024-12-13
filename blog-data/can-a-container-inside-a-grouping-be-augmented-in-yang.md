---
title: "can a container inside a grouping be augmented in yang?"
date: "2024-12-13"
id: "can-a-container-inside-a-grouping-be-augmented-in-yang"
---

Okay so you're asking if you can tweak a container within a grouping in YANG right that's the gist yeah I've been there done that got the t-shirt a few times let me tell you it's not as straightforward as you might hope but definitely doable

first things first let's get the terminology straight we're talking about YANG the data modeling language not some ancient greek deity for those of you who might be new to this the grouping is like a reusable building block we define it once then apply it in multiple places it's all about avoiding repetition in your models keeps things tidy but the catch comes when we need a bit of a twist a customization for a specific use of that grouping container we're referring to a specific data node structure holding other data items not the metal thing you see on cargo ships so now the core question can you enhance the contents of a container already declared within a grouping the short answer is yes but it needs some careful planning it's not like just shoving extra fields in hope it works there are rules we have to follow it's not chaotic programming

my early days I'm talking way back before some of you were even coding I had a similar situation I was trying to model network devices and we had this common interface configuration setup we put in a grouping it looked great all neat and tidy then boom we discovered that a few specific network devices needed some extra parameters that were simply not there in the original design and at first I was going to copy paste the entire grouping but with minor changes that obviously is not good practice so I ended up digging through the yang spec and found some solutions I tried to do that with inheritance but it is not really a language that supports that as it would break the core tenents of yang which is configuration oriented

so how do we go about this you essentially have two main options use `augment` statements or try `refine` let's dig deeper into each of them

**Using `augment`**

the `augment` statement is your friend here think of it as a way to add extra features to an existing data node it's like a software patch it does not change the original but it adds things to it we target the data node which is the container here within a grouping and then we can add new child nodes or modify some existing child nodes of the container but the key thing is that we cannot remove existing data nodes within an `augment` it only has the capability of expanding things so here is a simple code example

```yang
module example-module {
  namespace "urn:example:example-module";
  prefix ex;

  grouping common-interface-config {
    container interface {
      leaf name {
        type string;
      }
      leaf admin-status {
        type boolean;
        default true;
      }
    }
  }

  container devices {
    list device {
      key "name";
      leaf name {
        type string;
      }
      uses common-interface-config;
    }
  }

  augment "/ex:devices/ex:device/ex:interface" {
    leaf mtu {
      type uint16;
      default 1500;
    }
  }
}
```

in this example we define `common-interface-config` it has a `container interface` and then we add a `devices` container containing a list of `device` each `device` uses the grouping and finally we augment the `interface` container inside the `device` with an `mtu` leaf the important thing to notice is that we use the xpath `/ex:devices/ex:device/ex:interface` to point to the container to be modified

if your interface container needs a bit more customisation say a speed config only on some devices then we can augment it with other nodes with specific conditions

```yang
module example-module-2 {
  namespace "urn:example:example-module-2";
  prefix ex2;

  grouping common-interface-config {
    container interface {
      leaf name {
        type string;
      }
      leaf admin-status {
        type boolean;
        default true;
      }
    }
  }

  container devices {
    list device {
      key "name";
      leaf name {
        type string;
      }
      uses common-interface-config;
    }
  }

  augment "/ex2:devices/ex2:device/ex2:interface" {
     leaf speed {
       type enumeration {
         enum "100M";
         enum "1G";
         enum "10G";
       }
       default "1G";
     }
    when "/ex2:devices/ex2:device/ex2:name = 'specialDeviceA'";
  }
}
```

see the `when` statement it's the key to selective augmentation now only devices named `specialDeviceA` will have this speed configuration that is very powerful if you just want specific customization per devices

**Using `refine`**

Now `refine` is a different kind of beast it's used to fine tune the existing schema of a data node it does not add new children to it but changes properties of the ones already existing think of it as refining a recipe you might tweak the amount of sugar or change the cooking time it's subtle but important for example lets say that for one device we want the `admin-status` to not be changeable via configuration and we want to set the default to false

```yang
module example-module-3 {
  namespace "urn:example:example-module-3";
  prefix ex3;

  grouping common-interface-config {
    container interface {
      leaf name {
        type string;
      }
      leaf admin-status {
        type boolean;
        default true;
      }
    }
  }

  container devices {
    list device {
      key "name";
      leaf name {
        type string;
      }
      uses common-interface-config;
    }
  }

  refine "/ex3:devices/ex3:device/ex3:interface/ex3:admin-status" {
    config false;
     default false;
     description "Administrative status default is false";

  }
}
```

so here `refine` statement is used to set the admin-status as `config false` and change the default value to false with an additional description this way when someone is using the model they get an idea of why the configuration is not being changeable

**Important Considerations and a random joke**

It is important to understand how xpath works you need to know the exact location in the data tree to augment or refine it because that is how YANG works it is a tree structure at its core I was once debugging a model late night it was the culmination of a month long work and a wrong xpath made me go insane I kept looking at it for an hour thinking I had made a typo in the augmentation and it turned out I did but in the path I was using it and after I fixed it I just laughed at it out loud maybe I should have gone to bed earlier that night huh

another thing is `augment` statements can be spread across multiple yang files making it harder to track down what's going on it is good practice to have all augmentation related to a specific grouping to be in the same yang file to help with the organization of your models

in terms of resources to get better at this I would recommend `RFC 7950` that's the YANG spec it's the bible you should have it at hand it will clarify many things that are not clear from examples only then also look into the `YANG Data Modeling for Network Configuration` by Benoit Claise and Joe Clarke it is a great book for the more advance stuff

so yeah that is pretty much it augment and refine are your friends when you need to extend and tweak the containers of a grouping use them wisely and you'll avoid data model hell hope that clears things up good luck coding out there
