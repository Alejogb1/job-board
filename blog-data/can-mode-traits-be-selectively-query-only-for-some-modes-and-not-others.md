---
title: "Can mode traits be selectively query-only for some modes and not others?"
date: "2024-12-23"
id: "can-mode-traits-be-selectively-query-only-for-some-modes-and-not-others"
---

Let’s tackle this. I've navigated similar scenarios with complex data modeling more times than I care to remember, usually involving systems where data access patterns differ significantly based on the context. The specific question of selective query-only mode traits—where certain attributes are readable but not modifiable in some modes, while being fully read/write in others—is a nuanced problem that frequently crops up when dealing with application state management or data integrity considerations.

It's not simply about having read-only properties in the traditional sense, because the core concept we’re talking about is that this trait is *mode dependent*. Consider, for example, a user profile object. In ‘view’ mode, you might want to allow users to see all their information, such as phone number and address. In 'edit' mode, these are open for modification. But then in an 'admin review' mode, those fields should be visible for verification, yet strictly not editable. Let me explain how this kind of behavior is typically implemented.

Essentially, the key is decoupling the data representation from the access control policies. We don't directly mark data fields as query-only within the data structure itself. Instead, we leverage intermediary access control layers and apply conditional logic based on the active mode. This architecture allows the underlying data object to remain agnostic of specific access control constraints.

The most common method uses a variation of the adapter pattern or a facade pattern to handle this conditional access. The primary data object, often represented as a plain old data structure (pod) or a more encapsulated class, stores all the necessary attributes, regardless of access rights. A separate layer, often called the *access controller*, *data service*, or *repository* acts as the gatekeeper. This gatekeeper then determines, based on the current mode, whether a particular attribute can be read, modified, or both.

Here's an illustration through code using typescript. While I prefer c++ for more low-level work, typescript is excellent for illustrating complex logic without the noise of memory management:

```typescript
interface UserData {
    userId: number;
    username: string;
    email: string;
    phoneNumber: string;
    address: string;
}

enum Mode {
  View,
  Edit,
  AdminReview
}

class User {
  private data: UserData;
  private mode: Mode = Mode.View;

    constructor(initialData: UserData){
      this.data = initialData;
    }

    setMode(mode: Mode) {
        this.mode = mode;
    }

    getUsername(): string {
       return this.data.username;
    }

    getEmail(): string {
        return this.data.email;
    }

    getPhoneNumber(): string{
         return this.data.phoneNumber;
    }

    getAddress(): string {
        return this.data.address
    }

    setUsername(newUsername: string): void {
      if(this.mode === Mode.Edit){
        this.data.username = newUsername;
      } else {
          throw new Error("Cannot modify username in current mode.");
      }
    }

   setEmail(newEmail: string): void {
       if(this.mode === Mode.Edit)
       {
            this.data.email = newEmail;
       }
       else {
          throw new Error("Cannot modify email in current mode.");
       }
   }


  setPhoneNumber(newPhoneNumber: string): void {
     if(this.mode === Mode.Edit)
     {
          this.data.phoneNumber = newPhoneNumber;
      }
      else {
         throw new Error("Cannot modify phoneNumber in current mode.");
     }
  }


    setAddress(newAddress: string): void {
       if(this.mode === Mode.Edit)
       {
        this.data.address = newAddress;
       }
        else {
          throw new Error("Cannot modify address in current mode.");
       }
   }

}
```
In this example, the `User` class encapsulates the `UserData`. The mode is stored as a class variable and can be switched using the `setMode` method. Getter methods provide access to the data in any mode, whereas setter methods for username, email, phonenumber and address are conditionally executed based on mode, using if statements for each method.

A more structured, extensible, and less repetitive method uses a configuration map to define what operations are permitted in each mode. This removes the need for specific if checks in every setter. Here's how to modify the example, keeping it in typescript for clarity:

```typescript
interface UserData {
    userId: number;
    username: string;
    email: string;
    phoneNumber: string;
    address: string;
}

enum Mode {
    View,
    Edit,
    AdminReview
}

type Permissions = {
  [key in Mode]: { [field in keyof UserData]?: 'read' | 'write'};
}

class User {
  private data: UserData;
  private mode: Mode = Mode.View;

  private permissions: Permissions = {
      [Mode.View]: {
        username: 'read',
        email: 'read',
        phoneNumber: 'read',
        address: 'read'
      },
      [Mode.Edit]: {
        username: 'write',
        email: 'write',
        phoneNumber: 'write',
        address: 'write'

      },
      [Mode.AdminReview]: {
        username: 'read',
         email: 'read',
         phoneNumber: 'read',
         address: 'read'
      }
  };


    constructor(initialData: UserData){
      this.data = initialData;
    }

    setMode(mode: Mode) {
        this.mode = mode;
    }

    getUsername(): string {
       return this.data.username;
    }

    getEmail(): string {
        return this.data.email;
    }

    getPhoneNumber(): string{
         return this.data.phoneNumber;
    }

    getAddress(): string {
        return this.data.address
    }

    setUsername(newUsername: string): void {
        this.checkPermission('username', 'write');
        this.data.username = newUsername;
    }

   setEmail(newEmail: string): void {
       this.checkPermission('email', 'write');
       this.data.email = newEmail;
   }


  setPhoneNumber(newPhoneNumber: string): void {
     this.checkPermission('phoneNumber', 'write');
     this.data.phoneNumber = newPhoneNumber;
  }

  setAddress(newAddress: string): void {
       this.checkPermission('address', 'write');
       this.data.address = newAddress;
    }

    private checkPermission(field: keyof UserData, operation: 'read' | 'write'){
        const modePermission = this.permissions[this.mode]?.[field];
        if (!modePermission || modePermission !== operation){
            throw new Error(`Permission denied for ${field} ${operation} in ${Mode[this.mode]}`);
        }

    }

}

```

In this version, a `permissions` object defines the valid read/write operations per mode per attribute. `checkPermission` is used to enforce the rules. This is much more flexible. If you need to add more fields or modes, its just a matter of adding to the enum and permissions object.

Another way to approach this, suitable in some scenarios, involves using a *proxy object*. Instead of directly interacting with the `UserData`, you interact with a proxy that handles permission checks.

```typescript
interface UserData {
    userId: number;
    username: string;
    email: string;
    phoneNumber: string;
    address: string;
}

enum Mode {
    View,
    Edit,
    AdminReview
}

type Permissions = {
  [key in Mode]: { [field in keyof UserData]?: 'read' | 'write'};
}


class UserProxy {
  private data: UserData;
  private mode: Mode = Mode.View;

    private permissions: Permissions = {
      [Mode.View]: {
        username: 'read',
        email: 'read',
        phoneNumber: 'read',
        address: 'read'
      },
      [Mode.Edit]: {
        username: 'write',
        email: 'write',
        phoneNumber: 'write',
        address: 'write'
      },
      [Mode.AdminReview]: {
        username: 'read',
         email: 'read',
         phoneNumber: 'read',
         address: 'read'
      }
    };

    constructor(initialData: UserData) {
      this.data = initialData;
    }

    setMode(mode: Mode) {
      this.mode = mode;
    }


    get username(): string {
      return this.data.username;
    }


   get email(): string {
      return this.data.email;
    }

    get phoneNumber(): string {
        return this.data.phoneNumber;
    }

     get address(): string {
        return this.data.address;
    }



    set username(newUsername: string) {
      this.checkPermission('username', 'write');
      this.data.username = newUsername;
    }

    set email(newEmail: string) {
         this.checkPermission('email', 'write');
        this.data.email = newEmail;
    }

   set phoneNumber(newPhoneNumber: string) {
       this.checkPermission('phoneNumber', 'write');
      this.data.phoneNumber = newPhoneNumber;
    }


     set address(newAddress: string) {
        this.checkPermission('address', 'write');
       this.data.address = newAddress;
    }


  private checkPermission(field: keyof UserData, operation: 'read' | 'write'){
        const modePermission = this.permissions[this.mode]?.[field];
        if (!modePermission || modePermission !== operation){
            throw new Error(`Permission denied for ${field} ${operation} in ${Mode[this.mode]}`);
        }

    }
}

```

Here, the `UserProxy` acts as an intermediary, intercepting all accesses to the properties. The `set` and `get` keywords allow for more intuitive interactions. You'll notice this approach uses getter and setter methods directly at the class level, improving readability. It also uses the same checkPermission methodology as before.

In terms of resources, consider exploring books and papers focused on access control models and object-oriented design. Specifically, "Design Patterns: Elements of Reusable Object-Oriented Software" by the Gang of Four remains a bedrock resource for understanding patterns like adapter, facade, and proxy. For a more formal treatment of access control, the Bell-LaPadula model paper (though somewhat dated) provides a good understanding of mandatory access controls, which form the basis of many practical solutions. I'd also recommend exploring literature on object capability models for more advanced access control techniques if you're dealing with very fine-grained permissions. I've used these techniques across countless projects and found that decoupling data from access logic always leads to cleaner, maintainable, and more robust applications.
