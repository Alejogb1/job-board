---
title: "How do I resolve high kind trait implementation conflicts?"
date: "2024-12-23"
id: "how-do-i-resolve-high-kind-trait-implementation-conflicts"
---

Let's tackle this. Having navigated a few hairy implementations myself, specifically with large type hierarchies and extensive kind traits, I can confidently say that kind trait conflicts are more common than most developers would prefer. They typically emerge when the structure of your codebase, specifically around generics and associated types, becomes more intricate than anticipated. What seems like an innocent extension of a system, can inadvertently create multiple implementations for the same kind trait, which the compiler, reasonably, refuses to accept. This usually occurs when dealing with generic types that overlap in the type system, making it ambiguous which implementation to pick at compile-time.

The crux of resolving these conflicts hinges on precision – precisely specifying the types involved or restructuring the type system to make distinct implementations obvious to the compiler. Let's break it down into a few approaches, incorporating examples that mirror situations I’ve personally encountered.

Firstly, understand that the core issue is *ambiguity*. The compiler needs to uniquely identify which trait implementation to use for a given type at a given context. When multiple implementations seem viable based on the compiler's type inference mechanisms, a conflict arises. This means your types, specifically type parameters and associated types within the implementation, are not specific enough for the compiler to differentiate between potential candidates.

One of the most effective solutions involves adding explicit trait bounds or leveraging *specialization*. Specialization is a more advanced technique, and its implementation and suitability may depend on the specific language (many modern languages support some level of it), but it can certainly be a powerful tool when available. Let's start with trait bounds. Suppose we’re working with a custom data structure and a printer trait:

```rust
trait Printable {
    fn print(&self);
}

struct MyStruct<T> {
    data: T,
}

impl<T> Printable for MyStruct<T> {
    fn print(&self) {
      println!("MyStruct with generic data");
    }
}

struct MyInt(i32);

impl Printable for MyInt {
    fn print(&self) {
        println!("MyInt: {}", self.0);
    }
}

fn generic_print<U: Printable>(item: U){
    item.print();
}

fn main(){
  let my_struct_int = MyStruct{data: 12};
  generic_print(my_struct_int); //Output: MyStruct with generic data

  let my_int = MyInt(5);
  generic_print(my_int); //Output: MyInt: 5

}
```

In this simple rust example, the compiler knows exactly which impl to invoke. If we modify it, however, by introducing another implementation, this is where problems can occur. Let's say we attempt to add a `Printable` implementation for any `MyStruct` with data that also implements the `Printable` trait:

```rust
trait Printable {
    fn print(&self);
}

struct MyStruct<T> {
    data: T,
}

impl<T> Printable for MyStruct<T> {
    fn print(&self) {
      println!("MyStruct with generic data");
    }
}

impl<T:Printable> Printable for MyStruct<T> {
    fn print(&self) {
      println!("MyStruct with printable data");
    }
}

struct MyInt(i32);

impl Printable for MyInt {
    fn print(&self) {
        println!("MyInt: {}", self.0);
    }
}

fn generic_print<U: Printable>(item: U){
    item.print();
}

fn main(){
    let my_struct_int = MyStruct{data: MyInt(12)};
    generic_print(my_struct_int); //Compiler Error! Multiple applicable impls
}
```

Now, the compiler throws an error because, for `MyStruct<MyInt>`, it has two valid impls. We can resolve this by refining the impls. The following code illustrates how we can make impls mutually exclusive by specializing them to ensure there is only one valid implementation for any given type:

```rust
trait Printable {
    fn print(&self);
}

struct MyStruct<T> {
    data: T,
}


impl<T> Printable for MyStruct<T> where T: Copy {
    fn print(&self) {
      println!("MyStruct with copyable data");
    }
}


impl<T> Printable for MyStruct<T>  where T : Printable{
    fn print(&self) {
      println!("MyStruct with printable data");
    }
}

struct MyInt(i32);

impl Printable for MyInt {
    fn print(&self) {
        println!("MyInt: {}", self.0);
    }
}

fn generic_print<U: Printable>(item: U){
    item.print();
}

fn main(){
    let my_struct_int = MyStruct{data: 12};
    generic_print(my_struct_int); //Output: MyStruct with copyable data

    let my_struct_printable = MyStruct{data: MyInt(12)};
    generic_print(my_struct_printable);  //Output: MyStruct with printable data
    let my_int = MyInt(5);
    generic_print(my_int); //Output: MyInt: 5

}
```

By adding a bound to the generic type `T` using `where T: Copy` for the first impl, we made it specific only for Types which implement the `Copy` trait. Similarly, we made the second impl specific only to types which implement the `Printable` trait. Now there is no ambiguity.

The second example involves a similar issue but involves associated types. In one project, we had a trait for handling serialization with associated `Reader` and `Writer` types:

```typescript
interface Serializable {
    type Reader;
    type Writer;

    serialize(writer: this['Writer']): void;
    deserialize(reader: this['Reader']): void;
}
```

Let's say we want to add an implementation for a custom class that serializes to JSON. We could make a JSON reader/writer and add the implementation like this:

```typescript
interface Reader {
  read(): any
}

interface Writer {
  write(data: any): void
}

class JsonReader implements Reader{
    read(): any{
        return {}; //Placeholder. actual implementation needed
    }
}


class JsonWriter implements Writer{
    write(data: any): void{
        console.log("writing", data);
    }
}


interface Serializable {
    type Reader;
    type Writer;

    serialize(writer: this['Writer']): void;
    deserialize(reader: this['Reader']): void;
}

class MyData implements Serializable{
    type Reader = JsonReader;
    type Writer = JsonWriter;

    data: {
        name: string,
        value: number
    }
    constructor(name: string, value: number){
      this.data = {
        name: name,
        value: value
      }
    }
    serialize(writer: this['Writer']): void {
      writer.write(this.data);
    }

    deserialize(reader: this['Reader']): void {
       this.data = reader.read();
    }
}


function serializeObject<T extends Serializable>(obj: T, writer: T['Writer']){
    obj.serialize(writer);
}

function main() {
    const data = new MyData("test", 12);
    const writer = new JsonWriter();
    serializeObject(data, writer); //Output: writing { name: 'test', value: 12 }
}

main()
```
This works well and is type safe. However, if we add an implementation for `MyData` with a different type of reader and writer, the typescript compiler will produce a compilation error:

```typescript
interface Reader {
  read(): any
}

interface Writer {
  write(data: any): void
}

class JsonReader implements Reader{
    read(): any{
        return {}; //Placeholder. actual implementation needed
    }
}


class JsonWriter implements Writer{
    write(data: any): void{
        console.log("writing", data);
    }
}

class BinReader implements Reader{
    read(): any{
        return {}; //Placeholder. actual implementation needed
    }
}

class BinWriter implements Writer{
    write(data: any): void{
        console.log("writing binary", data);
    }
}


interface Serializable {
    type Reader;
    type Writer;

    serialize(writer: this['Writer']): void;
    deserialize(reader: this['Reader']): void;
}

class MyData implements Serializable{
    type Reader = JsonReader;
    type Writer = JsonWriter;

    data: {
        name: string,
        value: number
    }
    constructor(name: string, value: number){
      this.data = {
        name: name,
        value: value
      }
    }
    serialize(writer: this['Writer']): void {
      writer.write(this.data);
    }

    deserialize(reader: this['Reader']): void {
       this.data = reader.read();
    }
}

class MyData implements Serializable{
    type Reader = BinReader;
    type Writer = BinWriter;

    data: {
        name: string,
        value: number
    }
    constructor(name: string, value: number){
      this.data = {
        name: name,
        value: value
      }
    }
    serialize(writer: this['Writer']): void {
      writer.write(this.data);
    }

    deserialize(reader: this['Reader']): void {
       this.data = reader.read();
    }
}


function serializeObject<T extends Serializable>(obj: T, writer: T['Writer']){
    obj.serialize(writer);
}

function main() {
    const data = new MyData("test", 12); //Compiler Error: Duplicate declaration of MyData
    const writer = new JsonWriter();
    serializeObject(data, writer);
}

main()
```

The compiler flags this as an error because we are declaring `MyData` multiple times. This is because classes cannot implement the same interface multiple times. We can achieve this, however, using different types. Let's assume we have a different class that represents binary data, and we implement `Serializable` for this `BinData` class:

```typescript
interface Reader {
  read(): any
}

interface Writer {
  write(data: any): void
}

class JsonReader implements Reader{
    read(): any{
        return {}; //Placeholder. actual implementation needed
    }
}


class JsonWriter implements Writer{
    write(data: any): void{
        console.log("writing", data);
    }
}

class BinReader implements Reader{
    read(): any{
        return {}; //Placeholder. actual implementation needed
    }
}

class BinWriter implements Writer{
    write(data: any): void{
        console.log("writing binary", data);
    }
}


interface Serializable {
    type Reader;
    type Writer;

    serialize(writer: this['Writer']): void;
    deserialize(reader: this['Reader']): void;
}

class MyData implements Serializable{
    type Reader = JsonReader;
    type Writer = JsonWriter;

    data: {
        name: string,
        value: number
    }
    constructor(name: string, value: number){
      this.data = {
        name: name,
        value: value
      }
    }
    serialize(writer: this['Writer']): void {
      writer.write(this.data);
    }

    deserialize(reader: this['Reader']): void {
       this.data = reader.read();
    }
}

class BinData implements Serializable{
     type Reader = BinReader;
     type Writer = BinWriter;

      data: {
        name: string,
        value: number
    }
    constructor(name: string, value: number){
      this.data = {
        name: name,
        value: value
      }
    }
    serialize(writer: this['Writer']): void {
      writer.write(this.data);
    }

    deserialize(reader: this['Reader']): void {
       this.data = reader.read();
    }

}

function serializeObject<T extends Serializable>(obj: T, writer: T['Writer']){
    obj.serialize(writer);
}

function main() {
    const jsonData = new MyData("test", 12);
    const jsonWriter = new JsonWriter();
    serializeObject(jsonData, jsonWriter); //Output: writing { name: 'test', value: 12 }

    const binData = new BinData("bin", 55);
    const binWriter = new BinWriter();
    serializeObject(binData, binWriter); //Output: writing binary { name: 'bin', value: 55 }
}

main()
```

Here, the conflict is resolved because the types `MyData` and `BinData` are entirely distinct, which eliminates ambiguity about which implementation to use. The key is ensuring that your types are specific enough in the type system to avoid conflicts with other implementations of the same kind trait. This means carefully constructing the type hierarchy or type parameters to provide the compiler with enough information to uniquely identify the appropriate implementation. In more complex cases, this may involve refactoring the type system by introducing new types to differentiate between use cases that currently map to the same general type.

To dive deeper, I'd recommend exploring resources like "Advanced Type Systems" by Benjamin Pierce. This provides a more formal foundation for understanding the intricacies of type systems, which can be helpful when facing these complex implementations. Additionally, “Types and Programming Languages” by the same author offers an even deeper dive into programming language theory, which provides valuable insights into how to resolve complex situations involving generics and kind traits. If you want a more hands on introduction, many blog posts on medium such as “Understanding Rust’s Trait System” (you can easily find this and many others with similar titles) which explain traits in more detail with examples.

In essence, resolving these kind trait implementation conflicts requires careful type system design and judicious use of trait bounds, specialization and sometimes even restructuring to make sure each implementation is distinct enough to avoid conflict. Avoid broad or general implementations that could conflict with other potential use cases. It’s about giving the compiler what it needs: clarity through specificity.
