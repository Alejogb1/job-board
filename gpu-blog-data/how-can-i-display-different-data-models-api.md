---
title: "How can I display different data models (API responses) in SwiftUI detail views?"
date: "2025-01-30"
id: "how-can-i-display-different-data-models-api"
---
SwiftUI’s declarative nature necessitates a thoughtful approach when handling diverse data models, particularly within detail views. Attempting to directly render varying API response structures using a single SwiftUI view designed for one model type will invariably lead to runtime errors or inconsistent user experiences. I have, over several years of iOS development, encountered this scenario frequently, particularly in projects that integrated multiple third-party APIs. The key to addressing this challenge lies in employing abstraction and conditional logic to decouple the view’s structure from the specific data being presented.

The core problem arises from Swift's strong typing. SwiftUI views expect properties of a specific type. Directly binding properties from different, unrelated data models to the same UI element results in type mismatches. Therefore, the solution involves creating a mechanism that allows the detail view to adapt to varying data shapes, while still maintaining type safety and a predictable UI. This mechanism fundamentally revolves around protocols and generics.

Let's consider a hypothetical scenario where our application interacts with two APIs, one for retrieving book details and another for fetching author information. The book API returns a JSON structure that can be modeled as a `Book` struct:

```swift
struct Book: Decodable, Identifiable {
    let id: Int
    let title: String
    let author: String
    let publicationYear: Int
}
```

The author API returns a structure that can be represented by an `Author` struct:

```swift
struct Author: Decodable, Identifiable {
    let id: Int
    let name: String
    let biography: String
    let birthYear: Int
}
```

A naive approach would be to create separate detail views for `Book` and `Author`. This leads to code duplication and a maintenance nightmare if we introduce additional API responses in the future. A superior approach uses an abstraction, leveraging a protocol.

We can define a `DetailDisplayable` protocol:

```swift
protocol DetailDisplayable: Identifiable {
    var detailView: AnyView { get }
}
```

This protocol mandates that any data model conforming to it must provide a method to return an `AnyView`, which encapsulates the specific SwiftUI view responsible for rendering that model. This introduces the necessary level of abstraction. We extend `Book` and `Author` to conform to this protocol:

```swift
extension Book: DetailDisplayable {
    var detailView: AnyView {
        AnyView(
            VStack(alignment: .leading) {
                Text(title).font(.headline)
                Text("By: \(author)").font(.subheadline)
                Text("Published: \(publicationYear)")
            }
        )
    }
}

extension Author: DetailDisplayable {
    var detailView: AnyView {
        AnyView(
            VStack(alignment: .leading) {
                Text(name).font(.headline)
                Text("Born: \(birthYear)").font(.subheadline)
                Text(biography)
            }
        )
    }
}
```

Each implementation returns an `AnyView` containing the correct rendering logic for its respective struct. This enables us to use a single `DetailView` that can dynamically render different data models.

Now, we define our generic `DetailView`:

```swift
struct DetailView<Model: DetailDisplayable>: View {
    let model: Model

    var body: some View {
        model.detailView
            .padding()
    }
}
```

The `DetailView` is now generic, accepting any type that conforms to `DetailDisplayable`. Inside the `body`, it simply accesses the `detailView` property of the provided model and renders it. The padding is simply for better visual presentation.

To demonstrate the usage of this structure, assume that `book` is a variable of type `Book`, and `author` is a variable of type `Author`, both populated with data. You could instantiate the `DetailView` like this:

```swift
let book = Book(id: 1, title: "The Hitchhiker's Guide to the Galaxy", author: "Douglas Adams", publicationYear: 1979)
let author = Author(id: 2, name: "Jane Austen", biography: "English novelist known primarily for her six major novels...", birthYear: 1775)

struct ContentView: View {
    var body: some View {
        VStack{
            NavigationStack {
               NavigationLink {
                   DetailView(model: book)
                } label: {
                    Text("Show Book")
                }
               NavigationLink {
                    DetailView(model: author)
                } label: {
                   Text("Show Author")
                }
            }
        }

    }
}
```

This `ContentView` demonstrates how to navigate to two different detail views, one displaying a book and one displaying an author, using the same `DetailView` structure. The compiler ensures that the correct view is generated based on the type of `model`.

The `AnyView` wrapper can incur a small performance overhead as it type erases SwiftUI views. However, in most use cases involving relatively small and non-frequently updating detail views, the benefit of code reuse and maintainability far outweighs this cost. If performance becomes a critical concern for very complex and frequently redrawing detail views, consider exploring techniques like custom `View` structs, which utilize `@ViewBuilder` to dynamically construct more performant views, tailored to each model type, while still maintaining a separation of concern.

This protocol-based approach provides a flexible framework. Introducing new API models only requires defining their structure, and conforming to the `DetailDisplayable` protocol. The `DetailView` itself remains untouched and does not need to be modified.

For developers looking to expand on this topic, research the following areas in the Apple documentation: protocols, generics, `AnyView`, `ViewBuilder`, and type erasure. Additional resources from reputable iOS development blogs or training platforms often have practical demonstrations of these concepts. I suggest focusing on material that highlights the separation of UI and data handling concerns. Also, consider exploring more advanced SwiftUI techniques, such as the use of conditional views within the `detailView` property to handle variations within the data models themselves, and the use of property wrappers like `@State` and `@ObservedObject` if any user interaction with the displayed content is necessary.
