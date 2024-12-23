---
title: "How do I obtain a generic PsiClassReferenceType?"
date: "2024-12-23"
id: "how-do-i-obtain-a-generic-psiclassreferencetype"
---

Alright, let’s delve into the intricacies of obtaining a generic `PsiClassReferenceType`, a task I've encountered more times than I care to count across various Java plugin development endeavors over the years. Specifically, it’s a common hurdle when you're deeply involved with static code analysis or code generation features within IntelliJ plugin development. Obtaining a `PsiClassReferenceType` that encapsulates a generic type isn’t always straightforward. Let’s walk through it, focusing on practical approaches rather than theoretical abstractions.

The fundamental challenge here lies in the way generics are represented within the Psi (Program Structure Interface) model of IntelliJ. Generics exist as type parameters, and these type parameters are effectively placeholders. A `PsiClassReferenceType`, without any manipulation, represents the raw type (e.g., `List` instead of `List<String>`). What we need is a way to weave in the specific generic type arguments.

My initial encounters with this were largely related to custom intention actions. I was building an action that needed to introspect a class's method signatures, particularly focusing on methods with collections of parameterized types, and that's where the complexity started. I quickly learned that directly creating `PsiClassReferenceType` with the `<>` syntax isn't something the API directly supports. Rather, it's about manipulating existing or creating new `PsiTypes` correctly.

The core mechanism revolves around the `PsiSubstitutor` class. A `PsiSubstitutor` is, in essence, a map of type parameters to their concrete substitutions, and it's the key ingredient to creating a `PsiType` reflecting a generic type. The challenge then becomes: where do we start?

Let’s break it down with code examples:

**Example 1: Obtaining a Generic Type from Existing PsiElements**

This scenario arises when you're working with an existing `PsiType`. Let's say you're analyzing a method parameter, and it’s declared as `List<String>`.

```java
import com.intellij.openapi.project.Project;
import com.intellij.psi.*;
import com.intellij.psi.search.GlobalSearchScope;

public class GenericTypeExample {

    public static PsiClassReferenceType obtainGenericListType(PsiMethod method, Project project) {
        if (method == null) return null;

        // Let's assume the method has a parameter of type List<String>
        PsiParameter[] parameters = method.getParameterList().getParameters();
        if(parameters.length == 0) return null;

        PsiType parameterType = parameters[0].getType();

        if (!(parameterType instanceof PsiClassType)) {
            return null; // Not a class type
        }

        PsiClassType classType = (PsiClassType) parameterType;
        PsiClass resolvedClass = classType.resolve();

        if (resolvedClass == null || !resolvedClass.getQualifiedName().equals("java.util.List")) {
           return null; // Not a List, or not resolvable.
        }

        PsiType[] typeArguments = classType.getParameters(); // these ARE the type parameters
        if(typeArguments.length == 0) return null;

        // Let's say we want the List<String> type.
        // We just need to check if the first argument is a PsiClassType.
        // The parameter here will be a PsiClassType.
        if(!(typeArguments[0] instanceof PsiClassType)){
            return null; // We're expecting class type.
        }
        PsiClassType argumentClassType = (PsiClassType) typeArguments[0];

        if(argumentClassType.resolve() == null || !argumentClassType.resolve().getQualifiedName().equals("java.lang.String")){
           return null; // First parameter not a String
        }

        return classType; // Here we've correctly identified List<String>.

    }
}
```

In this example, we extract the `PsiType` from an existing element, in this case, a parameter. We then verify it is indeed a `PsiClassType` and that it’s resolvable to a `java.util.List`. Crucially, the `getParameters()` method gets us the generic arguments (e.g. `String` in `List<String>`). From here we can do additional checks, as I've shown, to ensure it matches what we're after, and it can be returned as is. It's worth mentioning that if we needed a *new* `PsiClassReferenceType` with different parameterizations, we'd need a different approach.

**Example 2: Creating a Generic Type from Scratch Using a `PsiSubstitutor`**

Now, consider you want to generate a method or class using the PSI model. You need to create a generic type, say, `Map<Integer, String>`. This is where building a `PsiSubstitutor` comes into play.

```java
import com.intellij.openapi.project.Project;
import com.intellij.psi.*;
import com.intellij.psi.search.GlobalSearchScope;

public class GenericTypeCreation {

     public static PsiClassReferenceType createGenericMapType(Project project) {
        if (project == null) return null;

        PsiClass mapClass = JavaPsiFacade.getInstance(project).findClass("java.util.Map", GlobalSearchScope.allScope(project));
        if (mapClass == null) return null;

        PsiClass integerClass = JavaPsiFacade.getInstance(project).findClass("java.lang.Integer", GlobalSearchScope.allScope(project));
        if(integerClass == null) return null;
        PsiClass stringClass = JavaPsiFacade.getInstance(project).findClass("java.lang.String", GlobalSearchScope.allScope(project));
        if(stringClass == null) return null;

        PsiType integerType = PsiType.getType(integerClass);
        PsiType stringType = PsiType.getType(stringClass);

        PsiTypeParameter[] typeParameters = mapClass.getTypeParameters();

        if(typeParameters.length != 2) return null;

        PsiSubstitutor substitutor = PsiSubstitutor.EMPTY
                .put(typeParameters[0], integerType)
                .put(typeParameters[1], stringType);


        return JavaPsiFacade.getElementFactory(project).createType(mapClass,substitutor);

    }
}
```

Here, we locate the `java.util.Map` class. Then, we find the concrete types `Integer` and `String`. We create `PsiTypes` for them. We then retrieve the type parameters from the `Map` class, ensuring that we have two (for the key and value).  A new `PsiSubstitutor` is created, and type parameters are associated with the desired concrete types. Finally, we generate a new `PsiClassReferenceType` with the substitutor, effectively creating `Map<Integer, String>`. This approach works when building Psi elements programmatically.

**Example 3: Working with Wildcards**

Sometimes you need to obtain a type containing wildcards such as `List<? extends Number>`.  This requires slightly more finesse, leveraging the `PsiWildcardType`.

```java
import com.intellij.openapi.project.Project;
import com.intellij.psi.*;
import com.intellij.psi.search.GlobalSearchScope;

public class WildcardTypeExample {

    public static PsiClassReferenceType createWildcardListType(Project project) {
        if (project == null) return null;
       PsiClass listClass = JavaPsiFacade.getInstance(project).findClass("java.util.List", GlobalSearchScope.allScope(project));
        if(listClass == null) return null;

        PsiClass numberClass = JavaPsiFacade.getInstance(project).findClass("java.lang.Number",GlobalSearchScope.allScope(project));
        if(numberClass == null) return null;

        PsiType numberType = PsiType.getType(numberClass);

        PsiWildcardType wildcardType = PsiWildcardType.createExtends(PsiManager.getInstance(project), numberType);

        PsiTypeParameter[] typeParameters = listClass.getTypeParameters();
        if(typeParameters.length != 1) return null;

        PsiSubstitutor substitutor = PsiSubstitutor.EMPTY
                .put(typeParameters[0], wildcardType);


        return JavaPsiFacade.getElementFactory(project).createType(listClass, substitutor);
    }
}
```
Here, we find both `List` and `Number`. Then we construct a `PsiWildcardType` indicating `? extends Number`. Then, similar to the `Map` example we use a substitutor to place the wildcard into the generic `List` type. The result is `List<? extends Number>` as a `PsiClassReferenceType`.

**Recommendations**

For a deep dive into the IntelliJ platform and PSI, I highly recommend reading "IntelliJ Platform SDK: Building Plugins for IntelliJ IDEA and Other IDEs," by Dmitry Jemerov and Gregory Shrago (currently only available as online documentation from Jetbrains - https://plugins.jetbrains.com/docs/intellij/welcome.html). It provides a thorough understanding of all the underlying concepts. In addition to that, exploring the source code for other JetBrains plugins, available on GitHub, can be extremely helpful. Pay specific attention to projects with code generation or analysis functionality. Also, “Effective Java,” by Joshua Bloch, isn’t specific to IntelliJ but offers a great understanding of Java generics themselves.

**Final Thoughts**

Navigating the nuances of `PsiClassReferenceType` with generics can indeed be challenging, but the IntelliJ platform offers rich tools for that. Understanding how to use `PsiSubstitutor` effectively is crucial for advanced plugin development. By following these patterns, I found it possible to build robust and insightful plugins. This detailed explanation should provide you with the tools to tackle this issue successfully. Remember that practice and working through actual use cases are the best ways to solidify these concepts.
