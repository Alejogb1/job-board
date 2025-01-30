---
title: "How do I define a microk8s controller filter?"
date: "2025-01-30"
id: "how-do-i-define-a-microk8s-controller-filter"
---
Defining a custom controller filter within the microk8s environment necessitates a deep understanding of Kubernetes' controller runtime and the specific mechanism by which microk8s handles resource management.  My experience implementing custom controllers for high-availability database deployments in microk8s environments highlights the importance of precise predicate definition and efficient resource utilization.  Simply put, a filter, in this context, acts as a conditional gatekeeper, influencing which resources a controller processes.

**1.  Clear Explanation**

A microk8s controller, like any Kubernetes controller, operates on a watch-process-update loop. It continuously observes changes in the Kubernetes API server, processes relevant events, and updates the cluster state accordingly.  To limit the scope of a controller's actions, we employ filters. These filters are implemented using predicates within the controller's reconciliation loop.  The predicate is a function that evaluates the incoming resource and returns `true` if the controller should handle it, and `false` otherwise. This allows for selective processing of resources, enhancing efficiency and preventing unintended modifications.

The design of an effective predicate demands careful consideration of several factors:  resource labels, annotations, namespaces, and specific resource fields.  Efficient predicates avoid unnecessary computations, ensuring the controller remains responsive under high loads.  For example, a predicate solely focused on a specific label reduces the processing overhead significantly compared to a predicate that examines multiple fields.  This is crucial in a constrained environment like microk8s where resource availability might be limited.

In microk8s, the implementation largely mirrors standard Kubernetes controller development.  However, the constrained nature of the environment underscores the need for lean and optimized controllers. Overly complex predicates can impact performance negatively.  My past experience with resource-intensive controllers highlighted the importance of profiling and optimization techniques to mitigate these performance bottlenecks.

**2. Code Examples with Commentary**

The following examples demonstrate three distinct ways to define predicates in a microk8s controller using the controller-runtime library.  All examples assume familiarity with Go and the controller-runtime framework.

**Example 1: Label-based Filtering**

This example demonstrates a simple predicate that filters resources based on a specific label.

```go
package main

import (
	"context"
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// MyPredicate implements a predicate that only processes resources with the label "app: myapp"
func MyPredicate(obj metav1.Object) bool {
	labels := obj.GetLabels()
	if labels == nil {
		return false
	}
	return labels["app"] == "myapp"
}

// ... (rest of the controller code)

// SetupWithManager sets up the controller with the manager
func (r *MyReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&MyResource{}).
		WithEventFilter(predicate.Funcs{
			CreateFunc: MyPredicate,
			UpdateFunc: MyPredicate,
			DeleteFunc: MyPredicate,
		}).
		Complete(r)
}
```

This code utilizes `predicate.Funcs` to define a predicate for create, update, and delete events.  Only resources with the label `app: myapp` will trigger the controller's reconciliation logic. The absence of labels results in the predicate returning `false`, effectively filtering out those resources.  This approach is straightforward and highly efficient for label-based filtering.


**Example 2: Namespace-based Filtering**

This example showcases a predicate filtering resources based on their namespace.

```go
package main

import (
	"context"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// NamespacePredicate filters resources based on the specified namespace
func NamespacePredicate(namespace string) predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(obj metav1.Object) bool {
			return obj.GetNamespace() == namespace
		},
		UpdateFunc: func(obj metav1.Object) bool {
			return obj.GetNamespace() == namespace
		},
		DeleteFunc: func(obj metav1.Object) bool {
			return obj.GetNamespace() == namespace
		},
	}
}

// ... (rest of the controller code)

// SetupWithManager sets up the controller with the manager
func (r *MyReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&MyResource{}).
		WithEventFilter(NamespacePredicate("mynamespace")).
		Complete(r)
}
```

Here, `NamespacePredicate` creates a reusable function that returns a `predicate.Funcs` tailored to a specific namespace.  This approach improves code maintainability and reusability.  The controller will only react to events originating from the "mynamespace" namespace.  This is particularly useful for isolating controllers to specific sections of the cluster.


**Example 3:  Combined Field and Annotation Filtering**

This more complex example combines label and annotation checks for more refined filtering.

```go
package main

import (
	"context"
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// ComplexPredicate combines label and annotation checks
func ComplexPredicate(obj metav1.Object) bool {
	labels := obj.GetLabels()
	annotations := obj.GetAnnotations()

	if labels == nil || annotations == nil {
		return false
	}

	return labels["app"] == "myapp" && annotations["managedBy"] == "mycontroller"
}

// ... (rest of the controller code)


// SetupWithManager sets up the controller with the manager
func (r *MyReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&MyResource{}).
		WithEventFilter(predicate.Funcs{
			CreateFunc: ComplexPredicate,
			UpdateFunc: ComplexPredicate,
			DeleteFunc: ComplexPredicate,
		}).
		Complete(r)
}
```

This predicate showcases a more intricate filtering mechanism, demonstrating the flexibility of the approach.  It only processes resources with the label `app: myapp` and the annotation `managedBy: mycontroller`. This allows for fine-grained control over which resources the controller manages.  The added complexity, however, necessitates careful consideration of performance implications.



**3. Resource Recommendations**

For a comprehensive understanding of Kubernetes controllers and the controller-runtime library, I strongly recommend consulting the official Kubernetes documentation and the controller-runtime project documentation.  Furthermore, studying existing Kubernetes controllers can provide valuable insight into practical implementation techniques and best practices.  Finally, leveraging profiling tools to analyze controller performance is invaluable for identifying and mitigating performance bottlenecks.  Thorough unit and integration testing should be an integral part of your development process to ensure controller robustness and reliability.
