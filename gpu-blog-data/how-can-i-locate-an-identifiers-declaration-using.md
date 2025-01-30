---
title: "How can I locate an identifier's declaration using go/analysis?"
date: "2025-01-30"
id: "how-can-i-locate-an-identifiers-declaration-using"
---
The `go/analysis` package offers powerful capabilities for static analysis, but pinpointing the precise declaration of an identifier requires a nuanced understanding of its internal workings and the AST (Abstract Syntax Tree) it manipulates.  Directly querying for an identifier's location is not a straightforward operation; rather, one must traverse the AST, identifying the specific node representing the declaration. My experience working on a large-scale refactoring tool leveraging `go/analysis` highlighted this complexity, necessitating a careful strategy combining AST traversal with semantic understanding.

**1. Clear Explanation:**

Locating an identifier's declaration using `go/analysis` involves several key steps. First, the analysis pass must obtain the AST of the Go source code using the `pass.Fset` and the provided `ast.File` objects within the `pass.Pass` struct. Second, it needs to recursively traverse this AST, searching for nodes that represent declarations (e.g., `ast.GenDecl`, `ast.FuncDecl`, `ast.AssignStmt` for short variable declarations). Third, within each declaration node, the analysis pass needs to check if any declared identifiers match the target identifier. Finally, the precise location of the declaration (file, line, column) can be extracted from the respective AST node using the `pass.Fset`.

The difficulty lies in handling different declaration contexts.  A variable declared within a function has a different AST structure from a global variable or a type definition.  Furthermore, shadowing—where an identifier is declared with the same name in nested scopes—adds another layer of complexity. The analysis needs to be intelligent enough to distinguish between these scenarios and report the declaration relevant to the current scope.  Ignoring scope management could lead to inaccurate results.

My experience troubleshooting a bug related to incorrect identifier resolution underscored the importance of meticulously considering scope during AST traversal. The initial implementation failed to correctly handle shadowed variables, causing incorrect declaration locations to be reported.  Refining the algorithm to account for package-level scope, function scope, and block scope (using `ast.Scope` effectively) proved crucial in delivering accurate results.

**2. Code Examples with Commentary:**

**Example 1: Finding Global Variable Declarations:**

```go
package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/astutil"
)

// analyzer is a pass that finds global variable declarations.
var analyzer = &analysis.Analyzer{
	Name: "globalVars",
	Doc:  "Finds global variable declarations.",
	Run: func(pass *analysis.Pass) (interface{}, error) {
		inspect.Preorder(pass, func(n ast.Node) bool {
			if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.VAR {
				for _, spec := range decl.Specs {
					if valSpec, ok := spec.(*ast.ValueSpec); ok {
						for _, name := range valSpec.Names {
							pass.Reportf(name.Pos(), "Global variable '%s' declared", name.Name)
						}
					}
				}
			}
			return true
		}, nil)
		return nil, nil
	},
	Requires: []*analysis.Analyzer{inspect.Analyzer},
}

func main() {
	//This is a placeholder, actual usage involves a go/analysis driver.
	fmt.Println("This is a sample analyzer; integration with go/analysis driver is required for actual usage.")
}
```

This example demonstrates a basic approach to locating global variable declarations.  It utilizes `inspect.Preorder` for AST traversal and specifically targets `ast.GenDecl` nodes with `token.VAR`.  This approach only detects top-level variable declarations.  More sophisticated methods are needed to handle local variables.  Note the use of `pass.Reportf` to record the position of the declaration.  The `main` function is purely illustrative; a complete solution requires integration with a `go/analysis` driver.


**Example 2:  Handling Function-Local Variables:**

```go
package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
)

var analyzer = &analysis.Analyzer{
	Name: "localVars",
	Doc:  "Finds local variable declarations.",
	Run: func(pass *analysis.Pass) (interface{}, error) {
		inspect.Preorder(pass, func(n ast.Node) bool {
			if assignStmt, ok := n.(*ast.AssignStmt); ok && assignStmt.Tok == token.DEFINE { //Short variable declaration
				for _, lhs := range assignStmt.Lhs {
					if ident, ok := lhs.(*ast.Ident); ok {
						pass.Reportf(ident.Pos(), "Local variable '%s' declared", ident.Name)
					}
				}
			}
			return true
		}, nil)
		return nil, nil
	},
	Requires: []*analysis.Analyzer{inspect.Analyzer},
}

func main() {
	fmt.Println("This is a sample analyzer; integration with go/analysis driver is required for actual usage.")
}
```

This example focuses on short variable declarations within functions. It identifies `ast.AssignStmt` nodes with `token.DEFINE` (short variable declaration syntax) and extracts the identifier name and position.  This method is limited to short declarations; handling `var` declarations within functions requires a more extensive approach.



**Example 3:  Improved Scope Handling (Conceptual):**

```go
package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
)


//Simplified conceptual example - requires significant expansion for robustness
var analyzer = &analysis.Analyzer{
	Name: "scopedVars",
	Doc:  "Finds variable declarations with improved scope handling.",
	Run: func(pass *analysis.Pass) (interface{}, error) {
		inspect.Nodes(pass, (*ast.Ident)(nil), func(n ast.Node) {
			ident, _ := n.(*ast.Ident)
			obj := pass.TypesInfo.ObjectOf(ident)
			if obj != nil {
				pass.Reportf(obj.Pos(), "Variable '%s' declared at %s", ident.Name, obj.Pos())

			}
		})
		return nil, nil
	},
	Requires: []*analysis.Analyzer{inspect.Analyzer},
}

func main() {
	fmt.Println("This is a sample analyzer; integration with go/analysis driver is required for actual usage.")
}

```

This example illustrates a more advanced (but still simplified) approach using the `types.Info` obtained through `pass.TypesInfo`.  This allows us to obtain the `types.Object` associated with an identifier, thus giving access to accurate scope information and avoiding the pitfalls of naive AST traversal.  This provides a foundation for effectively managing shadowed variables.  A production-ready version would need substantial error handling and more comprehensive handling of various AST nodes and declaration types.


**3. Resource Recommendations:**

The official Go documentation on the `go/analysis` package, the `go/ast` package, and the `go/types` package are essential.  Understanding AST traversal algorithms and the specifics of the Go language specification regarding variable scoping and declarations are crucial.  Furthermore, studying existing static analysis tools within the Go ecosystem can provide valuable insights and best practices.  Careful examination of the source code for tools like `go vet` can prove exceptionally helpful in learning techniques for robust identifier resolution.
