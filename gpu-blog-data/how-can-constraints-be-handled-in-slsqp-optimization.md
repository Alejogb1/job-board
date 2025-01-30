---
title: "How can constraints be handled in SLSQP optimization using openMDAO?"
date: "2025-01-30"
id: "how-can-constraints-be-handled-in-slsqp-optimization"
---
The Sequential Least Squares Programming (SLSQP) algorithm, a common choice for constrained optimization in OpenMDAO, directly addresses constraints by iteratively refining a search direction that satisfies both the objective function's gradient and the active constraint gradients. This hinges on constructing a quadratic approximation of the Lagrangian function within each iteration. My experience, across several aerospace design projects involving complex aerodynamic and structural analyses, confirms that effective handling of constraints with SLSQP within OpenMDAO requires careful implementation both in the problem setup and within the OpenMDAO components themselves.

Fundamentally, SLSQP seeks a search direction, denoted as ‘p’, which minimizes a quadratic model of the objective function while adhering to linearized constraints. The algorithm employs active set strategies to estimate which constraints are currently binding at each iteration. These constraints, those at or near their limits, are then explicitly considered in the search direction computation. This process involves solving a constrained quadratic programming subproblem. The key here is that SLSQP doesn’t directly minimize a constrained objective; it minimizes a sequence of approximations of the Lagrangian—a function that incorporates both the objective and constraints—thereby indirectly satisfying the constraints as the search converges.

A crucial aspect for practical implementation involves properly defining constraints within an OpenMDAO model. These are not simply static inequalities; they must be explicitly formulated within the `Problem` definition and connected to the appropriate outputs of the components that calculate them. The optimizer then interacts with these constraints through OpenMDAO’s internal infrastructure, passing constraint values and their Jacobian contributions at each optimization iteration. Neglecting to define and connect constraints accurately results in the algorithm proceeding without proper guidance, possibly converging to an infeasible solution or, worse, a solution that violates intended physical or design limits.

Let's examine practical implementation. In a structural design optimization, I was tasked with minimizing the mass of a beam while respecting a maximum stress constraint and a maximum deflection limit. This scenario translates directly to a typical optimization case: objective function (mass minimization), and inequality constraints (stress and deflection limits). The challenge lay in ensuring that the computed stress and deflection values—outputs of the finite element analysis component—were passed correctly to the optimizer's constraint evaluation.

Here's a simplified example, demonstrating constraint definition using OpenMDAO.

```python
import openmdao.api as om
import numpy as np

class BeamAnalysis(om.ExplicitComponent):
    """Simplified beam analysis; calculates stress and deflection."""
    def initialize(self):
        self.options.declare('length', types=float, default=1.0, desc='Beam Length')
        self.options.declare('youngs_modulus', types=float, default=200e9, desc='Youngs Modulus')
    def setup(self):
      self.add_input('area', val=1e-4, desc='Cross-sectional Area')
      self.add_input('load', val=1000.0, desc='Applied Load')
      self.add_output('stress', val=0.0, desc='Maximum Bending Stress')
      self.add_output('deflection', val=0.0, desc='Maximum Deflection')

      self.declare_partials('stress', ['area', 'load'])
      self.declare_partials('deflection', ['area', 'load'])

    def compute(self, inputs, outputs):
        L = self.options['length']
        E = self.options['youngs_modulus']
        I = (inputs['area']**2) / 12.0 # simplified moment of inertia for rectangle
        
        outputs['stress'] = (inputs['load'] * L/2.0) / I
        outputs['deflection'] = (inputs['load'] * L**3.0) / (3.0 * E * I)

    def compute_partials(self, inputs, J):
        L = self.options['length']
        E = self.options['youngs_modulus']
        I = (inputs['area']**2) / 12.0
        
        J['stress','area'] = -(inputs['load'] * L/2.0) / I**2 * (2.0/12.0*inputs['area'])
        J['stress','load'] = L/2.0/I

        J['deflection','area'] = -(inputs['load'] * L**3.0) / (3.0 * E * I**2) * (2.0/12.0*inputs['area'])
        J['deflection','load'] = L**3.0 / (3.0 * E * I)
        
class Mass(om.ExplicitComponent):
    """Calculates the mass of the beam."""
    def initialize(self):
      self.options.declare('length', types=float, default=1.0, desc='Beam Length')
      self.options.declare('density', types=float, default=7850.0, desc='Beam Material Density')
    def setup(self):
      self.add_input('area', val=1e-4, desc='Cross-sectional Area')
      self.add_output('mass', val=0.0, desc='Beam Mass')
      self.declare_partials('mass','area')

    def compute(self, inputs, outputs):
        outputs['mass'] = self.options['density'] * inputs['area'] * self.options['length']

    def compute_partials(self, inputs, J):
        J['mass', 'area'] = self.options['density'] * self.options['length']

prob = om.Problem()
model = prob.model

model.add_subsystem('beam_analysis', BeamAnalysis(), promotes_inputs=['area', 'load'], promotes_outputs=['stress','deflection'])
model.add_subsystem('mass', Mass(), promotes_inputs=['area'], promotes_outputs=['mass'])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9

prob.model.add_design_var('area', lower=1e-6, upper=1e-3)
prob.model.add_objective('mass')
prob.model.add_constraint('stress', upper=200e6) #stress limit of 200 MPa
prob.model.add_constraint('deflection', upper=0.005) #deflection limit of 5 mm

prob.setup()
prob.run_driver()

print('Optimized Area:', prob.get_val('area'))
print('Optimized Mass:', prob.get_val('mass'))
print('Stress:', prob.get_val('stress'))
print('Deflection:', prob.get_val('deflection'))

```

In this example, the `BeamAnalysis` component calculates the stress and deflection, and `Mass` computes the beam's mass. The optimization goal is to minimize mass (objective function) subject to a maximum stress of 200 MPa and a maximum deflection of 5 mm (constraints). These are declared using `add_objective` and `add_constraint` methods, respectively. The crucial point is that the constraint names ('stress' and 'deflection') directly match the output names of the component that calculates those values. This direct mapping allows OpenMDAO to pass these outputs into the SLSQP optimizer's constraint evaluation. We specify that we want the constraints to be less than or equal to given values by setting `upper` to those values.

Now, let's consider a slightly more complex constraint scenario, one where the constraint needs to consider outputs from multiple components.

```python
import openmdao.api as om
import numpy as np

class ComponentA(om.ExplicitComponent):
    """Component A produces output A."""
    def setup(self):
        self.add_input('x', val=1.0)
        self.add_output('output_a', val=2.0)
        self.declare_partials('output_a', 'x', val=1.0)

    def compute(self, inputs, outputs):
        outputs['output_a'] = 2.0 * inputs['x']

class ComponentB(om.ExplicitComponent):
    """Component B produces output B."""
    def setup(self):
        self.add_input('y', val=1.0)
        self.add_output('output_b', val=3.0)
        self.declare_partials('output_b', 'y', val=1.0)

    def compute(self, inputs, outputs):
        outputs['output_b'] = 3.0 * inputs['y']

class ConstraintComp(om.ExplicitComponent):
  """Combines two outputs into a single constraint."""
  def setup(self):
    self.add_input('output_a', val = 2.0)
    self.add_input('output_b', val=3.0)
    self.add_output('constraint', val=0.0)

    self.declare_partials('constraint', ['output_a', 'output_b'], val=1.0)

  def compute(self, inputs, outputs):
    outputs['constraint'] = inputs['output_a'] + inputs['output_b']

  def compute_partials(self, inputs, J):
    J['constraint', 'output_a'] = 1.0
    J['constraint', 'output_b'] = 1.0

prob = om.Problem()
model = prob.model

model.add_subsystem('comp_a', ComponentA(), promotes_inputs=['x'], promotes_outputs=['output_a'])
model.add_subsystem('comp_b', ComponentB(), promotes_inputs=['y'], promotes_outputs=['output_b'])
model.add_subsystem('constraint_comp', ConstraintComp(), promotes_inputs=['output_a', 'output_b'], promotes_outputs=['constraint'])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('x', lower=0, upper=5)
prob.model.add_design_var('y', lower=0, upper=5)
prob.model.add_objective('x')
prob.model.add_constraint('constraint', upper=5.0)

prob.setup()
prob.run_driver()

print('Optimized x:', prob.get_val('x'))
print('Optimized y:', prob.get_val('y'))
print('Constraint Value:', prob.get_val('constraint'))
```
Here, `ComponentA` and `ComponentB` produce outputs, `output_a` and `output_b`, respectively.  A new `ConstraintComp` takes these outputs as inputs and combines them to form a single constraint value using a simple sum. The optimization problem seeks to minimize `x` while keeping the sum of `output_a` and `output_b` less than or equal to 5.0. This example highlights that constraint functions can be complex aggregations of multiple component outputs. The use of a dedicated component to formulate the constraint function enhances modularity and maintainability in complex models.

Finally, consider a case where we need to impose bounds on a component's output indirectly. Instead of placing bounds on the input directly, we might want to constrain a computed output to be within a specific range, effectively placing indirect constraints on the input.

```python
import openmdao.api as om

class CompC(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=2.0)
        self.add_output('output_c', val=0.0)
        self.declare_partials('output_c','x')

    def compute(self, inputs, outputs):
      outputs['output_c'] = inputs['x']**2

    def compute_partials(self, inputs, J):
        J['output_c', 'x'] = 2*inputs['x']

prob = om.Problem()
model = prob.model
model.add_subsystem('comp_c', CompC(), promotes_inputs=['x'], promotes_outputs=['output_c'])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('x', lower=-5, upper=5)
prob.model.add_objective('x')
prob.model.add_constraint('output_c', lower=1.0, upper=4.0)

prob.setup()
prob.run_driver()

print("Optimized x:", prob.get_val('x'))
print("Optimized output_c:", prob.get_val('output_c'))
```

In this example, we constrain the output from CompC to be between 1 and 4, indirectly constraining the value of x. We achieve this by using both the lower and upper arguments to the add_constraint method.  This highlights a key versatility of the constraint handling in OpenMDAO.

In conclusion, effective constraint handling with SLSQP within OpenMDAO depends on accurate constraint definitions, correct connections to relevant component outputs, and sometimes the creation of intermediate components to combine or manipulate outputs into usable constraint forms. While the examples here demonstrate some straightforward applications, this capability extends to highly intricate problems encountered in engineering and scientific simulations. Proper use of `add_constraint`, combined with careful modular design, enables the successful application of constrained optimization with SLSQP in OpenMDAO. For further study, I recommend exploring the OpenMDAO documentation and the optimization theory behind Sequential Quadratic Programming. Publications on constrained optimization algorithms will also offer valuable insight.
