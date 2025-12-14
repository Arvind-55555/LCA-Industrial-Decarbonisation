"""
Multi-Objective Optimization for LCA
Balances emissions, cost, and feasibility constraints
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logging.warning("pymoo not available, using scipy optimization")

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Optimization objectives"""
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"


@dataclass
class OptimizationResult:
    """Optimization result container"""
    optimal_solution: np.ndarray
    objectives: Dict[str, float]
    constraints_satisfied: bool
    metadata: Dict[str, Any]


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for LCA.
    
    Formulation:
    min LCA(x)
    s.t. Technical feasibility
         Cost constraints (optional)
         Policy compliance (CBAM, ETS)
         Resource availability
    """
    
    def __init__(
        self,
        objectives: List[ObjectiveType],
        constraints: Optional[List[Callable]] = None
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: List of objectives to optimize
            constraints: List of constraint functions
        """
        self.objectives = objectives
        self.constraints = constraints or []
        
        logger.info(f"Multi-objective optimizer initialized with {len(objectives)} objectives")
    
    def optimize(
        self,
        objective_functions: Dict[ObjectiveType, Callable],
        bounds: Dict[str, Tuple[float, float]],
        initial_guess: Optional[np.ndarray] = None,
        method: str = "NSGA2"
    ) -> OptimizationResult:
        """
        Perform multi-objective optimization.
        
        Args:
            objective_functions: Dict mapping objective types to functions
            bounds: Variable bounds {var_name: (min, max)}
            initial_guess: Initial guess (optional)
            method: Optimization method
        
        Returns:
            Optimization result
        """
        if PYMOO_AVAILABLE and method == "NSGA2":
            return self._optimize_pymoo(objective_functions, bounds, initial_guess)
        else:
            return self._optimize_scipy(objective_functions, bounds, initial_guess)
    
    def _optimize_pymoo(
        self,
        objective_functions: Dict[ObjectiveType, Callable],
        bounds: Dict[str, Tuple[float, float]],
        initial_guess: Optional[np.ndarray]
    ) -> OptimizationResult:
        """Optimize using pymoo NSGA2"""
        # Define problem
        class LCAProblem(Problem):
            def __init__(self, obj_funcs, constraints, bounds_dict):
                n_vars = len(bounds_dict)
                xl = np.array([b[0] for b in bounds_dict.values()])
                xu = np.array([b[1] for b in bounds_dict.values()])
                
                n_obj = len(obj_funcs)
                n_constr = len(constraints)
                
                super().__init__(n_var=n_vars, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
                
                self.obj_funcs = obj_funcs
                self.constraints = constraints
                self.bounds_dict = bounds_dict
                self.var_names = list(bounds_dict.keys())
            
            def _evaluate(self, X, out, *args, **kwargs):
                n_pop = X.shape[0]
                n_obj = len(self.obj_funcs)
                n_constr = len(self.constraints)
                
                F = np.zeros((n_pop, n_obj))
                G = np.zeros((n_pop, n_constr))
                
                for i in range(n_pop):
                    x_dict = {name: X[i, j] for j, name in enumerate(self.var_names)}
                    
                    # Objectives
                    for j, (obj_type, obj_func) in enumerate(self.obj_funcs.items()):
                        F[i, j] = obj_func(x_dict)
                    
                    # Constraints
                    for j, constr_func in enumerate(self.constraints):
                        G[i, j] = constr_func(x_dict)
                
                out["F"] = F
                out["G"] = G
        
        # Create problem
        problem = LCAProblem(objective_functions, self.constraints, bounds)
        
        # Run optimization
        algorithm = NSGA2(pop_size=40)
        res = minimize(problem, algorithm, ("n_gen", 100), verbose=False)
        
        # Get best solution (first Pareto-optimal solution)
        best_idx = 0
        optimal_solution = res.X[best_idx]
        
        # Evaluate objectives
        x_dict = {name: optimal_solution[j] for j, name in enumerate(bounds.keys())}
        objectives = {
            obj_type.value: obj_func(x_dict)
            for obj_type, obj_func in objective_functions.items()
        }
        
        # Check constraints
        constraints_satisfied = all(
            constr(x_dict) <= 0 for constr in self.constraints
        )
        
        return OptimizationResult(
            optimal_solution=optimal_solution,
            objectives=objectives,
            constraints_satisfied=constraints_satisfied,
            metadata={"method": "NSGA2", "n_evaluations": len(res.X)}
        )
    
    def _optimize_scipy(
        self,
        objective_functions: Dict[ObjectiveType, Callable],
        bounds: Dict[str, Tuple[float, float]],
        initial_guess: Optional[np.ndarray]
    ) -> OptimizationResult:
        """Optimize using scipy (single objective)"""
        from scipy.optimize import minimize as scipy_minimize
        
        # Use first objective as primary
        primary_obj = list(objective_functions.values())[0]
        var_names = list(bounds.keys())
        
        def objective(x):
            x_dict = {name: x[i] for i, name in enumerate(var_names)}
            return primary_obj(x_dict)
        
        # Constraints
        constraint_list = []
        for constr_func in self.constraints:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x, f=constr_func: f({name: x[i] for i, name in enumerate(var_names)})
            })
        
        # Bounds
        bounds_list = [bounds[name] for name in var_names]
        
        # Initial guess
        if initial_guess is None:
            initial_guess = np.array([(b[0] + b[1]) / 2 for b in bounds_list])
        
        # Optimize
        result = scipy_minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds_list,
            constraints=constraint_list
        )
        
        # Evaluate all objectives
        x_dict = {name: result.x[i] for i, name in enumerate(var_names)}
        objectives = {
            obj_type.value: obj_func(x_dict)
            for obj_type, obj_func in objective_functions.items()
        }
        
        return OptimizationResult(
            optimal_solution=result.x,
            objectives=objectives,
            constraints_satisfied=result.success,
            metadata={"method": "scipy", "n_iterations": result.nit}
        )

