# optimizn
This Python library provides several optimization-related utilities that can be used to solve a variety of optimization problems.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

## Getting Started
This library is available for use on PyPI here: [https://pypi.org/project/optimizn/](https://pypi.org/project/optimizn/)

For local development, do the following. 
- Clone this repository.
- Set up and activate a Python3 virtual environment using `conda`. More info here: [https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
- Navigate to the `optimizn` repo.
- Run the command: `python3 setup.py install` to install the package in the conda virtual environment. 
- As development progresses, run the above command to update the build in the conda virtual environment.
- To run the unit tests, run the command: `pytest`

## Offerings

### Combinatorial optimization

#### Simulated Annealing
This library offers a generalizeable implementation for simulated annealing through a superclass called `SimAnnealProblem`, which can be imported like so: `from optimizn.combinatorial.simulated_annealing import SimAnnealProblem`. 

To use simulated annealing for their own optimization problem, users should create a subclass specific to their optimization problem that extends `SimAnnealProblem`. The subclass must implement the following methods.
- `get_initial_solution` (required): provides an initial solution
- `reset_candidate` (optional): resets the current solution, defaults to `get_initial_solution` but can be overridden
- `next_candidate` (required): produces a neighboring solution, given a current solution
- `get_temperature` (optional): returns a temperature value (lower/higher value means lower/higher chances of updating current solution to a less optimal solution) given a number of iterations, defaults to $\frac{4000}{1 + e^(x / 3000)}$ but can be overridden
- `cost` (required): objective function, returns a cost value for a given solution (lower cost value means more optimal solution)
- `cost_delta` (optional): default is the difference between two cost values, can be changed based on the nature of the problem

The simulated annealing algorithm can be run using the inherited `anneal` method of the subclass. Through the arguments to this function, the user can specify the number of iterations (`n_iter`, defaults to 10000), the reset probability (`reset_p`, defaults to 1/10000), the interval (number of iterations) for logging progress (`log_iters`, defaults to 10000), and time limit of the algorithm in seconds (`time_limit`, defaults to 3600). 

#### Branch and Bound
This library offers a generalizeable implementation for branch and bound through a superclass called `BnBProblem` which can be imported like so: `from optimizn.combinatorial.branch_and_bound import BnBProblem`. 

This superclass supports two types of branch and bound. The first type is traditional branch and bound, where partial solutions are not checked against the current best solution. The second type is look-ahead branch and bound, where partial solutions are completed and checked against the current best solution. 

To use branch and bound for their own optimization problem, users should create a subclass specific to their optimization problem that extends `BnBProblem`. The subclass must implement the following methods.
- `get_initial_solution` (optional): provides an initial solution, defaults to returning None but can be overridden
- `get_root` (required): provides the root solution, from which other solutions are obtainable via branching
- `branch` (required): produces other solutions from a current solution, which correspond to subproblems with additional constraints and constrained solution spaces
- `cost` (required): objective function, returns a cost value for a given solution (lower cost value means more optimal solution)
- `lbound` (required): returns the lowest cost value for a given solution and all other solutions in the same constrained solution space (lower cost value means more optimal solution)
- `is_feasible` (required): returns True if a given solution is feasible, False if not
- `complete_solution` (required/optional): completes a partial solution (required for modified branch and bound, optional for traditional branch and bound)
- `cost_delta` (optional): default is the difference between two cost values, can be changed based on the nature of the problem

When calling the constructor of the `BnBProblem` class, a selection strategy must be provided, using the `BnBSelectionStrategy` enum, which can be imported like so: `from optimizn.combinatorial.branch_and_bound import BnBSelectionStrategy`. There are three supported selection strategies: depth-first (`BnBSelectionStrategy.DEPTH_FIRST`, where the branch and bound algorithm selects and evaluates nodes in a depth-first-search manner), depth-first-best-first (`BnBSelectionStrategy.DEPTH_FIRST_BEST_FIRST`, where the algorithm selects and evaluates nodes in a depth-first-search manner, prioritizes lower bound for nodes of the same depth in the tree), or best-first-depth-first (`BnBSelectionStrategy.BEST_FIRST_DEPTH_FIRST`, where the algorithm selects and evaluates the node with the lowest lower bound, prioritizes depth in tree for nodes with the same lower bound).

The branch and bound algorithm can be run using the inherited `solve` method of the subclass. Through the arguments to this function, the user can specify the number of iterations (`iters_limit`, defaults to 1e6), the interval (number of iterations) for logging progress (`log_iters`, defaults to 100), the time limit of the algorithm in seconds (`time_limit`, defaults to 3600), and the type of branch and bound (`bnb_type`, 0 for traditional, 1 for look-ahead, defaults to 0). 

#### Continuous Training
Both the `SimAnnealProblem` and the `BnBProblem` extend a superclass called `OptProblem` that has a `persist` method, the default implementation of which saves optimization problem resources to three folders: `DailyObj` (contains the optimization problem parameters), `DailyOpt` (contains instances of the optimization problem class each corresponding to a single run of the optimization algorithm), and `GlobalOpt` (contains the instance of the optimization problem class with the most optimal solution seen across all runs of the optimization algorithm). The `persist` method can be overridden by the user. When calling either the `SimAnnealProblem` or `BnBProblem` constructor in their subclass, the user must specify the problem parameters through the `params` argument in order for their problem parameters, problem instances, and optimal solutions to be saved by continuous training.

These saved optimization problem resources can be loaded from these directories and the optimization algorithms can be run again, continuing from where they left off in their previous runs. For the default implementation of `persist`, a corresponding function called `load_latest_pckl` is provided to load optimization problem resources given the path to the desired folder (`DailyObj`, `DailyOpt`, or `GlobalOpt`). The `load_latest_pckl` function can be imported like so: `from optimizn.combinatorial.opt_problem import load_latest_pckl`.

Continuous training allows optimization algorithms to find the most optimal solutions they can given the compute time and resources they have, even if they are available in disjoint intervals. 
