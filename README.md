# optimizn
This Python library provides several optimization-related utilities that can be used to solve a variety of optimization problems.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

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
This library offers a generalizeable implementation for simulated annealing through a superclass called `SimAnnealProblem`, located in `./optimizn/combinatorial/anneal.py` file. 

To use simulated annealing for their own optimization problem, users should create a subclass specific to their optimization problem that extends `SimAnnealProblem`. The subclass must implement the following methods.
- `get_candidate` (required): provides an initial solution
- `next_candidate` (required): produces a neighboring solution, given a current solution
- `cost` (required): objective function, returns a cost value for a given solution (lower cost value means more optimal solution)
- `cost_delta` (optional): default is the difference between two cost values, can be changed based on the nature of the problem

#### Branch and Bound
This library offers a generalizeable implementation for branch and bound through a superclass called `BnBProblem`, located in `./optimizn/combinatorial/branch_and_bound.py` file. 

This superclass supports two types of branch and bound. The first type is traditional branch and bound, where partial solutions are not checked against the current best solution. The second type is modified branch and bound, where partial solutions are completed and checked against the current best solution. 

To use branch and bound for their own optimization problem, users should create a subclass specific to their optimization problem that extends `BnBProblem`. The subclass must implement the following methods.
- `get_candidate` (required): provides an initial solution
- `branch` (required): produces other solutions from a current solution, which correspond to subproblems with additional constraints and constrained solution spaces
- `cost` (required): objective function, returns a cost value for a given solution (lower cost value means more optimal solution)
- `lbound` (required): returns the lowest cost value for a given solution and all other solutions in the same constrained solution space (lower cost value means more optimal solution)
- `is_feasible` (required): returns True if a given solution is feasible, False if not
- `is_complete` (required): returns True if a given solution is complete, False if not
- `complete_solution` (required/optional): completes a partial solution (required for modified branch and bound, optional for traditional branch and bound)
- `cost_delta` (optional): default is the difference between two cost values, can be changed based on the nature of the problem

#### Continuous Training
Both the `SimAnnealProblem` and the `BnBProblem` extend a superclass called `OptProblem` that has a `persist` function, which saves optimization problems to three folders: `DailyObj` (contains the optimization problem parameters), `DailyOpt` (contains instances of the optimization problem class each corresponding to a single run of the optimization algorithm), and `GlobalOpt` (contains the instance of the optimization problem class with the most optimal solution seen across all runs of the optimization algorithm). The user must specify the problem parameters in their optimization problem class through a `params` attribute in order for their problem parameters, problem instances, and optimal solutions to be saved by continuous training.

These saved optimization problems (and their solutions) can be loaded from these directories and run again, so the algorithms can continue from where they left off in their previous runs. This allows them to find the most optimal solutions they can given the compute time and resources they have, even if they are available in disjoint intervals. 
