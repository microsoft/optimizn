# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from optimizn.combinatorial.opt_problem import OptProblem
import inspect


class BnBProblem(OptProblem):
    def __init__(self, params, logger=None):
        # initialization
        self.total_iters = 0
        self.total_time_elapsed = 0
        super().__init__(params, logger)

        # check initial solution
        if self.init_solution is not None and not self.is_feasible(
                self.init_solution):
            raise Exception('Initial solution is infeasible: '
                            + f'{self.init_solution}')
        
        # initialize call stack with root solution
        root_sol = self.get_root()
        root_gen = self.branch(root_sol)
        if not inspect.isgenerator(root_gen):
            raise Exception('Branch method must return a generator')
        self.call_stack = [(root_sol, root_gen)]
    
    def get_root(self):
        '''
        Produces the root solution, from which other solutions are obtainable
        through branching
        '''
        raise NotImplementedError(
            'Implement a method to get the root solution, from which all other'
            + ' solutions are obtainable through branching')

    def lbound(self, sol):
        '''
        Computes lower bound for a given solution and the solutions that can be
        obtained from it through branching
        '''
        raise NotImplementedError(
            'Implement a method to compute a lower bound on a given solution')

    def branch(self, sol):
        '''
        Generates other solutions from a given solution (branching), must
        return a generator
        '''
        raise NotImplementedError(
            'Implement a branching method to produce other solutions from a '
            + 'given solution, must return a generator')

    def is_feasible(self, sol):
        '''
        Checks if a solution is feasible (solves the optimization problem,
        adhering to its constraints)
        '''
        raise NotImplementedError(
            'Implement a method to check if a solution is feasible (solves '
            + 'the optimization problem, adhering to its constraints)')

    def complete_solution(self, sol):
        '''
        Completes an incomplete solution for early detection of solutions
        that are potentially more optimal than the most optimal solution
        already observed (only needed for modified branch and bound algorithm)
        '''
        raise NotImplementedError(
            'Implement a method to complete an incomplete solution')

    def _log_results(self, log_iters, force=False):
        if force or self.current_iters % int(log_iters) == 0:
            self.logger.info(f'Iterations (current run): {self.current_iters}')
            self.logger.info(f'Iterations (total): {self.total_iters}')
            self.logger.info(
                f'Time elapsed (current run): {self.current_time_elapsed} '
                + 'seconds')
            self.logger.info(
                f'Time elapsed (total): {self.total_time_elapsed} seconds')
            self.logger.info(f'Best solution: {self.best_solution}')
            self.logger.info(f'Best solution cost: {self.best_cost}')

    def _update_best_solution(self, sol):
        # get cost of solution and update best solution and cost if needed
        cost = self.cost(sol)
        if self.cost_delta(self.best_cost, cost) > 0:
            self.best_cost = cost
            self.best_solution = sol
            self.logger.info(
                f'Updated best solution to: {self.best_solution}')
            self.logger.info(
                f'Updated best solution cost to: {self.best_cost}')

    def solve(self, iters_limit=1e6, log_iters=100, time_limit=3600,
              bnb_type=0):
        '''
        This branch and bound implementation is based on the following sources.

        This method executes either the traditional (bnb_type=0) or look-ahead
        (bnb_type=1) branch and bound algorithm. In traditional branch and
        bound, partial solutions are not completed and are not evaluated
        against the current best solution, while in look-ahead branch
        and bound, they are.

        Sources:

        [1] J. Clausen, "Branch and bound algorithms - principles and
        examples.." https://imada.sdu.dk/u/jbj/heuristikker/TSPtext.pdf,
        March 1999. Online; accessed 16-December-2022.

        [2] A. Bari, "7.2 0/1 knapsack using branch and bound."
        https://www.youtube.com/watch?v=yV1d-b_NeK8, February 2018. Online;
        accessed 16-December-2022.
        '''

        # initialization for current run
        start = time.time()
        self.current_iters = 0
        self.current_time_elapsed = 0
        original_total_time_elapsed = self.total_time_elapsed
        self.terminate_early = False

        # helper function to check termination conditions
        def _terminate():
            return self.current_iters >= iters_limit or\
                self.current_time_elapsed >= time_limit

        # recursive helper function to evaluate a solution
        def _evaluate(sol, sol_gen):
            # compare against best solution
            if self.is_feasible(sol):
                # if solution is feasible, update best solution and best
                # solution cost if needed
                self._update_best_solution(sol)
                return
            else:
                # if algorithm type is 1 (modified branch and bound), then
                # complete solution and update best solution and best
                # solution cost if needed
                if bnb_type == 1:
                    completed_sol = self.complete_solution(sol)
                    if self.is_feasible(completed_sol):
                        self._update_best_solution(completed_sol)
            
            # check if a better solution could be obtained
            if self.cost_delta(self.best_cost, self.lbound(sol)) <= 0:
                return
            
            # update iterations count and time elapsed, log results
            self.current_iters += 1
            self.total_iters += 1
            self.current_time_elapsed = time.time() - start
            self.total_time_elapsed = original_total_time_elapsed +\
                self.current_time_elapsed
            self._log_results(log_iters)
            
            # check termination conditions
            if _terminate():
                self.logger.info(
                    'Iterations/time limit reached, terminating algorithm')
                self.terminate_early = True
                return

            # evaluate solutions obtained by branching on the current solution
            for next_sol in sol_gen:
                if next_sol is None:
                    # generator yielded nothing
                    continue
                next_sol_gen = self.branch(next_sol)
                self.call_stack.append((next_sol, next_sol_gen))
                _evaluate(next_sol, next_sol_gen)
                # do not remove solution from call stack if evaluation was
                # terminated early
                if self.terminate_early:
                    break
                self.call_stack.pop()
        
        # run branch and bound algorithm
        while len(self.call_stack) > 0:
            # check termination conditions
            if _terminate():
                self.logger.info(
                    'Iterations/time limit reached, terminating algorithm')
                break

            # evaluate solution on call stack
            sol, sol_gen = self.call_stack[-1]
            _evaluate(sol, sol_gen)
            # do not remove solution from call stack if evaluation was
            # terminated early
            if self.terminate_early:
                break
            self.call_stack.pop()
        
        # return best solution and best solution cost
        return self.best_solution, self.best_cost
