# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from queue import PriorityQueue
from optimizn.combinatorial.opt_problem import OptProblem
from copy import deepcopy


class BnBProblem(OptProblem):
    def __init__(self, params, logger=None):
        self.params = params
        self.queue = PriorityQueue()
        self.total_iters = 0
        self.total_time_elapsed = 0
        super().__init__(logger)
        if not self.is_feasible(self.best_solution) or not self.is_complete(
                self.best_solution):
            raise Exception('Initial solution is infeasible or incomplete: '
                            + f'{self.best_solution}')
    
    def get_root(self):
        '''
        Produces the solution corresponding to the root node of the problem
        space tree
        '''
        raise NotImplementedError(
            'Implement a method to get the solution corresponding to the '
            + 'root node of the problem space tree')

    def lbound(self, sol):
        '''
        Computes lower bound for a feasible solution and the solutions that
        can be obtained from it through branching
        '''
        raise NotImplementedError(
            'Implement a method to compute a lower bound on a feasible '
            + 'solution')

    def branch(self, sol):
        '''
        Generates other solutions from a feasible solution (branching)
        '''
        raise NotImplementedError(
            'Implement a branching method to produce other solutions from a '
            + 'feasible solution')

    def is_complete(self, sol):
        '''
        Checks if a solution is a complete solution (solves the optimization
        problem)
        '''
        raise NotImplementedError(
            'Implement a method to check if a solution is a complete '
            + 'solution (solves the optimization problem)')

    def is_feasible(self, sol):
        '''
        Checks if a solution is a feasible solution (is within the constraints
        of the optimization problem, can be a complete solution or a partial
        solution)
        '''
        raise NotImplementedError(
            'Implement a method to check if a solution is a feasible '
            + 'solution (is within the constraints of the optimization '
            + 'problem, can be a complete solution or a partial solution)')

    def complete_solution(self, sol):
        '''
        Completes an incomplete solution for early detection of solutions
        that are potentially more optimal than the most optimal solution
        already observed (only needed for modified branch and bound algorithm)
        '''
        raise NotImplementedError(
            'Implement a method to complete an incomplete solution')

    def _log_results(self, iters, log_iters, time_elapsed, force=False):
        if force or iters == 1 or iters % log_iters == 0:
            self.logger.info(f'Iterations (current run): {iters}')
            self.logger.info(f'Iterations (total): {self.total_iters}')
            queue = list(self.queue.queue)
            self.logger.info(f'Queue size: {len(queue)}')
            self.logger.info(
                f'Time elapsed (current run): {time_elapsed} seconds')
            self.logger.info(
                f'Time elapsed (total): {self.total_time_elapsed} seconds')
            self.logger.info(f'Best solution: {self.best_solution}')
            self.logger.info(f'Best solution cost: {self.best_cost}')

    def _update_best_solution(self, sol):
        # get cost of solution and update best solution and cost if needed
        cost = self.cost(sol)
        if self.cost_delta(self.best_cost, cost) > 0:
            self.best_cost = cost
            self.best_solution = deepcopy(sol)
            self.logger.info(
                f'Updated best solution to: {self.best_solution}')
            self.logger.info(
                f'Updated best solution cost to: {self.best_cost}')

    def solve(self, iters_limit=1e6, log_iters=100, time_limit=3600,
              bnb_type=0):
        '''
        This branch and bound implementation is based on the following sources.

        This method executes either the traditional (bnb_type=0) or modified
        (bnb_type=1) branch and bound algorithm. In traditional branch and
        bound, partial solutions are not completed and are not evaluated
        against the current best solution, while in modified branch
        and bound, they are.

        Sources:

        [1] J. Clausen, "Branch and bound algorithms - principles and
        examples.." https://imada.sdu.dk/u/jbj/heuristikker/TSPtext.pdf,
        March 1999. Online; accessed 16-December-2022.

        [2] A. Bari, "7.2 0/1 knapsack using branch and bound."
        https://www.youtube.com/watch?v=yV1d-b_NeK8, February 2018. Online;
        accessed 16-December-2022.
        '''
        # initialization
        start = time.time()
        iters = 0
        time_elapsed = 0
        original_total_time_elapsed = deepcopy(self.total_time_elapsed)
        sol_count = 1  # breaks ties between solutions with same lower bound
        # solutions generated earlier are given priority in such cases

        # if problem class instance is loaded, queue is saved as list, so
        # convert back to PriorityQueue
        if type(self.queue) is not PriorityQueue:
            queue = PriorityQueue()
            for item in self.queue:
                queue.put(item)
            self.queue = queue
        # otherwise, queue is created as PriorityQueue, so put root solution
        # onto PriorityQueue
        else:
            root_sol = self.get_root()
            self.queue.put((self.lbound(root_sol), sol_count, root_sol))

        # explore solutions
        while not self.queue.empty() and iters != iters_limit and\
                time_elapsed < time_limit:
            # get solution, skip if lower bound is not less than best solution
            # cost
            lbound, _, curr_sol = self.queue.get()
            if self.cost_delta(self.best_cost, lbound) <= 0:
                continue

            # get and process branched solutions
            next_sols = self.branch(curr_sol)
            for next_sol in next_sols:
                # skip infeasible solutions
                if not self.is_feasible(next_sol):
                    continue

                # process branched solution
                if self.is_complete(next_sol):
                    # if solution is complete, update best solution and best
                    # solution cost if needed
                    self._update_best_solution(next_sol)
                else:
                    # if algorithm type is 1 (modified branch and bound), then
                    # complete solution and update best solution and best
                    # solution cost if needed
                    if bnb_type == 1:
                        completed_sol = self.complete_solution(next_sol)
                        self._update_best_solution(completed_sol)

                    # if lower bound is less than best solution cost, put
                    # incomplete, feasible solution into queue
                    lbound = self.lbound(next_sol)
                    if self.cost_delta(self.best_cost, lbound) > 0:
                        sol_count += 1
                        self.queue.put((lbound, sol_count, next_sol))

            # log best solution and min cost, update iterations count and
            # time elapsed
            iters += 1
            self.total_iters += 1
            time_elapsed = time.time() - start
            self.total_time_elapsed = original_total_time_elapsed +\
                time_elapsed
            self._log_results(iters, log_iters, time_elapsed)

        # return best solution and cost
        self._log_results(iters, log_iters, time_elapsed, force=True)
        return self.best_solution, self.best_cost

    def persist(self):
        # convert the queue to a list before saving solution
        self.queue = list(self.queue.queue)
        super().persist()
