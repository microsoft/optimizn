# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from optimizn.combinatorial.opt_problem import OptProblem
import inspect
from queue import PriorityQueue
from enum import Enum


class BnBSelectionStrategy(Enum):
    DEPTH_FIRST = 'DEPTH_FIRST'
    DEPTH_FIRST_BEST_FIRST = 'DEPTH_FIRST_BEST_FIRST'
    BEST_FIRST_DEPTH_FIRST = 'BEST_FIRST_DEPTH_FIRST'


class BnBProblem(OptProblem):
    def __init__(self, params, bnb_selection_strategy, logger=None):
        if not isinstance(bnb_selection_strategy, BnBSelectionStrategy):
            raise Exception(
                f'Invalid value for bnb_selection_strategy, must be one of '
                + 'the following BnBSelectionStrategy enum values: DEPTH_FIRST'
                + 'DEPTH_FIRST_BEST_FIRST, BEST_FIRST')
    
        # initialize for depth-first branch and bound
        self.bnb_selection_strategy = bnb_selection_strategy
        if self.bnb_selection_strategy == BnBSelectionStrategy.DEPTH_FIRST:
            # initialization
            self.total_iters = 0
            self.total_time_elapsed = 0
            super().__init__({
                    'params': params,
                    'bnb_selection_strategy': bnb_selection_strategy
                }, logger)

            # check initial solution
            if self.init_solution is not None and not self.is_feasible(
                    self.init_solution):
                raise Exception('Initial solution is infeasible: '
                                + f'{self.init_solution}')
            
            # initialize stack with root solution
            root_sol = self.get_root()
            root_gen = self.branch(root_sol)
            if not inspect.isgenerator(root_gen):
                raise Exception('Branch method must return a generator when '
                                + 'running depth-first branch and bound')
            self.stack = [(root_sol, root_gen)]
        # initialize for depth-first-best-first or best-first branch and bound
        elif self.bnb_selection_strategy in {
                BnBSelectionStrategy.DEPTH_FIRST_BEST_FIRST,
                BnBSelectionStrategy.BEST_FIRST_DEPTH_FIRST}:
            self.priority_queue = PriorityQueue()
            self.total_iters = 0
            self.total_time_elapsed = 0
            super().__init__(params, logger)
            if self.init_solution is not None and not self.is_feasible(
                    self.init_solution):
                raise Exception('Initial solution is infeasible: '
                                + f'{self.init_solution}')

            self.sol_count = 1  # breaks ties between solutions with same lower
            # bound and depth, solutions generated earlier are given priority

            # put root solution onto PriorityQueue
            root_sol = self.get_root()
            if self.bnb_selection_strategy ==\
                    BnBSelectionStrategy.DEPTH_FIRST_BEST_FIRST:
                self.priority_queue.put(
                    (0, self.lbound(root_sol), self.sol_count, root_sol))
            else:
                self.priority_queue.put(
                    (self.lbound(root_sol), 0, self.sol_count, root_sol))
            # solution tuples consist of four values: lower bound, solution
            # depth, solution count, solution
        else:
            raise Exception(
                f'Invalid value for bnb_selection_strategy, must be one of '
                + 'the following BnBSelectionStrategy enum values: DEPTH_FIRST'
                + 'DEPTH_FIRST_BEST_FIRST, BEST_FIRST_DEPTH_FIRST')

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
        Generates other solutions from a given solution (branching)
        '''
        raise NotImplementedError(
            'Implement a branching method to produce other solutions from a '
            + 'given solution')

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
        already observed (only needed for look-ahead branch and bound
        algorithm)
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

    def _evaluate_solution(self, sol, bnb_type, sol_lb=None):
        # get lower bound of solution, if not provided
        if sol_lb is None:
            sol_lb = self.lbound(sol)

        # evaluate if a more optimal solution can be obtained
        branch = False
        if self.cost_delta(self.best_cost, sol_lb) > 0:
            branch = True
            # compare against best solution
            if self.is_feasible(sol):
                # if solution is feasible, update best solution and
                # best solution cost if needed
                self._update_best_solution(sol)
                branch = False
            else:
                # if algorithm type is 1 (look-ahead branch and bound),
                # then complete the partial solution and update best
                # solution and best solution cost if needed
                if bnb_type == 1:
                    completed_sol = self.complete_solution(sol)
                    if self.is_feasible(completed_sol):
                        self._update_best_solution(completed_sol)
        
        # return boolean to branch
        return branch

    def _update_progress_and_log(
            self, start, original_total_time_elapsed, log_iters):
        self.current_iters += 1
        self.total_iters += 1
        self.current_time_elapsed = time.time() - start
        self.total_time_elapsed = original_total_time_elapsed +\
            self.current_time_elapsed
        self._log_results(log_iters)

    def _terminate(self, iters_limit, time_limit):
        return self.current_iters >= iters_limit or\
            self.current_time_elapsed >= time_limit

    def _solve_df(self, iters_limit, log_iters, time_limit, bnb_type):
        '''
        This depth-first branch and bound implementation is based on the
        following sources.

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
        # converts a list to a generator
        def _list_to_gen(lst):
            for item in lst:
                yield item

        # initialization for current run
        start = time.time()
        self.current_iters = 0
        self.current_time_elapsed = 0
        original_total_time_elapsed = self.total_time_elapsed
        self.terminate_early = False
        # if stack elements contain lists instead of generators (instance
        # loaded from memory), convert lists back to generators
        if len(self.stack) != 0:
            if type(self.stack[0][1]) == list:
                self.stack = list(map(
                    lambda i: (i[0], _list_to_gen(i[1])), self.stack))

        # recursive helper function to evaluate a solution
        def _evaluate(sol, sol_gen, skip_eval):
            branch = True
            if not skip_eval:
                if self._terminate(iters_limit, time_limit):
                    self.logger.info(
                        'Iterations/time limit reached, terminating algorithm')
                    self.terminate_early = True
                    return
            
                # evaluate solution
                branch = self._evaluate_solution(sol, bnb_type)
                self._update_progress_and_log(
                    start, original_total_time_elapsed, log_iters)

            # evaluate solutions obtained by branching
            if branch:
                for next_sol in sol_gen:
                    next_sol_gen = self.branch(next_sol)
                    self.stack.append((next_sol, next_sol_gen))
                    _evaluate(next_sol, next_sol_gen, False)
                    # do not remove solution from stack if evaluation was
                    # terminated early
                    if self.terminate_early:
                        break
                    self.stack.pop()
        
        # run branch and bound algorithm
        skip_eval = False
        while len(self.stack) > 0:
            # evaluate solution
            sol, sol_gen = self.stack[-1]
            _evaluate(sol, sol_gen, skip_eval)
            # do not remove solution from stack if evaluation was
            # terminated early
            if self.terminate_early:
                break
            self.stack.pop()
            skip_eval = True

    def _solve_dfbef_befdf(self, iters_limit, log_iters, time_limit, bnb_type):
        '''
        This depth-first-best-first or best-first-depth-first branch and bound
        implementation is based on the following sources.

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

        # if problem class instance is loaded, priority_queue is saved as list,
        # so convert back to PriorityQueue
        if type(self.priority_queue) is not PriorityQueue:
            priority_queue = PriorityQueue()
            for item in self.priority_queue:
                priority_queue.put(item)
            self.priority_queue = priority_queue

        # evaluate solutions
        while not self.priority_queue.empty():
            if self._terminate(iters_limit, time_limit):
                self.logger.info(
                    'Iterations/time limit reached, terminating algorithm')
                self.terminate_early = True
                break

            # get solution from priority queue
            if self.bnb_selection_strategy ==\
                    BnBSelectionStrategy.DEPTH_FIRST_BEST_FIRST:
                depth, sol_lb, _, sol = self.priority_queue.get()
            else:
                sol_lb, depth, _, sol = self.priority_queue.get()
            
            # evaluate solution
            branch = self._evaluate_solution(sol, bnb_type, sol_lb)

            # put solutions obtained through branching on priority queue
            if branch:
                for next_sol in self.branch(sol):
                    next_sol_lb = self.lbound(next_sol)
                    self.sol_count += 1
                    if self.bnb_selection_strategy ==\
                            BnBSelectionStrategy.DEPTH_FIRST_BEST_FIRST:
                        self.priority_queue.put(
                            (depth - 1, next_sol_lb, self.sol_count, next_sol))
                    else:
                        self.priority_queue.put(
                            (next_sol_lb, depth - 1, self.sol_count, next_sol))
            
            # update progress and log
            self._update_progress_and_log(
                start, original_total_time_elapsed, log_iters)

    def solve(self, iters_limit=1e6, log_iters=100, time_limit=3600,
              bnb_type=0):
        if self.bnb_selection_strategy == BnBSelectionStrategy.DEPTH_FIRST:
            self._solve_df(iters_limit, log_iters, time_limit, bnb_type)
        elif self.bnb_selection_strategy in {
                BnBSelectionStrategy.BEST_FIRST_DEPTH_FIRST,
                BnBSelectionStrategy.DEPTH_FIRST_BEST_FIRST}:
            self._solve_dfbef_befdf(
                iters_limit, log_iters, time_limit, bnb_type)
        else:
            raise Exception(
                f'Invalid value for bnb_selection_strategy, must be one of '
                + 'the following BnBSelectionStrategy enum values: DEPTH_FIRST'
                + 'DEPTH_FIRST_BEST_FIRST, BEST_FIRST_DEPTH_FIRST')
    
        # log results, return best solution and best solution cost
        self._log_results(log_iters, force=True)
        return self.best_solution, self.best_cost

    def persist(self):
        if self.bnb_selection_strategy in {
                BnBSelectionStrategy.DEPTH_FIRST_BEST_FIRST,
                BnBSelectionStrategy.BEST_FIRST_DEPTH_FIRST}:
            # convert the priority queue to a list before saving solution
            self.priority_queue = list(self.priority_queue.queue)
        elif self.bnb_selection_strategy == BnBSelectionStrategy.DEPTH_FIRST:
            # convert generators in stack to list
            self.stack = list(map(lambda i: (i[0], list(i[1])), self.stack))
        else:
            raise Exception(
                f'Invalid value for bnb_selection_strategy, must be one of '
                + 'the following BnBSelectionStrategy enum values: DEPTH_FIRST'
                + 'DEPTH_FIRST_BEST_FIRST, BEST_FIRST_DEPTH_FIRST')
        super().persist()
