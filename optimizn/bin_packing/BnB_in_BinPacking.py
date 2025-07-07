"""
Branch and Bound implementation for Bin Packing Problem using optimizn package.
Combines BnB with best fit strategy and workload-based processing.
"""

import sys
import os
import time
import math
import logging
from typing import List, Tuple, Dict, Any, Optional, Generator

# Add optimizn to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'optimizn'))

from optimizn.combinatorial.branch_and_bound import BnBProblem, BnBSelectionStrategy
from FFD_and_BFD import best_fit_decreasing, generate_test_cases


class BinPackingBnBProblem(BnBProblem):
    """
    Branch and Bound implementation for bin packing that:
    1. Uses workload-based processing (each workload processed as a unit)
    2. Applies best fit strategy within each workload
    3. Enforces workload constraint: VMs from same workload must go to DIFFERENT nodes
    4. Sorts VMs within each workload in descending order for better solutions
    """
    
    def __init__(self, vm_lists: List[List[int]], node_capacities: List[int], 
                 bnb_selection_strategy: BnBSelectionStrategy = BnBSelectionStrategy.DEPTH_FIRST,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Branch and Bound bin packing problem.
        
        Args:
            vm_lists: List of workloads, each containing VM sizes
            node_capacities: List of node capacities
            bnb_selection_strategy: BnB strategy (DEPTH_FIRST, etc.)
            logger: Optional logger
        """
        self.vm_lists = vm_lists
        self.node_capacities = node_capacities
        self.num_workloads = len(vm_lists)
        self.num_nodes = len(node_capacities)
        
        # Sort VMs within each workload in descending order for better solutions
        self.sorted_vm_lists = []
        for workload in vm_lists:
            self.sorted_vm_lists.append(sorted(workload, reverse=True))
        
        # Flatten for total VM count
        self.all_vms = []
        for i, workload in enumerate(self.sorted_vm_lists):
            for vm_size in workload:
                self.all_vms.append((vm_size, i))  # (vm_size, workload_id)
        
        # Track best known solution (workload-aware BFD as upper bound for reference)
        bfd_result, self.bfd_bins_used, _ = self._workload_aware_best_fit(vm_lists, node_capacities)
        
        # Initialize BnB problem
        params = {
            'vm_lists': vm_lists,
            'node_capacities': node_capacities,
            'sorted_vm_lists': self.sorted_vm_lists
        }
        
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
        
        super().__init__(params, bnb_selection_strategy, logger)
        
        # Validate problem feasibility before starting
        self._validate_problem_feasibility()
        
        # No initial solution - let BnB start from scratch
        # The BFD result is kept only as a reference upper bound
        
        print(f"Initialized BnB for {self.num_workloads} workloads, {len(self.all_vms)} total VMs")
        print(f"BFD upper bound: {self.bfd_bins_used} bins")
    
    def get_root(self) -> Dict[str, Any]:
        """
        Get root solution - empty assignment where we start placing workloads.
        
        Returns:
            Root solution with empty assignments
        """
        return {
            'assignments': {},  # node_id -> list of (vm_size, workload_id)
            'workload_to_nodes': {},  # workload_id -> set of node_ids used
            'node_remaining_capacity': self.node_capacities.copy(),
            'placed_workloads': set(),  # Set of workload IDs that have been placed
            'workload_order': [],  # Order in which workloads were placed
            'bins_used': 0
        }
    
    def branch(self, sol: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Branch from current solution by trying each remaining workload as the next one to place.
        This explores all possible workload ordering combinations.
        
        Args:
            sol: Current partial solution
            
        Yields:
            New solutions with one more workload placed (trying all remaining workloads)
        """
        placed_workloads = sol['placed_workloads']
        
        # If all workloads are placed, this is a complete solution - no more branching
        if len(placed_workloads) >= self.num_workloads:
            return
        
        # Try each remaining unplaced workload as the next candidate
        for workload_id in range(self.num_workloads):
            if workload_id in placed_workloads:
                continue  # Skip already placed workloads
                
            workload_vms = self.sorted_vm_lists[workload_id]
            
            # If workload is empty, just mark it as placed and continue
            if not workload_vms:
                new_sol = {
                    'assignments': {k: v.copy() for k, v in sol['assignments'].items()},
                    'workload_to_nodes': {k: v.copy() for k, v in sol['workload_to_nodes'].items()},
                    'node_remaining_capacity': sol['node_remaining_capacity'].copy(),
                    'placed_workloads': placed_workloads.copy(),
                    'workload_order': sol['workload_order'].copy(),
                    'bins_used': sol['bins_used']
                }
                new_sol['placed_workloads'].add(workload_id)
                new_sol['workload_order'].append(workload_id)
                yield new_sol
                continue
            
            # Find possible placements for this workload
            placement_combinations = self._find_workload_placements(
                workload_vms, workload_id, sol
            )
            
            # Generate a branch for each valid placement combination
            for placement in placement_combinations:
                new_sol = self._apply_workload_placement(sol, workload_id, placement)
                yield new_sol
    
    def lbound(self, sol: Dict[str, Any]) -> int:
        """
        Calculate lower bound for current solution.
        
        For complete solutions, the lower bound equals the cost.
        For partial solutions, the lower bound is the current bins used 
        (we can't use fewer bins than we're already using).
        
        Args:
            sol: Current partial solution
            
        Returns:
            Lower bound on number of bins needed
        """
        return sol['bins_used']
    
    def is_feasible(self, sol: Dict[str, Any]) -> bool:
        """
        Check if solution is feasible.
        
        A solution is feasible if:
        1. It doesn't violate capacity constraints
        2. It doesn't violate workload constraints
        3. For the purposes of BnB "best solution" tracking, it must be complete
           (all workloads placed)
        
        Args:
            sol: Solution to check
            
        Returns:
            True if solution is feasible and complete
        """
        # Check if solution is complete (all workloads placed)
        # This prevents empty/partial solutions from being considered "best"
        if len(sol['placed_workloads']) < self.num_workloads:
            return False
            
        # All partial solutions are feasible if they don't violate constraints
        for node_id, vms in sol['assignments'].items():
            # Check capacity constraint
            total_size = sum(vm_size for vm_size, _ in vms)
            if total_size > self.node_capacities[node_id]:
                return False
        
        # Check workload coherence constraint: each workload's VMs must be on DIFFERENT nodes
        for workload_id, node_set in sol['workload_to_nodes'].items():
            # Count how many VMs from this workload are placed
            vms_placed = 0
            for node_id, vms in sol['assignments'].items():
                if node_id in node_set:
                    vms_placed += sum(1 for vm_size, wl_id in vms if wl_id == workload_id)
            
            # If workload has multiple VMs, they must be on different nodes
            if vms_placed > 1 and len(node_set) < vms_placed:
                return False  # Workload constraint violated: multiple VMs on same node
        
        return True
    
    def complete_solution(self, sol: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete partial solution using greedy best fit for remaining workloads.
        
        Args:
            sol: Partial solution
            
        Returns:
            Completed solution
        """
        # Create a copy for completion
        completed_sol = {
            'assignments': {k: v.copy() for k, v in sol['assignments'].items()},
            'workload_to_nodes': {k: v.copy() for k, v in sol['workload_to_nodes'].items()},
            'node_remaining_capacity': sol['node_remaining_capacity'].copy(),
            'placed_workloads': sol['placed_workloads'].copy(),
            'workload_order': sol['workload_order'].copy(),
            'bins_used': sol['bins_used']
        }
        
        # Place remaining workloads greedily
        for workload_id in range(self.num_workloads):
            if workload_id in completed_sol['placed_workloads']:
                continue
                
            workload_vms = self.sorted_vm_lists[workload_id]
            
            if not workload_vms:  # Skip empty workloads
                completed_sol['placed_workloads'].add(workload_id)
                completed_sol['workload_order'].append(workload_id)
                continue
            
            # Try to place this workload using our workload placement function
            placements = self._find_workload_placements(workload_vms, workload_id, completed_sol)
            
            if placements:
                # Apply the first (greedy) placement
                placement = placements[0]
                temp_sol = self._apply_workload_placement(completed_sol, workload_id, placement)
                # Update completed_sol with the placement
                completed_sol = temp_sol
            else:
                # Cannot place workload - return infeasible solution
                completed_sol['bins_used'] = float('inf')
                return completed_sol
        
        return completed_sol
    
    def cost(self, sol: Dict[str, Any]) -> float:
        """
        Calculate cost of solution (number of bins used).
        For partial solutions, return a lower bound estimate.
        
        Args:
            sol: Solution to evaluate
            
        Returns:
            Lower bound estimate of bins needed
        """
        bins_used_so_far = sol['bins_used']
        
        # If this is a complete solution, return exact cost
        if len(sol['workload_to_nodes']) == len(self.vm_lists):
            return bins_used_so_far
        
        # For partial solutions, compute a lower bound
        # Start with bins already used
        lower_bound = bins_used_so_far
        
        # Add lower bound for remaining workloads
        placed_workloads = set(sol['workload_to_nodes'].keys())
        remaining_workloads = [i for i in range(len(self.vm_lists)) if i not in placed_workloads]
        
        for workload_idx in remaining_workloads:
            vm_sizes = self.vm_lists[workload_idx]
            total_size = sum(vm_sizes)
            
            # Lower bound: at least ceil(total_size / capacity) bins needed
            # This is a very conservative lower bound
            min_bins_needed = math.ceil(total_size / max(self.node_capacities))
            lower_bound += min_bins_needed
        
        return lower_bound
    
    def _workload_aware_best_fit(self, vm_lists: List[List[int]], node_capacities: List[int]) -> Tuple[Dict[int, List[int]], int, float]:
        """
        Implement workload-aware best-fit strategy that processes workloads in order.
        This gives a more accurate upper bound for our BnB algorithm.
        
        Args:
            vm_lists: List of workloads, each containing VM sizes
            node_capacities: List of node capacities
            
        Returns:
            Tuple of (node_assignments, bins_used, execution_time)
        """
        start_time = time.time()
        
        # Initialize state
        node_assignments = {}  # node_id -> list of vm_sizes
        node_remaining = node_capacities.copy()
        workload_to_nodes = {}  # workload_id -> set of node_ids
        bins_used = 0
        
        # Process each workload in order
        for workload_id, workload in enumerate(vm_lists):
            if not workload:  # Skip empty workloads
                continue
                
            # Sort VMs in this workload in descending order
            sorted_vms = sorted(workload, reverse=True)
            workload_to_nodes[workload_id] = set()
            
            # Place each VM using best-fit strategy
            # CONSTRAINT: VMs from same workload must go to DIFFERENT nodes
            for vm_size in sorted_vms:
                best_node = None
                best_remaining = float('inf')
                
                # Find best fit among nodes NOT already used by this workload
                for node_id in range(len(node_capacities)):
                    if node_id not in workload_to_nodes[workload_id] and node_remaining[node_id] >= vm_size:
                        remaining_after = node_remaining[node_id] - vm_size
                        if remaining_after < best_remaining:
                            best_node = node_id
                            best_remaining = remaining_after
                
                # Place the VM
                if best_node is not None:
                    if best_node not in node_assignments:
                        node_assignments[best_node] = []
                        bins_used += 1
                    node_assignments[best_node].append(vm_size)
                    node_remaining[best_node] -= vm_size
                    workload_to_nodes[workload_id].add(best_node)
                else:
                    # Insufficient nodes available
                    raise ValueError(f"Cannot place VM of size {vm_size} from workload {workload_id}: "
                                   f"insufficient nodes available. All {len(node_capacities)} nodes "
                                   f"are either full or already used by this workload.")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return node_assignments, bins_used, execution_time

    def _find_workload_placements(self, workload_vms: List[int], workload_id: int, 
                                 sol: Dict[str, Any]) -> List[List[int]]:
        """
        Find all possible ways to place a complete workload using best-fit strategy.
        CONSTRAINT: VMs from the same workload must be placed on DIFFERENT nodes.
        Each placement is a list of node_ids corresponding to each VM in the workload.
        
        Args:
            workload_vms: List of VM sizes in this workload (sorted descending)
            workload_id: ID of the workload being placed
            sol: Current solution state
            
        Returns:
            List of placement combinations, where each combination is a list of node_ids
        """
        # For now, implement a simple greedy best-fit strategy that generates one placement
        # This can be extended to generate multiple placements for more thorough exploration
        
        placement = []
        temp_remaining = sol['node_remaining_capacity'].copy()
        nodes_used_by_workload = set()
        
        # Place each VM in the workload using best-fit
        # CRITICAL: Each VM must go to a DIFFERENT node (workload constraint)
        for vm_size in workload_vms:
            best_node = None
            best_remaining = float('inf')
            
            # Find best fit among nodes NOT already used by this workload
            for node_id in range(len(temp_remaining)):
                if (node_id not in nodes_used_by_workload and 
                    temp_remaining[node_id] >= vm_size):
                    remaining_after = temp_remaining[node_id] - vm_size
                    if remaining_after < best_remaining:
                        best_node = node_id
                        best_remaining = remaining_after
            
            # If we found a placement
            if best_node is not None:
                placement.append(best_node)
                temp_remaining[best_node] -= vm_size
                nodes_used_by_workload.add(best_node)
            else:
                # No valid placement found for this workload
                return []
        
        return [placement] if placement else []
    
    def _apply_workload_placement(self, sol: Dict[str, Any], workload_id: int, 
                                 placement: List[int]) -> Dict[str, Any]:
        """
        Apply a workload placement to create a new solution.
        
        Args:
            sol: Current solution
            workload_id: ID of workload being placed
            placement: List of node_ids for each VM in the workload
            
        Returns:
            New solution with the workload placed
        """
        new_sol = {
            'assignments': {k: v.copy() for k, v in sol['assignments'].items()},
            'workload_to_nodes': {k: v.copy() for k, v in sol['workload_to_nodes'].items()},
            'node_remaining_capacity': sol['node_remaining_capacity'].copy(),
            'placed_workloads': sol['placed_workloads'].copy(),
            'workload_order': sol['workload_order'].copy(),
            'bins_used': sol['bins_used']  # Will be recalculated below
        }
        
        # Add this workload to placed workloads and order
        new_sol['placed_workloads'].add(workload_id)
        new_sol['workload_order'].append(workload_id)
        
        workload_vms = self.sorted_vm_lists[workload_id]
        new_sol['workload_to_nodes'][workload_id] = set()
        
        # Place each VM according to the placement
        for vm_size, node_id in zip(workload_vms, placement):
            # Update assignments
            if node_id not in new_sol['assignments']:
                new_sol['assignments'][node_id] = []
            new_sol['assignments'][node_id].append((vm_size, workload_id))
            
            # Update workload to nodes mapping
            new_sol['workload_to_nodes'][workload_id].add(node_id)
            
            # Update remaining capacity
            new_sol['node_remaining_capacity'][node_id] -= vm_size
        
        # Correctly calculate bins_used as the number of nodes with assignments
        new_sol['bins_used'] = len(new_sol['assignments'])
        
        return new_sol

    def _validate_problem_feasibility(self):
        """
        Validate if the problem is feasible given the node constraints.
        
        Raises:
            ValueError: If the problem cannot be solved with available nodes
        """
        # Check basic capacity constraint
        total_vm_size = sum(sum(workload) for workload in self.vm_lists)
        total_capacity = sum(self.node_capacities)
        
        if total_vm_size > total_capacity:
            raise ValueError(f"Problem is infeasible: total VM size ({total_vm_size}) "
                           f"exceeds total node capacity ({total_capacity})")
        
        # Check if any single VM is too large for any node
        max_vm_size = max(max(workload) if workload else 0 for workload in self.vm_lists)
        max_node_capacity = max(self.node_capacities)
        
        if max_vm_size > max_node_capacity:
            raise ValueError(f"Problem is infeasible: largest VM size ({max_vm_size}) "
                           f"exceeds largest node capacity ({max_node_capacity})")
        
        # Check if we have enough nodes for the minimum possible solution
        # Each workload needs as many nodes as it has VMs (workload constraint)
        # But different workloads can share nodes, so we need max(workload_sizes)
        max_workload_size = max(len(workload) for workload in self.vm_lists if workload)
        available_nodes = len(self.node_capacities)
        
        if max_workload_size > available_nodes:
            raise ValueError(f"Problem is infeasible: largest workload has {max_workload_size} VMs "
                           f"(requiring {max_workload_size} different nodes) but only have {available_nodes} nodes")

    def is_complete(self, sol: Dict[str, Any]) -> bool:
        """
        Check if solution is complete (all workloads placed).
        
        Args:
            sol: Solution to check
            
        Returns:
            True if all workloads are placed
        """
        return len(sol['placed_workloads']) == self.num_workloads

    def _is_constraint_feasible(self, sol: Dict[str, Any]) -> bool:
        """
        Check if solution satisfies constraints (without requiring completeness).
        Used during branching to validate partial solutions.
        
        Args:
            sol: Solution to check
            
        Returns:
            True if solution satisfies constraints
        """
        # Check capacity constraints
        for node_id, vms in sol['assignments'].items():
            total_size = sum(vm_size for vm_size, _ in vms)
            if total_size > self.node_capacities[node_id]:
                return False
        
        # Check workload constraint: each workload's VMs must be on DIFFERENT nodes
        for workload_id, node_set in sol['workload_to_nodes'].items():
            vms_placed = 0
            for node_id, vms in sol['assignments'].items():
                if node_id in node_set:
                    vms_placed += sum(1 for vm_size, wl_id in vms if wl_id == workload_id)
            
            if vms_placed > 1 and len(node_set) < vms_placed:
                return False
        
        return True

def test_branch_and_bound():
    """
    Test the Branch and Bound implementation and compare with BFD.
    Creates a feasible test case where VMs from same workload can go to different nodes.
    """
    print("="*80)
    print("BRANCH AND BOUND BIN PACKING TEST")
    print("="*80)
    
    # Create a feasible test case manually
    # Workloads with small number of VMs per workload to ensure feasibility
    print("Creating feasible test case...")
    test_workloads = [
        [8],        # Workload 0: 1 VM (needs 1 node)
        [6, 5],     # Workload 1: 2 VMs (needs 2 nodes) 
        [4],        # Workload 2: 1 VM (needs 1 node)
        [3, 2]      # Workload 3: 2 VMs (needs 2 nodes)
    ]
    # Total: 6 VMs, needs 6 nodes minimum
    test_node_capacities = [16, 16, 16, 16, 16, 16, 16, 16]  # 8 nodes available
    
    print(f"\nGenerated Feasible Test Case:")
    print(f"  Workloads: {test_workloads}")
    print(f"  Node capacities: {test_node_capacities}")
    print(f"  Total VMs: {sum(len(wl) for wl in test_workloads)}")
    print(f"  Total capacity needed: {sum(sum(wl) for wl in test_workloads)}")
    print(f"  Total available capacity: {sum(test_node_capacities)}")
    print(f"  Minimum nodes needed: {sum(len(wl) for wl in test_workloads)} (constraint: VMs from same workload on different nodes)")
    print(f"  Nodes available: {len(test_node_capacities)}")
    
    # Test baseline BFD
    print(f"\n{'-'*60}")
    print("BASELINE: Best-Fit Decreasing (BFD)")
    print(f"{'-'*60}")
    
    try:
        bfd_assignments, bfd_bins, bfd_time = best_fit_decreasing(test_workloads, test_node_capacities)
        print(f"BFD Result: {bfd_bins} bins used in {bfd_time:.6f} seconds")
        
        print(f"BFD Node assignments:")
        for node_id in sorted(bfd_assignments.keys()):
            vms = bfd_assignments[node_id]
            capacity_used = sum(vms)
            node_capacity = test_node_capacities[node_id]
            capacity_remaining = node_capacity - capacity_used
            print(f"  Node {node_id}: {vms} → used: {capacity_used}/{node_capacity}, remaining: {capacity_remaining}")
    
    except Exception as e:
        print(f"BFD failed: {e}")
        bfd_bins = float('inf')
        bfd_time = 0
    
    # Test Branch and Bound
    print(f"\n{'-'*60}")
    print("BRANCH AND BOUND (using optimizn)")
    print(f"{'-'*60}")
    
    # Initialize BnB solver
    try:
        bnb_solver = BinPackingBnBProblem(test_workloads, test_node_capacities)
        
        # Solve with BnB using optimizn interface
        start_time = time.time()
        solution, bnb_cost = bnb_solver.solve(iters_limit=5000, log_iters=1000)
        bnb_time = time.time() - start_time
        
        bnb_iterations = bnb_solver.total_iters
        
        # Convert best solution to assignments for comparison
        if solution is not None:
            # Check if solution is complete
            if solution.get('workloads_placed', 0) < len(test_workloads):
                print(f"Warning: BnB found incomplete solution - only {solution.get('workloads_placed', 0)}/{len(test_workloads)} workloads placed")
                print(f"This may indicate insufficient nodes for a complete solution")
                bnb_cost = float('inf')
            else:
                bnb_assignments = solution.get('assignments', {})
                
                print(f"BnB Result: {bnb_cost} bins used in {bnb_time:.6f} seconds")
                print(f"BnB Node assignments:")
                for node_id in sorted(bnb_assignments.keys()):
                    vms_with_workloads = bnb_assignments[node_id]
                    vm_sizes = [vm_size for vm_size, _ in vms_with_workloads]
                    capacity_used = sum(vm_sizes)
                    capacity_remaining = test_node_capacities[node_id] - capacity_used
                    print(f"  Node {node_id}: {vm_sizes} → used: {capacity_used}/{test_node_capacities[node_id]}, remaining: {capacity_remaining}")
                
                # Verify constraint satisfaction
                print(f"\nConstraint verification:")
                for wl_id, node_set in solution['workload_to_nodes'].items():
                    vms_in_workload = len(test_workloads[wl_id])
                    nodes_used = len(node_set)
                    status = "✓" if nodes_used == vms_in_workload else "✗"
                    print(f"  Workload {wl_id}: {vms_in_workload} VMs on {nodes_used} different nodes - {status}")
        else:
            print(f"No solution found - this indicates insufficient nodes for any feasible solution")
            bnb_cost = float('inf')
            bnb_time = 0
            bnb_iterations = 0
    
    except Exception as e:
        print(f"BnB failed: {e}")
        bnb_cost = float('inf')
        bnb_time = 0
        bnb_iterations = 0
    
    # Comparison
    print(f"\n{'-'*60}")
    print("ALGORITHM COMPARISON")
    print(f"{'-'*60}")
    
    print(f"Best-Fit Decreasing (BFD):")
    print(f"  Bins used: {bfd_bins}")
    print(f"  Execution time: {bfd_time:.6f}s")
    
    print(f"\nBranch and Bound:")
    print(f"  Bins used: {bnb_cost}")
    print(f"  Execution time: {bnb_time:.6f}s")
    print(f"  Iterations: {bnb_iterations}")
    
    if bnb_cost < bfd_bins:
        improvement = bfd_bins - bnb_cost
        improvement_pct = (improvement / bfd_bins) * 100
        print(f"\nResult: BnB improved by {improvement} bins ({improvement_pct:.1f}%) ✓")
    elif bnb_cost == bfd_bins:
        print(f"\nResult: BnB matched BFD performance ≈")
    else:
        degradation = bnb_cost - bfd_bins
        degradation_pct = (degradation / bfd_bins) * 100
        print(f"\nResult: BnB used {degradation} more bins ({degradation_pct:.1f}% worse) ✗")
    
    # Handle speed comparison with division by zero protection
    if bnb_time > 0 and bfd_time > 0:
        speedup = bfd_time / bnb_time
        if speedup > 1:
            print(f"Speed: BFD was {speedup:.1f}x faster")
        else:
            print(f"Speed: BnB was {1/speedup:.1f}x faster")
    elif bnb_time == 0:
        print(f"Speed: BnB execution time too small to measure")
    elif bfd_time == 0:
        print(f"Speed: BFD execution time too small to measure")
    else:
        print(f"Speed: Both execution times too small to measure")


def test_bnb_with_different_parameters():
    """
    Test BnB with different parameter settings to find optimal configuration.
    Creates a feasible test case for parameter tuning.
    """
    print(f"\n{'='*80}")
    print("BRANCH AND BOUND PARAMETER TUNING")
    print(f"{'='*80}")
    
    # Generate simple feasible test case for parameter tuning
    print("Creating feasible test case for parameter tuning...")
    simple_workloads = [
        [8],      # Workload 0: 1 VM (needs 1 node)
        [6],      # Workload 1: 1 VM (needs 1 node)
        [4, 3]    # Workload 2: 2 VMs (needs 2 nodes)
    ]
    # Total: 4 VMs, needs 4 nodes minimum
    simple_node_capacities = [16, 16, 16, 16, 16]  # 5 nodes available
    
    print(f"\nParameter tuning with feasible test case:")
    print(f"  Workloads: {simple_workloads}")
    print(f"  Total VMs: {sum(len(wl) for wl in simple_workloads)}")
    print(f"  Minimum nodes needed: {sum(len(wl) for wl in simple_workloads)}")
    print(f"  Nodes available: {len(simple_node_capacities)}")
    
    # Get BFD baseline using imported function from FFD_and_BFD
    bfd_assignments, bfd_bins, bfd_time = best_fit_decreasing(simple_workloads, simple_node_capacities)
    print(f"  BFD baseline: {bfd_bins} bins")
    
    # Test different parameter combinations
    parameter_sets = [
        {"iters_limit": 1000, "log_iters": 500},
        {"iters_limit": 2000, "log_iters": 500},
        {"iters_limit": 3000, "log_iters": 1000},
        {"iters_limit": 5000, "log_iters": 1000},
    ]
    
    best_params = None
    best_cost = float('inf')
    best_time = float('inf')
    
    print(f"\n{'-'*60}")
    print("PARAMETER TESTING")
    print(f"{'-'*60}")
    
    for i, params in enumerate(parameter_sets, 1):
        print(f"\nTest {i}: {params}")
        
        try:
            bnb_solver = BinPackingBnBProblem(simple_workloads, simple_node_capacities)
            
            start_time = time.time()
            solution, bnb_cost = bnb_solver.solve(**params)
            bnb_time = time.time() - start_time
            
            bnb_iterations = bnb_solver.total_iters
            
            print(f"  Result: {bnb_cost} bins in {bnb_time:.4f}s")
            print(f"  Iterations: {bnb_iterations}")
            
            # Track best configuration
            if bnb_cost < best_cost or (bnb_cost == best_cost and bnb_time < best_time):
                best_params = params
                best_cost = bnb_cost
                best_time = bnb_time
                
        except Exception as e:
            print(f"  Failed: {e}")
    
    print(f"\n{'-'*60}")
    print("BEST PARAMETER CONFIGURATION")
    print(f"{'-'*60}")
    print(f"Best parameters: {best_params}")
    print(f"Best result: {best_cost} bins in {best_time:.4f}s")
    print(f"Improvement over BFD: {bfd_bins - best_cost} bins")


def test_multiple_scenarios():
    """
    Test BnB with multiple generated scenarios of different sizes and complexities.
    Compare performance between Branch and Bound and Best-Fit Decreasing.
    """
    print(f"\n{'='*80}")
    print("MULTIPLE SCENARIO TESTING: BnB vs BFD PERFORMANCE")
    print(f"{'='*80}")
    
    # Test scenarios with different complexities
    scenarios = [
        {"name": "Small", "workloads": 10, "nodes": 20, "capacity": 12, "max_vms": 6},
        {"name": "Medium", "workloads": 50, "nodes": 100, "capacity": 16, "max_vms": 15},
        {"name": "Large", "workloads": 100, "nodes": 200, "capacity": 32, "max_vms": 15},
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'-'*60}")
        print(f"SCENARIO: {scenario['name']} ({scenario['workloads']} workloads)")
        print(f"{'-'*60}")
        
        # Generate test case
        workloads, node_capacities = generate_test_cases(
            workload_size=scenario['workloads'],
            number_of_nodes=scenario['nodes'],
            bin_capacity=scenario['capacity'],
            max_vms=scenario['max_vms']
        )
        
        total_vms = sum(len(wl) for wl in workloads)
        total_capacity_needed = sum(sum(wl) for wl in workloads)
        print(f"Generated {len(workloads)} workloads with {total_vms} total VMs")
        print(f"Total capacity needed: {total_capacity_needed}, Available: {len(node_capacities) * scenario['capacity']}")
        
        scenario_result = {
            'name': scenario['name'],
            'workloads': len(workloads),
            'total_vms': total_vms,
            'capacity_needed': total_capacity_needed
        }
        
        # Test BFD baseline
        try:
            bfd_assignments, bfd_bins, bfd_time = best_fit_decreasing(workloads, node_capacities)
            print(f"BFD: {bfd_bins} bins in {bfd_time:.6f}s")
            scenario_result['bfd_bins'] = bfd_bins
            scenario_result['bfd_time'] = bfd_time
        except Exception as e:
            print(f"BFD failed: {e}")
            scenario_result['bfd_bins'] = float('inf')
            scenario_result['bfd_time'] = 0
        
        # Test BnB
        bnb_solver = BinPackingBnBProblem(workloads, node_capacities)
        try:
            start_time = time.time()
            bnb_solver.solve(iters_limit=2000, log_iters=1000)
            bnb_time = time.time() - start_time
            
            bnb_cost = bnb_solver.best_cost
            bnb_iterations = bnb_solver.total_iters
            
            scenario_result['bnb_bins'] = bnb_cost
            scenario_result['bnb_time'] = bnb_time
            scenario_result['bnb_iterations'] = bnb_iterations
            
            improvement = scenario_result['bfd_bins'] - bnb_cost if scenario_result['bfd_bins'] != float('inf') else 0
            improvement_pct = (improvement / scenario_result['bfd_bins'] * 100) if scenario_result['bfd_bins'] > 0 else 0
            
            print(f"BnB: {bnb_cost} bins in {bnb_time:.6f}s ({bnb_iterations} iterations)")
            print(f"Improvement: {improvement} bins ({improvement_pct:.1f}%)")
            
            scenario_result['improvement'] = improvement
            scenario_result['improvement_pct'] = improvement_pct
            
        except Exception as e:
            print(f"BnB failed: {e}")
            scenario_result['bnb_bins'] = float('inf')
            scenario_result['bnb_time'] = 0
            scenario_result['improvement'] = 0
            scenario_result['improvement_pct'] = 0
        
        results.append(scenario_result)
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Scenario':<10} {'BFD Bins':<10} {'BnB Bins':<10} {'Improvement':<12} {'BFD Time':<12} {'BnB Time':<12} {'Speedup':<10}")
    print(f"{'-'*80}")
    
    total_improvement = 0
    scenarios_with_improvement = 0
    
    for result in results:
        if result['bfd_bins'] != float('inf') and result['bnb_bins'] != float('inf'):
            speedup = result['bfd_time'] / result['bnb_time'] if result['bnb_time'] > 0 else float('inf')
            speedup_str = f"{speedup:.1f}x" if speedup != float('inf') else "∞"
            
            print(f"{result['name']:<10} {result['bfd_bins']:<10} {result['bnb_bins']:<10} "
                  f"{result['improvement_pct']:<11.1f}% {result['bfd_time']:<11.6f}s {result['bnb_time']:<11.6f}s {speedup_str:<10}")
            
            if result['improvement'] > 0:
                scenarios_with_improvement += 1
            total_improvement += result['improvement_pct']
    
    avg_improvement = total_improvement / len(results) if results else 0
    
    print(f"\nOverall Results:")
    print(f"  Average improvement: {avg_improvement:.1f}%")
    print(f"  Scenarios with improvement: {scenarios_with_improvement}/{len(results)}")
    print(f"  Success rate: {scenarios_with_improvement/len(results)*100:.1f}%")


def test_node_constraint_validation():
    """
    Test cases to demonstrate how the BnB algorithm handles node constraints.
    """
    print("="*80)
    print("NODE CONSTRAINT VALIDATION TEST")
    print("="*80)
    
    # Test Case 1: Feasible problem
    print("\nTest Case 1: Feasible Problem")
    print("-" * 40)
    workloads1 = [[5, 3], [4, 2], [6]]
    capacities1 = [8, 8, 8]  # 3 nodes of capacity 8 each
    
    print(f"Workloads: {workloads1}")
    print(f"Node capacities: {capacities1}")
    print(f"Total VM capacity needed: {sum(sum(wl) for wl in workloads1)}")
    print(f"Total available capacity: {sum(capacities1)}")
    
    try:
        # Test BFD
        bfd_result, bfd_bins, bfd_time = best_fit_decreasing(workloads1, capacities1)
        print(f"✓ BFD succeeded: {bfd_bins} bins used")
        
        # Test BnB
        bnb_solver = BinPackingBnBProblem(workloads1, capacities1)
        bnb_solver.solve(iters_limit=1000)
        print(f"✓ BnB succeeded: {bnb_solver.best_cost} bins used")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test Case 2: Insufficient total capacity
    print("\nTest Case 2: Insufficient Total Capacity")
    print("-" * 45)
    workloads2 = [[10, 8], [9, 7], [6, 5]]
    capacities2 = [16, 16]  # Only 32 total capacity, need 45
    
    print(f"Workloads: {workloads2}")
    print(f"Node capacities: {capacities2}")
    print(f"Total VM capacity needed: {sum(sum(wl) for wl in workloads2)}")
    print(f"Total available capacity: {sum(capacities2)}")
    
    try:
        # Test BFD first
        bfd_result, bfd_bins, bfd_time = best_fit_decreasing(workloads2, capacities2)
        print(f"BFD result: {bfd_bins} bins used")
    except Exception as e:
        print(f"✗ BFD failed: {e}")
    
    try:
        # Test BnB - should fail at initialization
        bnb_solver = BinPackingBnBProblem(workloads2, capacities2)
        print(f"✗ BnB should have failed but didn't")
    except Exception as e:
        print(f"✓ BnB correctly failed: {e}")
    
    # Test Case 3: VM too large for any single node
    print("\nTest Case 3: VM Too Large for Any Node")
    print("-" * 42)
    workloads3 = [[20], [5], [3]]  # First VM is size 20, larger than any node
    capacities3 = [16, 16, 16]  # All nodes have capacity 16
    
    print(f"Workloads: {workloads3}")
    print(f"Node capacities: {capacities3}")
    print(f"Largest VM: {max(max(wl) for wl in workloads3)}")
    print(f"Largest node capacity: {max(capacities3)}")
    
    try:
        # Test BFD first
        bfd_result, bfd_bins, bfd_time = best_fit_decreasing(workloads3, capacities3)
        print(f"BFD result: {bfd_bins} bins used")
    except Exception as e:
        print(f"✗ BFD failed: {e}")
    
    try:
        # Test BnB - should fail at initialization
        bnb_solver = BinPackingBnBProblem(workloads3, capacities3)
        print(f"✗ BnB should have failed but didn't")
    except Exception as e:
        print(f"✓ BnB correctly failed: {e}")
    
    # Test Case 4: Too many workloads for available nodes
    print("\nTest Case 4: Too Many Workloads for Available Nodes")
    print("-" * 52)
    workloads4 = [[1], [1], [1], [1], [1]]  # 5 workloads
    capacities4 = [16, 16, 16]  # Only 3 nodes available
    
    print(f"Workloads: {workloads4}")
    print(f"Node capacities: {capacities4}")
    print(f"Number of workloads: {len(workloads4)}")
    print(f"Number of nodes: {len(capacities4)}")
    
    try:
        # Test BFD first
        bfd_result, bfd_bins, bfd_time = best_fit_decreasing(workloads4, capacities4)
        print(f"BFD result: {bfd_bins} bins used")
    except Exception as e:
        print(f"✗ BFD failed: {e}")
    
    try:
        # Test BnB - should fail at initialization
        bnb_solver = BinPackingBnBProblem(workloads4, capacities4)
        print(f"✗ BnB should have failed but didn't")
    except Exception as e:
        print(f"✓ BnB correctly failed: {e}")


def test_workload_order_branching():
    """
    Test to demonstrate that the enhanced BnB explores all possible workload orderings.
    Given workloads [0, 3] placed, it should try [0, 3, 1], [0, 3, 2], [0, 3, 4].
    """
    print("="*80)
    print("WORKLOAD ORDER BRANCHING TEST")
    print("="*80)
    
    # Create a test case where order matters - use more nodes to avoid constraint issues
    vm_lists = [
        [5],     # Workload 0: one VM of size 5
        [4],     # Workload 1: one VM of size 4  
        [3],     # Workload 2: one VM of size 3
        [6],     # Workload 3: one VM of size 6
        [2],     # Workload 4: one VM of size 2
    ]
    node_capacities = [8, 8, 8, 8, 8, 8]  # Six nodes to satisfy constraint
    
    print(f"Test Case:")
    print(f"  Workloads: {vm_lists}")
    print(f"  Node capacities: {node_capacities}")
    print(f"  Question: If we place workloads [0, 3], what are the next branches?")
    print(f"  Expected: Should try workloads 1, 2, and 4 as next candidates")
    
    # Create BnB problem
    bnb_problem = BinPackingBnBProblem(vm_lists, node_capacities)
    
    print(f"\n{'-'*60}")
    print("STEP-BY-STEP BRANCHING ANALYSIS")
    print(f"{'-'*60}")
    
    # Start from root
    root = bnb_problem.get_root()
    print(f"1. Root solution:")
    print(f"   placed_workloads: {root['placed_workloads']}")
    print(f"   workload_order: {root['workload_order']}")
    
    # First level: branches from root (should try all workloads 0,1,2,3,4)
    first_level_branches = list(bnb_problem.branch(root))
    print(f"\n2. First level branches from root ({len(first_level_branches)} branches):")
    for i, branch in enumerate(first_level_branches):
        workload_added = branch['workload_order'][-1] if branch['workload_order'] else 'None'
        print(f"   Branch {i+1}: Added workload {workload_added}, order: {branch['workload_order']}")
    
    # Find the branch that placed workload 0
    workload_0_branch = None
    for branch in first_level_branches:
        if branch['workload_order'] == [0]:
            workload_0_branch = branch
            break
    
    if workload_0_branch:
        print(f"\n3. Selected branch with workload 0 placed:")
        print(f"   placed_workloads: {workload_0_branch['placed_workloads']}")
        print(f"   workload_order: {workload_0_branch['workload_order']}")
        print(f"   assignments: {workload_0_branch['assignments']}")
        
        # Second level: branches from workload 0 (should try workloads 1,2,3,4)
        second_level_branches = list(bnb_problem.branch(workload_0_branch))
        print(f"\n4. Second level branches from [0] ({len(second_level_branches)} branches):")
        for i, branch in enumerate(second_level_branches):
            workload_added = branch['workload_order'][-1] if len(branch['workload_order']) > 1 else 'None'
            print(f"   Branch {i+1}: Added workload {workload_added}, order: {branch['workload_order']}")
        
        # Find the branch that placed workloads [0, 3]
        workload_0_3_branch = None
        for branch in second_level_branches:
            if len(branch['workload_order']) >= 2 and branch['workload_order'] == [0, 3]:
                workload_0_3_branch = branch
                break
        
        if workload_0_3_branch:
            print(f"\n5. Selected branch with workloads [0, 3] placed:")
            print(f"   placed_workloads: {workload_0_3_branch['placed_workloads']}")
            print(f"   workload_order: {workload_0_3_branch['workload_order']}")
            print(f"   assignments: {workload_0_3_branch['assignments']}")
            
            # Third level: branches from [0, 3] (should try workloads 1,2,4)
            third_level_branches = list(bnb_problem.branch(workload_0_3_branch))
            print(f"\n6. Third level branches from [0, 3] ({len(third_level_branches)} branches):")
            print(f"   Expected: Should explore [0,3,1], [0,3,2], [0,3,4]")
            for i, branch in enumerate(third_level_branches):
                workload_added = branch['workload_order'][-1] if len(branch['workload_order']) > 2 else 'None'
                print(f"   Branch {i+1}: Added workload {workload_added}, order: {branch['workload_order']}")
                print(f"            assignments: {branch['assignments']}")
            
            # Verify we got the expected combinations
            expected_orders = [[0, 3, 1], [0, 3, 2], [0, 3, 4]]
            actual_orders = [branch['workload_order'] for branch in third_level_branches]
            
            print(f"\n{'-'*40}")
            print("VERIFICATION")
            print(f"{'-'*40}")
            print(f"Expected orders: {expected_orders}")
            print(f"Actual orders:   {actual_orders}")
            
            if set(map(tuple, expected_orders)) == set(map(tuple, actual_orders)):
                print("✅ SUCCESS: All expected workload combinations were explored!")
            else:
                print("❌ FAILURE: Missing or extra combinations found")
                missing = set(map(tuple, expected_orders)) - set(map(tuple, actual_orders))
                extra = set(map(tuple, actual_orders)) - set(map(tuple, expected_orders))
                if missing:
                    print(f"   Missing: {list(missing)}")
                if extra:
                    print(f"   Extra: {list(extra)}")
        else:
            print("❌ Could not find branch with workloads [0, 3]")
    else:
        print("❌ Could not find branch with workload 0")
    
    print(f"\n{'='*80}")

# Run the tests
if __name__ == "__main__":
    test_branch_and_bound()
    test_bnb_with_different_parameters()
    test_multiple_scenarios()
    # test_node_constraint_validation()
    # test_workload_order_branching()