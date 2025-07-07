# %%
# Simulated Annealing for Bin Packing Problem using optimizn library

from optimizn.combinatorial.simulated_annealing import SimAnnealProblem
import random
from typing import List, Tuple, Dict, Any

# Import BFD and test case generator from the main FFD_and_BFD file
from FFD_and_BFD import best_fit_decreasing, generate_test_cases


class BinPackingSimulatedAnnealing(SimAnnealProblem):
    """
    Simulated Annealing solver for the bin packing problem using optimizn library.
    """
    
    def __init__(self, vm_lists: List[List[int]], node_capacities: List[int], params=None, logger=None):
        """
        Initialize the Simulated Annealing solver.
        
        Args:
            vm_lists: List of lists, where each inner list contains VM configurations
            node_capacities: List of capacities for each node
            params: Parameters for the optimizer (optional)
            logger: Logger instance (optional)
        """
        self.vm_lists = vm_lists
        self.node_capacities = node_capacities
        
        # Flatten all VMs for easier manipulation
        self.all_vms = []
        for group_id, vm_list in enumerate(vm_lists):
            for vm in vm_list:
                self.all_vms.append((vm, group_id))
        
        self.original_order = self.all_vms.copy()
        self.total_vms = len(self.all_vms)
        
        # Set default params if none provided
        if params is None:
            params = {
                'vm_lists': vm_lists,
                'node_capacities': node_capacities,
                'total_vms': self.total_vms
            }
        
        print(f"Initialized SA for {len(vm_lists)} workloads with {self.total_vms} total VMs")
        
        # Initialize the parent class
        super().__init__(params, logger)
    
    def get_initial_solution(self) -> List[Tuple[int, int]]:
        """
        Get initial solution that exactly matches BFD placement.
        This ensures the initial cost equals BFD cost.
        
        Returns:
            List of (vm_size, group_id) tuples that when processed will yield same cost as BFD
        """
        try:
            # Create the exact same VM ordering that BFD uses internally
            # This is how BFD processes VMs: all VMs sorted by size descending
            vm_with_groups = []
            for group_id, vm_list in enumerate(self.vm_lists):
                for vm in vm_list:
                    vm_with_groups.append((vm, group_id))
            
            # Sort all VMs by size in descending order (exactly like BFD does)
            vm_ordering = sorted(vm_with_groups, key=lambda x: x[0], reverse=True)
            
            print(f"Initial solution created using BFD ordering: {len(vm_ordering)} VMs")
            return vm_ordering
            
        except Exception as e:
            print(f"Error in BFD initial solution: {e}")
            # Fallback to sorted order
            return sorted(self.all_vms, key=lambda x: x[0], reverse=True)
    
    def next_candidate(self, current_solution: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Generate next candidate solution by randomly swapping two VM configurations.
        
        Args:
            current_solution: Current solution as list of (vm_size, group_id) tuples
            
        Returns:
            New candidate solution with two VMs swapped
        """
        # Create a copy of the current solution
        new_solution = current_solution.copy()
        
        # Randomly select two different positions to swap
        if len(new_solution) < 2:
            return new_solution
        
        pos1 = random.randint(0, len(new_solution) - 1)
        pos2 = random.randint(0, len(new_solution) - 1)
        
        # Ensure we're swapping different positions
        while pos2 == pos1:
            pos2 = random.randint(0, len(new_solution) - 1)
        
        # Swap the VMs
        new_solution[pos1], new_solution[pos2] = new_solution[pos2], new_solution[pos1]
        
        return new_solution
    
    def cost(self, solution: List[Tuple[int, int]]) -> int:
        """
        Calculate the cost of a solution (number of bins used).
        This applies best-fit algorithm with group constraints to match BFD exactly.
        
        Args:
            solution: Solution as list of (vm_size, group_id) tuples in current order
            
        Returns:
            Number of bins used (cost to minimize)
        """
        try:
            # Apply best-fit algorithm with group constraints (same as BFD)
            # Do NOT sort - respect the ordering in the solution
            bins = []
            group_to_nodes = {}  # Track which nodes each group has used
            
            for vm_size, group_id in solution:
                # Initialize group tracking if not seen before
                if group_id not in group_to_nodes:
                    group_to_nodes[group_id] = set()
                
                # Try to fit the VM into an existing node using Best-Fit strategy
                best_node_id = None
                best_remaining_capacity = float('inf')
                
                for node_id, node in enumerate(bins):
                    # Check if node has capacity and is not already used by this group
                    if node_id not in group_to_nodes[group_id]:
                        current_capacity = sum(node)
                        remaining_capacity = self.node_capacities[node_id] - current_capacity
                        
                        # Check if VM fits and if this is the best fit so far
                        if (vm_size <= remaining_capacity and 
                            remaining_capacity < best_remaining_capacity):
                            best_node_id = node_id
                            best_remaining_capacity = remaining_capacity
                
                # Place the VM in the best-fit node or create a new node
                if best_node_id is not None:
                    bins[best_node_id].append(vm_size)
                    group_to_nodes[group_id].add(best_node_id)
                else:
                    # Need a new bin
                    if len(bins) >= len(self.node_capacities):
                        # No more bins available - return penalty
                        return len(self.node_capacities) + 100
                    
                    # Create new bin
                    bin_id = len(bins)
                    bins.append([vm_size])
                    group_to_nodes[group_id].add(bin_id)
            
            return len(bins)
            
        except Exception as e:
            print(f"Error in cost calculation: {e}")
            # Return a high penalty cost if calculation fails
            return len(self.node_capacities)
    
    def calculate_cost(self, solution: List[Tuple[int, int]]) -> int:
        """
        Backward compatibility method. Delegates to cost().
        
        Args:
            solution: Solution as list of (vm_size, group_id) tuples
            
        Returns:
            Number of bins used (cost to minimize)
        """
        return self.cost(solution)
    
    def next_candidate(self, current_solution: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Generate next candidate solution by randomly swapping two VM configurations.
        
        Args:
            current_solution: Current solution as list of (vm_size, group_id) tuples
            
        Returns:
            New candidate solution with two VMs swapped
        """
        # Create a copy of the current solution
        new_solution = current_solution.copy()
        
        # Randomly select two different positions to swap
        if len(new_solution) < 2:
            return new_solution
        
        pos1 = random.randint(0, len(new_solution) - 1)
        pos2 = random.randint(0, len(new_solution) - 1)
        
        # Ensure we're swapping different positions
        while pos2 == pos1:
            pos2 = random.randint(0, len(new_solution) - 1)
        
        # Swap the VMs
        new_solution[pos1], new_solution[pos2] = new_solution[pos2], new_solution[pos1]
        
        return new_solution
    
    def solution_to_vm_lists(self, solution: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Convert solution back to vm_lists format preserving the VM ordering.
        The ordering matters for the bin packing algorithm.
        
        Args:
            solution: List of (vm_size, group_id) tuples in the desired order
            
        Returns:
            List of lists in vm_lists format with VMs in the specified order
        """
        # Create vm_lists with the ordering preserved
        vm_lists_ordered = [[] for _ in range(len(self.vm_lists))]
        
        # Process VMs in the order specified by the solution
        for vm_size, group_id in solution:
            vm_lists_ordered[group_id].append(vm_size)
        
        return vm_lists_ordered

    def solve_with_params(self, n_iter: int = 10000, reset_p: float = 1/10000, 
                         time_limit: int = 3600, log_iters: int = 1000) -> Tuple[List[Tuple[int, int]], int, Dict[str, Any]]:
        """
        Solve the bin packing problem using the optimizn Simulated Annealing.
        
        Args:
            n_iter: Number of iterations
            reset_p: Reset probability
            time_limit: Time limit in seconds
            log_iters: Logging interval
            
        Returns:
            tuple: (best_solution, best_cost, stats)
        """
        print(f"Starting SA with optimizn: {n_iter} iterations, time limit: {time_limit}s")
        
        # Use the optimizn anneal method
        best_solution, best_cost = self.anneal(
            n_iter=n_iter,
            reset_p=reset_p,
            time_limit=time_limit,
            log_iters=log_iters
        )
        
        # Create stats dictionary for compatibility
        stats = {
            'iterations': self.total_iters,
            'execution_time': self.total_time_elapsed,
            'initial_cost': self.init_cost,
            'final_cost': self.current_cost,
            'best_cost': best_cost,
            'total_moves': self.total_iters,  # Approximation
            'accepted_moves': -1,  # Not tracked in optimizn version
            'acceptance_rate': -1,  # Not tracked in optimizn version
            'cost_history': [],
            'temperature_history': [],
            'temperature_final': -1
        }
        
        print(f"\nSA completed:")
        print(f"  Iterations: {self.total_iters}")
        print(f"  Time elapsed: {self.total_time_elapsed:.4f}s")
        print(f"  Initial cost: {self.init_cost} bins")
        print(f"  Best cost: {best_cost} bins")
        improvement = self.init_cost - best_cost
        improvement_pct = (improvement / self.init_cost * 100) if self.init_cost > 0 else 0
        print(f"  Improvement: {improvement} bins ({improvement_pct:.1f}%)")
        
        return best_solution, best_cost, stats
    
    def solution_to_assignments(self, solution: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """
        Convert a solution to node assignments using BFD algorithm.
        
        Args:
            solution: Solution as list of (vm_size, group_id) tuples in order
            
        Returns:
            Dictionary mapping node_id to list of VM sizes
        """
        # Convert solution to vm_lists format
        vm_lists_ordered = self.solution_to_vm_lists(solution)
        
        try:
            # Use BFD to get the actual assignments
            assignments, _, _ = best_fit_decreasing(vm_lists_ordered, self.node_capacities)
            return assignments
        except Exception as e:
            print(f"Error in solution_to_assignments: {e}")
            return {}

# %%
# =======================================================================
# SIMULATED ANNEALING TEST AND COMPARISON
# =======================================================================

def test_simulated_annealing():
    """
    Test the Simulated Annealing implementation and compare with BFD.
    """
    print("="*80)
    print("SIMULATED ANNEALING BIN PACKING TEST")
    print("="*80)
    
    # Generate test case using FFD_and_BFD function
    print("Generating test case...")
    test_workloads, test_node_capacities = generate_test_cases(
        workload_size=5,     # 5 workloads
        number_of_nodes=8,   # 8 nodes
        bin_capacity=16,     # capacity 16 each
        max_vms=6            # max 6 VMs per workload
    )
    
    print(f"\nGenerated Test Case:")
    print(f"  Workloads: {test_workloads}")
    print(f"  Node capacities: {test_node_capacities}")
    print(f"  Total VMs: {sum(len(wl) for wl in test_workloads)}")
    print(f"  Total capacity needed: {sum(sum(wl) for wl in test_workloads)}")
    print(f"  Total available capacity: {sum(test_node_capacities)}")
    
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
    
    # Test Simulated Annealing
    print(f"\n{'-'*60}")
    print("SIMULATED ANNEALING (using optimizn)")
    print(f"{'-'*60}")
    
    # Initialize SA solver
    sa_solver = BinPackingSimulatedAnnealing(test_workloads, test_node_capacities)
    
    # Solve with SA using optimizn interface
    try:
        sa_solution, sa_cost, sa_stats = sa_solver.solve_with_params(
            n_iter=5000,
            reset_p=1/5000,
            time_limit=60,
            log_iters=1000
        )
        
        # Convert to assignments for comparison
        sa_assignments = sa_solver.solution_to_assignments(sa_solution)
        
        print(f"\nSA Node assignments:")
        for node_id in sorted(sa_assignments.keys()):
            vms = sa_assignments[node_id]
            capacity_used = sum(vms)
            capacity_remaining = test_node_capacities[node_id] - capacity_used
            print(f"  Node {node_id}: {vms} → used: {capacity_used}/16, remaining: {capacity_remaining}")
    
    except Exception as e:
        print(f"SA failed: {e}")
        sa_cost = float('inf')
        sa_stats = {'execution_time': 0}
    
    # Comparison
    print(f"\n{'-'*60}")
    print("ALGORITHM COMPARISON")
    print(f"{'-'*60}")
    
    print(f"Best-Fit Decreasing (BFD):")
    print(f"  Bins used: {bfd_bins}")
    print(f"  Execution time: {bfd_time:.6f}s")
    
    print(f"\nSimulated Annealing:")
    print(f"  Bins used: {sa_cost}")
    print(f"  Execution time: {sa_stats['execution_time']:.6f}s")
    print(f"  Iterations: {sa_stats.get('iterations', 'N/A')}")
    print(f"  Acceptance rate: {sa_stats.get('acceptance_rate', 'N/A'):.3f}")
    
    if sa_cost < bfd_bins:
        improvement = bfd_bins - sa_cost
        improvement_pct = (improvement / bfd_bins) * 100
        print(f"\nResult: SA improved by {improvement} bins ({improvement_pct:.1f}%) ✓")
    elif sa_cost == bfd_bins:
        print(f"\nResult: SA matched BFD performance ≈")
    else:
        degradation = sa_cost - bfd_bins
        degradation_pct = (degradation / bfd_bins) * 100
        print(f"\nResult: SA used {degradation} more bins ({degradation_pct:.1f}% worse) ✗")
    
    # Handle speed comparison with division by zero protection
    if sa_stats['execution_time'] > 0 and bfd_time > 0:
        speedup = bfd_time / sa_stats['execution_time']
        if speedup > 1:
            print(f"Speed: BFD was {speedup:.1f}x faster")
        else:
            print(f"Speed: SA was {1/speedup:.1f}x faster")
    elif sa_stats['execution_time'] == 0:
        print(f"Speed: SA execution time too small to measure")
    elif bfd_time == 0:
        print(f"Speed: BFD execution time too small to measure")
    else:
        print(f"Speed: Both execution times too small to compare")


def test_sa_with_different_parameters():
    """
    Test SA with different parameter settings to find optimal configuration.
    Uses the "Small" scenario from multiple scenario test for more meaningful differentiation.
    """
    print(f"\n{'='*80}")
    print("SIMULATED ANNEALING PARAMETER TUNING")
    print(f"{'='*80}")
    
    # Generate "Small" scenario test case for parameter tuning (same as in multiple_scenarios)
    print("Generating 'Small' scenario test case for parameter tuning...")
    simple_workloads, simple_node_capacities = generate_test_cases(
        workload_size=10,    # 10 workloads (increased from 3)
        number_of_nodes=20,  # 20 nodes (increased from 5)
        bin_capacity=12,     # capacity 12 each (decreased from 16)
        max_vms=6           # max 6 VMs per workload (increased from 3)
    )
    
    print(f"\nParameter tuning with 'Small' scenario test case:")
    print(f"  Workloads: {len(simple_workloads)}")
    print(f"  Total VMs: {sum(len(wl) for wl in simple_workloads)}")
    print(f"  Total capacity needed: {sum(sum(wl) for wl in simple_workloads)}")
    print(f"  Total available capacity: {sum(simple_node_capacities)}")
    
    # Get BFD baseline using imported function from FFD_and_BFD
    bfd_assignments, bfd_bins, bfd_time = best_fit_decreasing(simple_workloads, simple_node_capacities)
    print(f"  BFD baseline: {bfd_bins} bins in {bfd_time:.6f}s")
    
    # Test different parameter combinations
    parameter_sets = [
        {"n_iter": 1000, "reset_p": 1/1000, "time_limit": 30, "log_iters": 500},
        {"n_iter": 2000, "reset_p": 1/2000, "time_limit": 60, "log_iters": 500},
        {"n_iter": 3000, "reset_p": 1/5000, "time_limit": 90, "log_iters": 1000},
        {"n_iter": 5000, "reset_p": 1/10000, "time_limit": 120, "log_iters": 1000},
    ]
    
    best_params = None
    best_cost = float('inf')
    best_time = float('inf')
    
    print(f"\n{'-'*60}")
    print("PARAMETER TESTING")
    print(f"{'-'*60}")
    
    for i, params in enumerate(parameter_sets, 1):
        print(f"\nTest {i}: {params}")
        
        sa_solver = BinPackingSimulatedAnnealing(simple_workloads, simple_node_capacities)
        
        try:
            _, sa_cost, sa_stats = sa_solver.solve_with_params(**params)
            
            print(f"  Result: {sa_cost} bins in {sa_stats['execution_time']:.4f}s")
            print(f"  Iterations: {sa_stats['iterations']}")
            
            # Track best configuration
            if sa_cost < best_cost or (sa_cost == best_cost and sa_stats['execution_time'] < best_time):
                best_params = params
                best_cost = sa_cost
                best_time = sa_stats['execution_time']
                
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
    Test SA with multiple generated scenarios of different sizes and complexities.
    Compare performance between Simulated Annealing and Best-Fit Decreasing.
    """
    print(f"\n{'='*80}")
    print("MULTIPLE SCENARIO TESTING: SA vs BFD PERFORMANCE")
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
        
        # Test SA
        sa_solver = BinPackingSimulatedAnnealing(workloads, node_capacities)
        try:
            _, sa_cost, sa_stats = sa_solver.solve_with_params(
                n_iter=2000,
                reset_p=1/5000,
                time_limit=60,
                log_iters=1000
            )
            
            scenario_result['sa_bins'] = sa_cost
            scenario_result['sa_time'] = sa_stats.get('execution_time', 0)
            scenario_result['sa_iterations'] = sa_stats.get('iterations', 0)
            
            improvement = scenario_result['bfd_bins'] - sa_cost if scenario_result['bfd_bins'] != float('inf') else 0
            improvement_pct = (improvement / scenario_result['bfd_bins'] * 100) if scenario_result['bfd_bins'] > 0 else 0
            
            print(f"SA: {sa_cost} bins in {scenario_result['sa_time']:.6f}s ({scenario_result['sa_iterations']} iterations)")
            print(f"Improvement: {improvement} bins ({improvement_pct:.1f}%)")
            
            scenario_result['improvement'] = improvement
            scenario_result['improvement_pct'] = improvement_pct
            
        except Exception as e:
            print(f"SA failed: {e}")
            scenario_result['sa_bins'] = float('inf')
            scenario_result['sa_time'] = 0
            scenario_result['improvement'] = 0
            scenario_result['improvement_pct'] = 0
        
        results.append(scenario_result)
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Scenario':<10} {'BFD Bins':<10} {'SA Bins':<10} {'Improvement':<12} {'BFD Time':<12} {'SA Time':<12} {'Speedup':<10}")
    print(f"{'-'*80}")
    
    total_improvement = 0
    scenarios_with_improvement = 0
    
    for result in results:
        if result['bfd_bins'] != float('inf') and result['sa_bins'] != float('inf'):
            speedup = result['bfd_time'] / result['sa_time'] if result['sa_time'] > 0 else float('inf')
            speedup_str = f"{speedup:.1f}x" if speedup != float('inf') else "∞"
            
            print(f"{result['name']:<10} {result['bfd_bins']:<10} {result['sa_bins']:<10} "
                  f"{result['improvement_pct']:<11.1f}% {result['bfd_time']:<11.6f}s {result['sa_time']:<11.6f}s {speedup_str:<10}")
            
            if result['improvement'] > 0:
                scenarios_with_improvement += 1
            total_improvement += result['improvement_pct']
    
    avg_improvement = total_improvement / len(results) if results else 0
    
    print(f"\nOverall Results:")
    print(f"  Average improvement: {avg_improvement:.1f}%")
    print(f"  Scenarios with improvement: {scenarios_with_improvement}/{len(results)}")
    print(f"  Success rate: {scenarios_with_improvement/len(results)*100:.1f}%")


# Run the tests
if __name__ == "__main__":
    test_simulated_annealing()
    test_sa_with_different_parameters()
    test_multiple_scenarios()
# %%
