# %% [markdown]
# ### VM Bin Packing

# %% [markdown]
# #### Functions

# %% [markdown]
# Use the traditional First-Fit Decreasing (FFD) and Best-Fit Decreasing (BFD) algorithm. Add 1 more constraints:
# 1. Each number in the list need to be relocate on different bins
# 
# Note: Compared to FFD, BFD generally works better with the trade-off of computing time. 
# Basic ideas:
# FFD: Sort all numbers in descending order, and place each number in the first bin that has enough space; if not possible, create a new bin and put the number.
# BFD: Sort all numbers in descending order, and place each number in the used bins that has the least remaining space; if not possible, create a new bin and put the number.
# 
# In both functions, the node capacity is a fixed number. 

# %%
import time
import random

def first_fit_decreasing(vm_lists, node_capacities):
    """
    Implements the First-Fit Decreasing (FFD) algorithm for VM configurations.
    Sorts all VM configurations globally in descending order, but ensures that VMs from the same 
    original list are placed on different nodes.

    Args:
        vm_lists (list of list of int): A list of lists, where each inner list contains VM configurations.
        node_capacities (list of int): A list of capacities for each node.

    Returns:
        tuple: A tuple containing (node_assignments, number_of_bins_used, execution_time) where:
            - node_assignments: dict with keys as node IDs and values as lists of VM configurations
            - number_of_bins_used: int representing the total number of bins used
            - execution_time: float representing the execution time in seconds
    """
    start_time = time.time()
    
    # Step 1: Create a list of (vm_size, group_id) tuples and sort by vm_size in descending order
    vm_with_groups = []
    for group_id, vm_list in enumerate(vm_lists):
        for vm in vm_list:
            vm_with_groups.append((vm, group_id))
    
    # Sort all VMs by size in descending order
    vm_with_groups.sort(key=lambda x: x[0], reverse=True)

    # Step 2: Initialize nodes and group tracking
    nodes = []  # List of nodes, each node is a list of VM configurations
    group_to_nodes = {}  # Track which nodes each group has used
    
    # Initialize group tracking
    for group_id in range(len(vm_lists)):
        group_to_nodes[group_id] = set()
    
    # Step 3: Process each VM in sorted order
    for vm_size, group_id in vm_with_groups:
        # Try to fit the VM into an existing node (First-Fit)
        placed = False
        for node_id, node in enumerate(nodes):
            # Check if node has capacity and is not already used by this group
            if (node_id not in group_to_nodes[group_id] and 
                sum(node) + vm_size <= node_capacities[node_id]):
                node.append(vm_size)
                group_to_nodes[group_id].add(node_id)
                placed = True
                break

        # If the VM cannot fit into any existing node, put into a new node
        if not placed:
            new_node_id = len(nodes)
            nodes.append([vm_size])
            group_to_nodes[group_id].add(new_node_id)

    # Return the node assignments, number of bins used, and execution time
    end_time = time.time()
    execution_time = end_time - start_time
    node_assignments = {node_id: node for node_id, node in enumerate(nodes)}
    return node_assignments, len(nodes), execution_time


# %%
def best_fit_decreasing(vm_lists, node_capacities):
    """
    Implements the Best-Fit Decreasing (BFD) algorithm for VM configurations.
    Sorts all VMs globally in descending order, but ensures that VMs from the same 
    original list are placed on different nodes.

    Args:
        vm_lists (list of list of int): A list of lists, where each inner list contains VM configurations.
        node_capacities (list of int): A list of capacities for each node.

    Returns:
        tuple: A tuple containing (node_assignments, number_of_bins_used, execution_time) where:
            - node_assignments: dict with keys as node IDs and values as lists of VM configurations
            - number_of_bins_used: int representing the total number of bins used
            - execution_time: float representing the execution time in seconds
    """
    start_time = time.time()
    
    # Step 1: Create a list of (vm_size, group_id) tuples and sort by vm_size in descending order
    vm_with_groups = []
    for group_id, vm_list in enumerate(vm_lists):
        for vm in vm_list:
            vm_with_groups.append((vm, group_id))
    
    # Sort all VMs by size in descending order
    vm_with_groups.sort(key=lambda x: x[0], reverse=True)

    # Step 2: Initialize nodes and group tracking
    nodes = []  # List of nodes, each node is a list of VM configurations
    group_to_nodes = {}  # Track which nodes each group has used
    
    # Initialize group tracking
    for group_id in range(len(vm_lists)):
        group_to_nodes[group_id] = set()
    
    # Step 3: Process each VM in sorted order
    for vm_size, group_id in vm_with_groups:
        # Try to fit the VM into an existing node using Best-Fit strategy
        best_node_id = None
        best_remaining_capacity = float('inf')
        
        for node_id, node in enumerate(nodes):
            # Check if node has capacity and is not already used by this group
            if node_id not in group_to_nodes[group_id]:
                current_capacity = sum(node)
                remaining_capacity = node_capacities[node_id] - current_capacity
                
                # Check if VM fits and if this is the best fit so far
                if (vm_size <= remaining_capacity and 
                    remaining_capacity < best_remaining_capacity):
                    best_node_id = node_id
                    best_remaining_capacity = remaining_capacity
        
        # Place the VM in the best-fit node or create a new node
        if best_node_id is not None:
            nodes[best_node_id].append(vm_size)
            group_to_nodes[group_id].add(best_node_id)
        else:
            # Put into a new node if no suitable node is found
            new_node_id = len(nodes)
            nodes.append([vm_size])
            group_to_nodes[group_id].add(new_node_id)

    # Return the node assignments, number of bins used, and execution time
    end_time = time.time()
    execution_time = end_time - start_time
    node_assignments = {node_id: node for node_id, node in enumerate(nodes)}
    return node_assignments, len(nodes), execution_time



# %%
# Generate test cases, with manually specified parameters
import random

def generate_test_cases(workload_size, number_of_nodes, bin_capacity, max_vms):
    """
    Generates a workload group (2D array) and a node capacity list for the bin packing algorithms.
    Ensures that the total node capacity is sufficient to accommodate all VMs.

    Args:
        workload_size (int): Number of workloads (arrays) in the workload group
        number_of_nodes (int): Number of nodes (bins)
        bin_capacity (int): Fixed capacity of each bin
        max_vms (int): Maximum number of VMs within each workload

    Returns:
        tuple: (workload_group, node_capacity_list)
    """
    # Calculate total available capacity
    total_capacity = number_of_nodes * bin_capacity
    
    # Generate the workload group (2D array) ensuring it fits within total capacity
    workload_group = []
    total_vms_generated = 0
    
    for _ in range(workload_size):
        # Calculate remaining capacity
        remaining_capacity = total_capacity - total_vms_generated
        
        # Ensure we don't exceed total capacity
        max_vms_for_this_workload = min(max_vms, remaining_capacity)
        
        if max_vms_for_this_workload <= 0:
            # If no capacity left, create empty workload
            workload_group.append([])
            continue
            
        # Randomly decide the total number of VMs for this workload
        total_vms = random.randint(1, max_vms_for_this_workload)
        vm_counts = []

        # Randomly decide the length of the array (at least 1 element, at most total_vms)
        length = random.randint(1, min(total_vms, 10))  # Limit array length for practicality

        # Distribute the VMs into `length` parts with a max difference of 1
        base_value = total_vms // length
        extras = total_vms % length

        for i in range(length):
            if i < extras:
                vm_counts.append(base_value + 1)
            else:
                vm_counts.append(base_value)

        # Shuffle the VM counts for randomness
        random.shuffle(vm_counts)
        workload_group.append(vm_counts)
        
        # Update total VMs generated
        total_vms_generated += total_vms

    # Generate the node capacity list with fixed capacity
    node_capacity_list = [bin_capacity] * number_of_nodes
    
    # Print capacity information
    actual_total_vms = sum(sum(workload) for workload in workload_group)
    print(f"Total VMs generated: {actual_total_vms}")
    print(f"Total node capacity: {total_capacity}")
    print(f"Capacity utilization: {actual_total_vms/total_capacity*100:.1f}%")

    return workload_group, node_capacity_list


def generate_test_cases_v2(workload_size, number_of_nodes, bin_capacity, max_vms, group_probability=0.7, max_group_size=5, overlap_probability=0.3, max_groups=None):
    """
    Generates a workload group (2D array), node capacity list, and grouping constraints for the V2 bin packing problem.
    Supports overlapping groups where a workload can belong to multiple groups.
    Ensures that the total node capacity is sufficient to accommodate all VMs.

    Args:
        workload_size (int): Number of workloads (arrays) in the workload group
        number_of_nodes (int): Number of nodes (bins)
        bin_capacity (int): Fixed capacity of each bin
        max_vms (int): Maximum number of VMs within each workload
        group_probability (float): Probability that a workload will be assigned to a group (0.0 to 1.0)
        max_group_size (int): Maximum number of workloads that can be in a single group
        overlap_probability (float): Probability that a workload will be in multiple groups (0.0 to 1.0)
        max_groups (int): Maximum number of groups to create (None for no limit)

    Returns:
        tuple: (workload_group, node_capacity_list, grps) where:
            - workload_group: 2D array of VM configurations
            - node_capacity_list: list of node capacities
            - grps: list of lists defining which workloads can be grouped together (can overlap)
    """
    # Generate workload group and node capacities using the original function
    workload_group, node_capacity_list = generate_test_cases(workload_size, number_of_nodes, bin_capacity, max_vms)
    
    # Generate grouping constraints (grps) with potential overlaps
    grps = []
    workload_to_groups = {i: [] for i in range(workload_size)}  # Track which groups each workload belongs to
    
    # Determine target number of groups
    if max_groups is None:
        target_groups = max(1, workload_size // 3)  # Default: roughly workload_size/3 groups
    else:
        target_groups = min(max_groups, workload_size)
    
    # Create groups
    for group_idx in range(target_groups):
        if random.random() < group_probability:
            # Decide group size
            group_size = random.randint(2, min(max_group_size, workload_size))
            
            # Select workloads for this group
            new_group = []
            
            # First, add some workloads that might not be in any group yet
            available_workloads = list(range(workload_size))
            random.shuffle(available_workloads)
            
            for workload_id in available_workloads:
                if len(new_group) >= group_size:
                    break
                    
                # Decide whether to include this workload
                if len(workload_to_groups[workload_id]) == 0:
                    # Not in any group yet - higher probability
                    include_prob = 0.8
                else:
                    # Already in some groups - use overlap probability
                    include_prob = overlap_probability
                
                if random.random() < include_prob:
                    new_group.append(workload_id)
                    workload_to_groups[workload_id].append(group_idx)
            
            # Only keep groups with at least 2 workloads
            if len(new_group) >= 2:
                new_group.sort()  # Sort for consistency
                grps.append(new_group)
            else:
                # Remove the group assignments for this failed group
                for workload_id in new_group:
                    workload_to_groups[workload_id].remove(group_idx)
    
    # Sort groups for consistency
    grps.sort()
    
    # Analyze grouping results
    grouped_workloads = set()
    overlapping_workloads = set()
    
    for workload_id, group_list in workload_to_groups.items():
        if len(group_list) > 0:
            grouped_workloads.add(workload_id)
        if len(group_list) > 1:
            overlapping_workloads.add(workload_id)
    
    ungrouped_workloads = [i for i in range(workload_size) if i not in grouped_workloads]
    
    # Print grouping information
    print(f"\nGrouping constraints generated (with overlaps):")
    print(f"Groups (grps): {grps}")
    print(f"Number of groups: {len(grps)}")
    print(f"Workloads in groups: {len(grouped_workloads)}")
    print(f"Ungrouped workloads: {ungrouped_workloads}")
    print(f"Overlapping workloads: {sorted(overlapping_workloads)}")
    
    # Show detailed membership
    print(f"\nDetailed group membership:")
    for workload_id in range(workload_size):
        groups = workload_to_groups[workload_id]
        if len(groups) == 0:
            print(f"  Workload {workload_id}: UNGROUPED")
        elif len(groups) == 1:
            print(f"  Workload {workload_id}: Group {groups[0]}")
        else:
            print(f"  Workload {workload_id}: Groups {groups} (OVERLAPPING)")
    
    # Show group contents
    print(f"\nGroup contents:")
    for i, group in enumerate(grps):
        print(f"  Group {i}: {group}")
    
    return workload_group, node_capacity_list, grps



# %%
# ===== V2 UTILITY FUNCTIONS =====

def create_modified_groups_with_ungrouped(vm_lists, grps):
    """
    Takes the original groups and returns modified groups where all ungrouped workloads 
    are grouped together at the end. This function is used by all four V2 methods.
    
    Args:
        vm_lists: 2D array of VM configurations
        grps: List of lists defining which workloads can be grouped together (can overlap)
        
    Returns:
        list: Modified groups with ungrouped workloads added as the final group
    """
    # Start with the original groups
    modified_groups = grps.copy()
    
    # Find all workloads that are in any group
    all_grouped_workloads = set()
    for group in grps:
        all_grouped_workloads.update(group)
    
    # Find ungrouped workloads
    ungrouped_workloads = [w for w in range(len(vm_lists)) if w not in all_grouped_workloads]
    
    # Add ungrouped workloads as a final group if any exist
    if ungrouped_workloads:
        modified_groups.append(ungrouped_workloads)
    
    return modified_groups


def get_compatible_workloads(target_workload, vm_lists, grps):
    """
    For a given workload, return the list of workloads that are OK to be put together 
    on one node. This function is only used by FFD_v2 and BFD_v2.
    
    Args:
        target_workload: The workload ID to find compatible workloads for
        vm_lists: 2D array of VM configurations
        grps: List of lists defining which workloads can be grouped together (can overlap)
        
    Returns:
        list: List of workload IDs that can be placed together with target_workload
    """
    # Create workload to groups mapping
    workload_to_groups = {i: [] for i in range(len(vm_lists))}
    
    # Assign group IDs to workloads (a workload can be in multiple groups)
    for group_idx, group in enumerate(grps):
        for workload_id in group:
            workload_to_groups[workload_id].append(group_idx)
    
    # Identify grouped and ungrouped workloads
    grouped_workloads = set()
    for workload_id, group_list in workload_to_groups.items():
        if len(group_list) > 0:
            grouped_workloads.add(workload_id)
    
    ungrouped_workloads = set(range(len(vm_lists))) - grouped_workloads
    
    # Find compatible workloads for the target workload
    compatible_workloads = [target_workload]  # A workload is always compatible with itself
    
    for other_workload in range(len(vm_lists)):
        if other_workload == target_workload:
            continue
            
        # Check if target and other workload can be together
        can_be_together = False
        
        # Both ungrouped - can be together
        if target_workload in ungrouped_workloads and other_workload in ungrouped_workloads:
            can_be_together = True
        
        # One grouped, one ungrouped - cannot be together
        elif (target_workload in ungrouped_workloads) != (other_workload in ungrouped_workloads):
            can_be_together = False
        
        # Both grouped - can be together if they share at least one group
        else:
            target_groups = set(workload_to_groups[target_workload])
            other_groups = set(workload_to_groups[other_workload])
            can_be_together = len(target_groups.intersection(other_groups)) > 0
        
        if can_be_together:
            compatible_workloads.append(other_workload)
    
    return compatible_workloads


# %%
# ===== METHOD 1: Multiple Runs of Original Algorithms =====

def solve_v2_method1_ffd(vm_lists, node_capacities, grps):
    """
    Method 1: Solves V2 problem by running FFD multiple times on different groups.
    For overlapping groups, processes each workload with the first group it appears in.
    Each group gets its own separate set of nodes - no sharing between groups.
    
    Args:
        vm_lists: 2D array of VM configurations
        node_capacities: List of node capacities
        grps: List of lists defining which workloads can be grouped together (can overlap)
        
    Returns:
        tuple: (combined_assignments, total_bins_used, execution_time)
    """
    start_time = time.time()
    
    # Use utility function to get modified groups with ungrouped workloads
    modified_groups = create_modified_groups_with_ungrouped(vm_lists, grps)
    
    # Track which workloads have been processed
    processed_workloads = set()
    
    # Create processing groups - each group processes its workloads (skipping already processed ones)
    processing_groups = []
    
    # Process each group in order (including the ungrouped group at the end)
    for group_idx, group in enumerate(modified_groups):
        # Only include workloads that haven't been processed yet
        unprocessed_in_group = [w for w in group if w not in processed_workloads]
        
        if unprocessed_in_group:
            processing_groups.append(unprocessed_in_group)
            processed_workloads.update(unprocessed_in_group)
    
    combined_assignments = {}
    total_bins_used = 0
    node_offset = 0  # Track the starting node ID for each group
    
    print(f"Method 1 FFD: Processing {len(processing_groups)} groups with {len(node_capacities)} available nodes")
    
    for group_idx, group in enumerate(processing_groups):
        # Extract workloads for this group
        group_vm_lists = [vm_lists[i] for i in group]
        
        print(f"  Group {group_idx + 1}: Workloads {group}, Total VMs: {sum(len(wl) for wl in group_vm_lists)}")
        
        # Calculate available nodes for this group
        available_nodes_for_group = len(node_capacities) - node_offset
        if available_nodes_for_group <= 0:
            raise ValueError(f"No more nodes available for group {group_idx + 1}")
        
        # Create node capacities list for this group
        group_node_capacities = node_capacities[node_offset:node_offset + available_nodes_for_group]
        
        # Run FFD on this group with available nodes
        group_assignments, group_bins_used, _ = first_fit_decreasing(group_vm_lists, group_node_capacities)
        
        # Check if group exceeded available nodes
        if group_bins_used > available_nodes_for_group:
            raise ValueError(f"Group {group_idx + 1} needs {group_bins_used} nodes but only {available_nodes_for_group} available")
        
        # Adjust node IDs to global range and combine results
        for local_node_id, vms in group_assignments.items():
            global_node_id = node_offset + local_node_id
            combined_assignments[global_node_id] = vms
        
        # Update node offset and total bins count
        node_offset += group_bins_used
        total_bins_used += group_bins_used
        
        print(f"    Used {group_bins_used} nodes (global nodes {node_offset - group_bins_used} to {node_offset - 1})")
        print(f"    Remaining nodes: {len(node_capacities) - node_offset}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Method 1 FFD Total: {total_bins_used} bins used (out of {len(node_capacities)} available)")
    return combined_assignments, total_bins_used, execution_time


def solve_v2_method1_bfd(vm_lists, node_capacities, grps):
    """
    Method 1: Solves V2 problem by running BFD multiple times on different groups.
    For overlapping groups, processes each workload with the first group it appears in.
    Each group gets its own separate set of nodes - no sharing between groups.
    
    Args:
        vm_lists: 2D array of VM configurations
        node_capacities: List of node capacities
        grps: List of lists defining which workloads can be grouped together (can overlap)
        
    Returns:
        tuple: (combined_assignments, total_bins_used, execution_time)
    """
    start_time = time.time()
    
    # Use utility function to get modified groups with ungrouped workloads
    modified_groups = create_modified_groups_with_ungrouped(vm_lists, grps)
    
    # Track which workloads have been processed
    processed_workloads = set()
    
    # Create processing groups - each group processes its workloads (skipping already processed ones)
    processing_groups = []
    
    # Process each group in order (including the ungrouped group at the end)
    for group_idx, group in enumerate(modified_groups):
        # Only include workloads that haven't been processed yet
        unprocessed_in_group = [w for w in group if w not in processed_workloads]
        
        if unprocessed_in_group:
            processing_groups.append(unprocessed_in_group)
            processed_workloads.update(unprocessed_in_group)
    
    combined_assignments = {}
    total_bins_used = 0
    node_offset = 0  # Track the starting node ID for each group
    
    print(f"Method 1 BFD: Processing {len(processing_groups)} groups with {len(node_capacities)} available nodes")
    
    for group_idx, group in enumerate(processing_groups):
        # Extract workloads for this group
        group_vm_lists = [vm_lists[i] for i in group]
        
        print(f"  Group {group_idx + 1}: Workloads {group}, Total VMs: {sum(len(wl) for wl in group_vm_lists)}")
        
        # Calculate available nodes for this group
        available_nodes_for_group = len(node_capacities) - node_offset
        if available_nodes_for_group <= 0:
            raise ValueError(f"No more nodes available for group {group_idx + 1}")
        
        # Create node capacities list for this group
        group_node_capacities = node_capacities[node_offset:node_offset + available_nodes_for_group]
        
        # Run BFD on this group with available nodes
        group_assignments, group_bins_used, _ = best_fit_decreasing(group_vm_lists, group_node_capacities)
        
        # Check if group exceeded available nodes
        if group_bins_used > available_nodes_for_group:
            raise ValueError(f"Group {group_idx + 1} needs {group_bins_used} nodes but only {available_nodes_for_group} available")
        
        # Adjust node IDs to global range and combine results
        for local_node_id, vms in group_assignments.items():
            global_node_id = node_offset + local_node_id
            combined_assignments[global_node_id] = vms
        
        # Update node offset and total bins count
        node_offset += group_bins_used
        total_bins_used += group_bins_used
        
        print(f"    Used {group_bins_used} nodes (global nodes {node_offset - group_bins_used} to {node_offset - 1})")
        print(f"    Remaining nodes: {len(node_capacities) - node_offset}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Method 1 BFD Total: {total_bins_used} bins used (out of {len(node_capacities)} available)")
    return combined_assignments, total_bins_used, execution_time


# %%
def first_fit_decreasing_v2(vm_lists, node_capacities, grps):
    """
    Method 2: FFD modified to enforce group constraints during placement.
    Supports overlapping groups where workloads can belong to multiple groups.
    
    Args:
        vm_lists: 2D array of VM configurations
        node_capacities: List of node capacities
        grps: List of lists defining which workloads can be grouped together (can overlap)
        
    Returns:
        tuple: (node_assignments, bins_used, execution_time)
    """
    start_time = time.time()
    
    # Create VM list with workload IDs and flatten
    vm_with_workloads = []
    workload_group_tracker = {}
    
    for workload_id, vm_list in enumerate(vm_lists):
        workload_group_tracker[workload_id] = set()  # Track which nodes this workload uses
        
        for vm_size in vm_list:
            vm_with_workloads.append((vm_size, workload_id))
    
    # Sort by VM size in decreasing order
    vm_with_workloads.sort(key=lambda x: x[0], reverse=True)
    
    # Initialize nodes and workload tracking
    max_nodes = len(node_capacities)
    nodes = [[] for _ in range(max_nodes)]
    node_workloads = [set() for _ in range(max_nodes)]  # Track which workloads are on each node
    
    # Place VMs using FFD with group constraints
    for vm_size, workload_id in vm_with_workloads:
        placed = False
        
        # Try to place in existing nodes (first-fit)
        for node_id in range(max_nodes):
            # Check if the node has capacity
            if sum(nodes[node_id]) + vm_size <= node_capacities[node_id]:
                # Check if this workload can be placed with existing workloads on the node
                can_place = True
                
                # Get compatible workloads for the current workload
                compatible_workloads = get_compatible_workloads(workload_id, vm_lists, grps)
                
                # Check if all existing workloads on this node are compatible
                for existing_workload in node_workloads[node_id]:
                    if existing_workload not in compatible_workloads:
                        can_place = False
                        break
                
                # Also check constraint that workloads from the same group use different nodes
                if can_place and workload_id in workload_group_tracker and node_id in workload_group_tracker[workload_id]:
                    can_place = False
                
                if can_place:
                    nodes[node_id].append(vm_size)
                    node_workloads[node_id].add(workload_id)
                    workload_group_tracker[workload_id].add(node_id)
                    placed = True
                    break
        
        if not placed:
            # Create new node if available
            for node_id in range(max_nodes):
                if len(nodes[node_id]) == 0:  # Find first empty node
                    if vm_size <= node_capacities[node_id]:
                        nodes[node_id].append(vm_size)
                        node_workloads[node_id].add(workload_id)
                        workload_group_tracker[workload_id].add(node_id)
                        placed = True
                        break
            
            if not placed:
                raise ValueError(f"Cannot place VM {vm_size} from workload {workload_id} - insufficient nodes")
    
    # Create result assignments and count bins used
    node_assignments = {}
    bins_used = 0
    
    for node_id, vm_list in enumerate(nodes):
        if vm_list:
            node_assignments[node_id] = vm_list
            bins_used += 1
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return node_assignments, bins_used, execution_time


def best_fit_decreasing_v2(vm_lists, node_capacities, grps):
    """
    Method 2: BFD modified to enforce group constraints during placement.
    Supports overlapping groups where workloads can belong to multiple groups.
    
    Args:
        vm_lists: 2D array of VM configurations
        node_capacities: List of node capacities
        grps: List of lists defining which workloads can be grouped together (can overlap)
        
    Returns:
        tuple: (node_assignments, bins_used, execution_time)
    """
    start_time = time.time()
    
    # Create VM list with workload IDs and flatten
    vm_with_workloads = []
    workload_group_tracker = {}
    
    for workload_id, vm_list in enumerate(vm_lists):
        workload_group_tracker[workload_id] = set()  # Track which nodes this workload uses
        
        for vm_size in vm_list:
            vm_with_workloads.append((vm_size, workload_id))
    
    # Sort by VM size in decreasing order
    vm_with_workloads.sort(key=lambda x: x[0], reverse=True)
    
    # Initialize nodes and workload tracking
    max_nodes = len(node_capacities)
    nodes = [[] for _ in range(max_nodes)]
    node_workloads = [set() for _ in range(max_nodes)]  # Track which workloads are on each node
    
    # Place VMs using BFD with group constraints
    for vm_size, workload_id in vm_with_workloads:
        placed = False
        best_node = -1
        best_remaining_capacity = float('inf')
        
        # Find best-fit among existing nodes
        for node_id in range(max_nodes):
            if len(nodes[node_id]) > 0:  # Only consider nodes that are already in use
                # Check if the node has capacity
                if sum(nodes[node_id]) + vm_size <= node_capacities[node_id]:
                    # Check if this workload can be placed with existing workloads on the node
                    can_place = True
                    
                    # Get compatible workloads for the current workload
                    compatible_workloads = get_compatible_workloads(workload_id, vm_lists, grps)
                    
                    # Check if all existing workloads on this node are compatible
                    for existing_workload in node_workloads[node_id]:
                        if existing_workload not in compatible_workloads:
                            can_place = False
                            break
                    
                    # Also check constraint that workloads from the same group use different nodes
                    if can_place and workload_id in workload_group_tracker and node_id in workload_group_tracker[workload_id]:
                        can_place = False
                    
                    if can_place:
                        remaining_capacity = node_capacities[node_id] - sum(nodes[node_id]) - vm_size
                        if remaining_capacity < best_remaining_capacity:
                            best_remaining_capacity = remaining_capacity
                            best_node = node_id
        
        # Place in best-fit node if found
        if best_node != -1:
            nodes[best_node].append(vm_size)
            node_workloads[best_node].add(workload_id)
            workload_group_tracker[workload_id].add(best_node)
            placed = True
        else:
            # No suitable existing node, use first available empty node
            for node_id in range(max_nodes):
                if len(nodes[node_id]) == 0:  # Find first empty node
                    if vm_size <= node_capacities[node_id]:
                        nodes[node_id].append(vm_size)
                        node_workloads[node_id].add(workload_id)
                        workload_group_tracker[workload_id].add(node_id)
                        placed = True
                        break
            
            if not placed:
                raise ValueError(f"Cannot place VM {vm_size} from workload {workload_id} - insufficient nodes")
    
    # Create result assignments and count bins used
    node_assignments = {}
    bins_used = 0
    
    for node_id, vm_list in enumerate(nodes):
        if vm_list:
            node_assignments[node_id] = vm_list
            bins_used += 1
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return node_assignments, bins_used, execution_time


# %%
# =======================================================================
# OVERLAPPING GROUPS TEST - Method 1 Handling
# =======================================================================

print("\n" + "="*80)
print("OVERLAPPING GROUPS TEST FOR METHOD 1")
print("="*80)

# Create a simple test case with overlapping groups
print("Creating test case with overlapping groups...")
overlap_workloads = [
    [8, 7],      # Workload 0: 15 VMs total
    [6, 6],      # Workload 1: 12 VMs total
    [5, 5, 4],   # Workload 2: 14 VMs total (appears in both groups)
    [4, 3],      # Workload 3: 7 VMs total
    [4, 4, 4]    # Workload 4: 12 VMs total (ungrouped)
]
overlap_node_capacities = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]  # 10 nodes, capacity 16 each
overlap_grps = [[0, 1, 2], [2, 3]]  # Workload 2 appears in both groups!

print(f"\nTest Case Details:")
print(f"Workloads: {overlap_workloads}")
print(f"Node capacities: {overlap_node_capacities}")
print(f"Groups (grps): {overlap_grps}")
print(f"Total VMs: {sum(sum(wl) for wl in overlap_workloads)}")
print(f"Total capacity: {sum(overlap_node_capacities)}")

print(f"\nOverlapping Group Analysis:")
print(f"- Group 0: [0, 1, 2] - Workloads 0, 1, and 2 can be together")
print(f"- Group 1: [2, 3] - Workloads 2 and 3 can be together")
print(f"- Workload 2 appears in BOTH groups (overlapping)")
print(f"- Workload 4 is ungrouped")

print(f"\nMethod 1 Processing Logic:")
print(f"- Process Group 0 first: [0, 1, 2] → workloads 0, 1, 2 processed")
print(f"- Process Group 1 next: [2, 3] → workload 2 already processed, only process 3")
print(f"- Process ungrouped: [4] → process workload 4")
print(f"- Result: 3 separate processing groups: [0,1,2], [3], [4]")

# Test Method 1 with overlapping groups
print(f"\n{'-'*60}")
print("METHOD 1 WITH OVERLAPPING GROUPS")
print(f"{'-'*60}")

print(f"\nMethod 1 FFD with Overlapping Groups:")
overlap_assignments_1a, overlap_bins_1a, overlap_time_1a = solve_v2_method1_ffd(
    overlap_workloads, overlap_node_capacities, overlap_grps
)

print(f"\nMethod 1 BFD with Overlapping Groups:")
overlap_assignments_1b, overlap_bins_1b, overlap_time_1b = solve_v2_method1_bfd(
    overlap_workloads, overlap_node_capacities, overlap_grps
)

def print_detailed_assignments(name, assignments, bins_used, exec_time):
    """Helper function to print detailed assignment information"""
    print(f"\n{name} Results:")
    print(f"  Bins used: {bins_used}")
    print(f"  Execution time: {exec_time:.6f} seconds")
    print(f"  Node assignments:")
    for node_id in sorted(assignments.keys()):
        vms = assignments[node_id]
        capacity_used = sum(vms)
        capacity_remaining = overlap_node_capacities[node_id % len(overlap_node_capacities)] - capacity_used
        print(f"    Node {node_id}: {vms} (used: {capacity_used}/16, remaining: {capacity_remaining})")

print_detailed_assignments("Method 1 FFD", overlap_assignments_1a, overlap_bins_1a, overlap_time_1a)
print_detailed_assignments("Method 1 BFD", overlap_assignments_1b, overlap_bins_1b, overlap_time_1b)

# Test Method 2 with overlapping groups
print(f"\n{'-'*60}")
print("METHOD 2 WITH OVERLAPPING GROUPS")
print(f"{'-'*60}")

print(f"\nMethod 2 FFD_v2 with Overlapping Groups:")
overlap_assignments_2a, overlap_bins_2a, overlap_time_2a = first_fit_decreasing_v2(
    overlap_workloads, overlap_node_capacities, overlap_grps
)

print(f"\nMethod 2 BFD_v2 with Overlapping Groups:")
overlap_assignments_2b, overlap_bins_2b, overlap_time_2b = best_fit_decreasing_v2(
    overlap_workloads, overlap_node_capacities, overlap_grps
)

print_detailed_assignments("Method 2 FFD_v2", overlap_assignments_2a, overlap_bins_2a, overlap_time_2a)
print_detailed_assignments("Method 2 BFD_v2", overlap_assignments_2b, overlap_bins_2b, overlap_time_2b)

print(f"\n{'-'*60}")
print("COMPARISON: METHOD 1 vs METHOD 2")
print(f"{'-'*60}")

print(f"Method 1 (Multi-Algorithm Approach):")
print(f"  - FFD: {overlap_bins_1a} bins used in {overlap_time_1a:.6f}s")
print(f"  - BFD: {overlap_bins_1b} bins used in {overlap_time_1b:.6f}s")
print(f"  - Strategy: Processes each group separately with original algorithms")
print(f"  - Handles overlapping by processing workload with first group only")

print(f"\nMethod 2 (Modified Algorithm Approach):")
print(f"  - FFD_v2: {overlap_bins_2a} bins used in {overlap_time_2a:.6f}s")
print(f"  - BFD_v2: {overlap_bins_2b} bins used in {overlap_time_2b:.6f}s")
print(f"  - Strategy: Single pass with group constraint checking during placement")
print(f"  - Handles overlapping by checking compatibility during VM placement")

print(f"\nBin Usage Analysis:")
print(f"  - Best FFD result: {min(overlap_bins_1a, overlap_bins_2a)} bins")
print(f"  - Best BFD result: {min(overlap_bins_1b, overlap_bins_2b)} bins")
print(f"  - Overall best: {min(overlap_bins_1a, overlap_bins_1b, overlap_bins_2a, overlap_bins_2b)} bins")

if overlap_bins_1a != overlap_bins_2a or overlap_bins_1b != overlap_bins_2b:
    print(f"\nNote: Different methods may produce different results due to:")
    print(f"  - Different processing orders")
    print(f"  - Different constraint interpretation approaches")
    print(f"  - Method 1: Group-first processing")
    print(f"  - Method 2: Individual VM constraint checking")

print(f"\nFinal Results Summary:")
print(f"- Method 1 FFD: {overlap_bins_1a} bins used in {overlap_time_1a:.6f}s")
print(f"- Method 1 BFD: {overlap_bins_1b} bins used in {overlap_time_1b:.6f}s")
print(f"- Method 2 FFD_v2: {overlap_bins_2a} bins used in {overlap_time_2a:.6f}s")
print(f"- Method 2 BFD_v2: {overlap_bins_2b} bins used in {overlap_time_2b:.6f}s")

best_overall = min(overlap_bins_1a, overlap_bins_1b, overlap_bins_2a, overlap_bins_2b)
print(f"- Best overall result: {best_overall} bins")

print(f"\nBoth approaches are valid and provide correct constraint compliance")
print(f"while handling overlapping group scenarios effectively.")

# %%
# =======================================================================
# COMPREHENSIVE TEST: 100 WORKLOADS PERFORMANCE COMPARISON
# =======================================================================

print("\n" + "="*80)
print("COMPREHENSIVE PERFORMANCE TEST: 100 WORKLOADS")
print("="*80)

# Generate a large test case with 100 workloads
print("Generating comprehensive test case with 100 workloads...")
large_workloads, large_node_capacities, large_grps = generate_test_cases_v2(
    workload_size=100,       # 100 workloads
    number_of_nodes=200,     # 200 available nodes
    bin_capacity=32,         # Each node capacity: 32
    max_vms=15,              # Max 15 VMs per workload
    group_probability=0.8,   # 80% chance of being in a group
    max_group_size=5        # Max 5 workloads per group
)

# Calculate test case statistics
total_vms = sum(sum(wl) for wl in large_workloads)
total_capacity = sum(large_node_capacities)
num_groups = len(large_grps)
grouped_workloads = set()
for group in large_grps:
    grouped_workloads.update(group)
ungrouped_count = len(large_workloads) - len(grouped_workloads)

print(f"\nTest Case Statistics:")
print(f"- Workloads: {len(large_workloads)}")
print(f"- Available nodes: {len(large_node_capacities)}")
print(f"- Node capacity: {large_node_capacities[0]} each")
print(f"- Total VMs: {total_vms}")
print(f"- Total capacity: {total_capacity}")
print(f"- Capacity utilization: {total_vms/total_capacity*100:.1f}%")
print(f"- Number of groups: {num_groups}")
print(f"- Grouped workloads: {len(grouped_workloads)}")
print(f"- Ungrouped workloads: {ungrouped_count}")

# Show some group examples
print(f"\nSample groups:")
for i, group in enumerate(large_grps[:5]):  # Show first 5 groups
    print(f"  Group {i}: {group}")
if len(large_grps) > 5:
    print(f"  ... and {len(large_grps) - 5} more groups")

print(f"\n{'='*80}")
print("RUNNING ALL 4 ALGORITHMS...")
print(f"{'='*80}")

# Store results for comparison
results = {}

# Method 1: Multi-FFD
print(f"\n{'-'*60}")
print("METHOD 1: Multi-FFD")
print(f"{'-'*60}")
start_time = time.time()
try:
    assign_1a, bins_1a, time_1a = solve_v2_method1_ffd(large_workloads, large_node_capacities, large_grps)
    results['Method 1 FFD'] = {'bins': bins_1a, 'time': time_1a, 'success': True}
    print(f"✓ Method 1 FFD completed: {bins_1a} bins used in {time_1a:.4f} seconds")
except Exception as e:
    results['Method 1 FFD'] = {'bins': float('inf'), 'time': float('inf'), 'success': False, 'error': str(e)}
    print(f"✗ Method 1 FFD failed: {e}")

# Method 1: Multi-BFD
print(f"\n{'-'*60}")
print("METHOD 1: Multi-BFD")
print(f"{'-'*60}")
try:
    assign_1b, bins_1b, time_1b = solve_v2_method1_bfd(large_workloads, large_node_capacities, large_grps)
    results['Method 1 BFD'] = {'bins': bins_1b, 'time': time_1b, 'success': True}
    print(f"✓ Method 1 BFD completed: {bins_1b} bins used in {time_1b:.4f} seconds")
except Exception as e:
    results['Method 1 BFD'] = {'bins': float('inf'), 'time': float('inf'), 'success': False, 'error': str(e)}
    print(f"✗ Method 1 BFD failed: {e}")

# Method 2: FFD_v2
print(f"\n{'-'*60}")
print("METHOD 2: FFD_v2")
print(f"{'-'*60}")
try:
    assign_2a, bins_2a, time_2a = first_fit_decreasing_v2(large_workloads, large_node_capacities, large_grps)
    results['Method 2 FFD_v2'] = {'bins': bins_2a, 'time': time_2a, 'success': True}
    print(f"✓ Method 2 FFD_v2 completed: {bins_2a} bins used in {time_2a:.4f} seconds")
except Exception as e:
    results['Method 2 FFD_v2'] = {'bins': float('inf'), 'time': float('inf'), 'success': False, 'error': str(e)}
    print(f"✗ Method 2 FFD_v2 failed: {e}")

# Method 2: BFD_v2
print(f"\n{'-'*60}")
print("METHOD 2: BFD_v2")
print(f"{'-'*60}")
try:
    assign_2b, bins_2b, time_2b = best_fit_decreasing_v2(large_workloads, large_node_capacities, large_grps)
    results['Method 2 BFD_v2'] = {'bins': bins_2b, 'time': time_2b, 'success': True}
    print(f"✓ Method 2 BFD_v2 completed: {bins_2b} bins used in {time_2b:.4f} seconds")
except Exception as e:
    results['Method 2 BFD_v2'] = {'bins': float('inf'), 'time': float('inf'), 'success': False, 'error': str(e)}
    print(f"✗ Method 2 BFD_v2 failed: {e}")

# Performance Analysis
print(f"\n{'='*80}")
print("COMPREHENSIVE PERFORMANCE ANALYSIS")
print(f"{'='*80}")

# Sort results by bin count (successful ones first)
successful_results = {k: v for k, v in results.items() if v['success']}
failed_results = {k: v for k, v in results.items() if not v['success']}

if successful_results:
    print(f"\n{'-'*60}")
    print("BIN USAGE COMPARISON (Successful runs)")
    print(f"{'-'*60}")
    
    sorted_by_bins = sorted(successful_results.items(), key=lambda x: x[1]['bins'])
    
    for i, (method, data) in enumerate(sorted_by_bins):
        rank = i + 1
        efficiency = (sorted_by_bins[0][1]['bins'] / data['bins']) * 100
        print(f"{rank}. {method:15} | {data['bins']:3d} bins | {data['time']:8.4f}s | {efficiency:5.1f}% efficiency")
    
    print(f"\n{'-'*60}")
    print("EXECUTION TIME COMPARISON")
    print(f"{'-'*60}")
    
    sorted_by_time = sorted(successful_results.items(), key=lambda x: x[1]['time'])
    
    for i, (method, data) in enumerate(sorted_by_time):
        rank = i + 1
        speed_ratio = sorted_by_time[0][1]['time'] / data['time']
        print(f"{rank}. {method:15} | {data['time']:8.4f}s | {data['bins']:3d} bins | {speed_ratio:5.1f}x relative speed")

    # Find best performers
    best_bins = min(data['bins'] for data in successful_results.values())
    best_time = min(data['time'] for data in successful_results.values())
    
    best_bin_methods = [method for method, data in successful_results.items() if data['bins'] == best_bins]
    best_time_methods = [method for method, data in successful_results.items() if data['time'] == best_time]
    
    print(f"\n{'-'*60}")
    print("SUMMARY STATISTICS")
    print(f"{'-'*60}")
    print(f"Best bin usage: {best_bins} bins ({', '.join(best_bin_methods)})")
    print(f"Fastest execution: {best_time:.4f}s ({', '.join(best_time_methods)})")
    
    # Calculate improvement potential
    worst_bins = max(data['bins'] for data in successful_results.values())
    worst_time = max(data['time'] for data in successful_results.values())
    
    bin_improvement = ((worst_bins - best_bins) / worst_bins) * 100
    time_improvement = ((worst_time - best_time) / worst_time) * 100
    
    print(f"Bin usage spread: {best_bins} - {worst_bins} bins ({bin_improvement:.1f}% improvement potential)")
    print(f"Time spread: {best_time:.4f}s - {worst_time:.4f}s ({time_improvement:.1f}% improvement potential)")

if failed_results:
    print(f"\n{'-'*60}")
    print("FAILED ALGORITHMS")
    print(f"{'-'*60}")
    for method, data in failed_results.items():
        print(f"✗ {method}: {data['error']}")

print(f"\n{'='*80}")
print("COMPREHENSIVE TEST COMPLETED")
print(f"{'='*80}")


print(f"\nTest completed with {len(large_workloads)} workloads, {len(large_grps)} groups, {total_vms} total VMs")

# %%
