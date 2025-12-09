"""
Greedy Node Association (GoA) Algorithm
=======================================
Implementation of Algorithm 1 from the ShapeFL paper.

Purpose: Given a set of edge aggregators, determine which edge aggregator
         each distributed node should associate with.

Objective: Minimize the combined cost function:
    ΔJ_ne = κ_e * c_ne - γ * (1/|ε|) * ΔS_ne

Where:
- κ_e: number of edge epochs per cloud aggregation
- c_ne: communication cost between node n and edge aggregator e
- γ: trade-off weight between communication cost and data distribution
- ΔS_ne: average reduction of data distribution diversity when associating n with e

Reference: Paper Section IV-C, Algorithm 1 (page 2606)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NodeAssociationResult:
    """Result of the GoA algorithm."""

    # Mapping from node_id to associated edge_aggregator_id
    associations: Dict[int, int]
    # Set of nodes associated with each edge aggregator
    edge_nodes: Dict[int, Set[int]]
    # Total data size at each edge aggregator
    edge_data_sizes: Dict[int, int]


class GreedyNodeAssociation:
    """
    Greedy Node Association (GoA) Algorithm.

    Implements Algorithm 1 from the ShapeFL paper to associate each
    distributed computing node with an optimal edge aggregator.
    """

    def __init__(
        self,
        edge_aggregators: List[int],  # Set ε of selected edge aggregators
        communication_costs: Dict[
            Tuple[int, int], float
        ],  # c_ne for each (node, edge) pair
        similarity_matrix: np.ndarray,  # S_ij: data distribution similarity
        data_sizes: Dict[int, int],  # D_n: data size of each node
        kappa_e: int = 2,  # Edge epochs per cloud aggregation
        gamma: float = 2800.0,  # Trade-off weight
        B_e: int = 10,  # Max nodes per edge aggregator
    ):
        """
        Initialize the GoA algorithm.

        Args:
            edge_aggregators: List of node IDs selected as edge aggregators
            communication_costs: Dictionary mapping (node_id, edge_id) to cost
            similarity_matrix: N x N matrix where S[i,j] is similarity between nodes i and j
            data_sizes: Dictionary mapping node_id to data size
            kappa_e: Number of local epochs before edge aggregation
            gamma: Trade-off weight between communication cost and diversity
            B_e: Maximum number of nodes that can associate with each edge aggregator
        """
        self.edge_aggregators = set(edge_aggregators)
        self.c_ne = communication_costs
        self.S = similarity_matrix
        self.D = data_sizes
        self.kappa_e = kappa_e
        self.gamma = gamma
        self.B_e = B_e

        # All nodes (including those that are edge aggregators)
        self.all_nodes = set(data_sizes.keys())

        # Nodes that need to be associated (excluding edge aggregators themselves)
        # In the paper, edge aggregators can also be computing nodes
        self.nodes_to_assign = self.all_nodes - self.edge_aggregators

    def compute_delta_J(
        self,
        node: int,
        edge: int,
        current_edge_nodes: Set[int],
        current_edge_data_size: int,
    ) -> float:
        """
        Compute ΔJ_ne: the cost-benefit of associating node n with edge e.

        From Algorithm 1, line 9:
        ΔJ_ne = κ_e * c_ne - γ * (1/|ε|) * ΔS_ne

        Where ΔS_ne is the change in diversity measure.

        Args:
            node: Node ID to potentially associate
            edge: Edge aggregator ID
            current_edge_nodes: Set of nodes currently associated with edge
            current_edge_data_size: Total data size currently at edge

        Returns:
            ΔJ_ne value (lower is better)
        """
        # Communication cost term
        comm_cost = self.kappa_e * self.c_ne.get((node, edge), float("inf"))

        # Compute diversity term ΔS_ne
        # This is the average similarity between node n and existing nodes at edge e
        if len(current_edge_nodes) == 0:
            delta_S = 0.0
        else:
            # Sum of S_ij * D_i * D_j for all pairs involving node n
            D_n = self.D[node]
            total_similarity = 0.0
            for other_node in current_edge_nodes:
                D_other = self.D[other_node]
                S_ij = self.S[node, other_node]
                total_similarity += S_ij * D_n * D_other

            # Normalize by the combinatorial factor
            new_data_size = current_edge_data_size + D_n
            # Paper uses D_e choose 2 for normalization
            if new_data_size > 1:
                delta_S = total_similarity / (new_data_size * (new_data_size - 1) / 2)
            else:
                delta_S = 0.0

        # Trade-off between communication cost and diversity
        num_edges = len(self.edge_aggregators)
        delta_J = comm_cost - self.gamma * (1.0 / num_edges) * delta_S

        return delta_J

    def run(self) -> NodeAssociationResult:
        """
        Execute the GoA algorithm (Algorithm 1).

        Returns:
            NodeAssociationResult containing the node-edge associations
        """
        # Initialize (Algorithm 1, lines 1-2)
        # M_e: set of nodes associated with edge e
        M_e: Dict[int, Set[int]] = {e: set() for e in self.edge_aggregators}
        # D_e: total data size at edge e
        D_e: Dict[int, int] = {e: 0 for e in self.edge_aggregators}
        # N_a: unassigned nodes
        N_a = set(self.nodes_to_assign)

        # Association results
        associations: Dict[int, int] = {}

        # Main loop (Algorithm 1, lines 3-15)
        while len(N_a) > 0:
            best_node = None
            best_edge = None
            best_delta_J = float("inf")

            # For each unassigned node
            for n in N_a:
                # For each edge aggregator with capacity
                for e in self.edge_aggregators:
                    # Check capacity constraint (line 5)
                    if len(M_e[e]) >= self.B_e:
                        continue

                    # Compute ΔJ_ne
                    delta_J = self.compute_delta_J(n, e, M_e[e], D_e[e])

                    # Track the best (minimum) ΔJ_ne
                    if delta_J < best_delta_J:
                        best_delta_J = delta_J
                        best_node = n
                        best_edge = e

            if best_node is None:
                # No valid assignment found (all edges at capacity)
                print(
                    f"Warning: Could not assign {len(N_a)} nodes - all edges at capacity"
                )
                break

            # Associate the best node with the best edge (lines 11-14)
            M_e[best_edge].add(best_node)
            N_a.remove(best_node)
            D_e[best_edge] += self.D[best_node]
            associations[best_node] = best_edge

        # Edge aggregators are associated with themselves
        for e in self.edge_aggregators:
            associations[e] = e

        return NodeAssociationResult(
            associations=associations, edge_nodes=M_e, edge_data_sizes=D_e
        )


def run_goa(
    edge_aggregators: List[int],
    nodes: List[int],
    communication_costs: Dict[Tuple[int, int], float],
    similarity_matrix: np.ndarray,
    data_sizes: Dict[int, int],
    kappa_e: int = 2,
    gamma: float = 2800.0,
    B_e: int = 10,
) -> NodeAssociationResult:
    """
    Convenience function to run the GoA algorithm.

    Args:
        edge_aggregators: List of edge aggregator IDs
        nodes: List of all node IDs
        communication_costs: Dictionary mapping (node_id, edge_id) to cost
        similarity_matrix: N x N similarity matrix
        data_sizes: Dictionary mapping node_id to data size
        kappa_e: Edge epochs before cloud aggregation
        gamma: Trade-off weight
        B_e: Max nodes per edge aggregator

    Returns:
        NodeAssociationResult
    """
    goa = GreedyNodeAssociation(
        edge_aggregators=edge_aggregators,
        communication_costs=communication_costs,
        similarity_matrix=similarity_matrix,
        data_sizes=data_sizes,
        kappa_e=kappa_e,
        gamma=gamma,
        B_e=B_e,
    )
    return goa.run()


if __name__ == "__main__":
    # Test the GoA algorithm with sample data
    print("Testing GoA Algorithm")
    print("=" * 50)

    # Sample configuration: 5 nodes, 2 edge aggregators
    num_nodes = 5
    edge_aggs = [0, 2]  # Nodes 0 and 2 are edge aggregators

    # Random similarity matrix (symmetric)
    np.random.seed(42)
    S = np.random.rand(num_nodes, num_nodes)
    S = (S + S.T) / 2  # Make symmetric
    np.fill_diagonal(S, 0)  # Zero diagonal (node is not similar to itself)

    # Communication costs (simplified: based on "distance")
    comm_costs = {}
    for n in range(num_nodes):
        for e in edge_aggs:
            # Simulated cost based on node distance
            comm_costs[(n, e)] = abs(n - e) * 100  # Simple linear cost

    # Data sizes
    data_sizes = {i: 180 for i in range(num_nodes)}  # 180 samples per node

    # Run GoA
    result = run_goa(
        edge_aggregators=edge_aggs,
        nodes=list(range(num_nodes)),
        communication_costs=comm_costs,
        similarity_matrix=S,
        data_sizes=data_sizes,
        kappa_e=2,
        gamma=2800.0,
        B_e=10,
    )

    print("\nResults:")
    print(f"Associations: {result.associations}")
    print(f"Edge nodes: {result.edge_nodes}")
    print(f"Edge data sizes: {result.edge_data_sizes}")
