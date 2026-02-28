"""
Greedy Node Association (GoA) Algorithm
=======================================
Implementation of Algorithm 1 from the ShapeFL paper.

Purpose: Given a set of edge aggregators, determine which edge aggregator
         each distributed node should associate with.

Objective (Eq. 14):
    min_Y  kappa_c * Sum y_ne * c_ne  -  gamma/|E| * Sum_{e in E} [1/C(D_e,2)] * Sum_{i,j in M_e} S_ij * D_i * D_j

Per-step greedy criterion (Algorithm 1, line 9):
    Delta_J_ne = kappa_c * c_ne  -  gamma * (1/|E|) * Delta_S_ne

Where Delta_S_ne (line 8) is the CHANGE in the diversity measure when adding node n to edge e:
    Delta_S_ne = [Sum_{i,j in M'_e} S_ij*D_i*D_j / C(D'_e,2)] - [Sum_{i,j in M_e} S_ij*D_i*D_j / C(D_e,2)]

Reference: Paper Section IV-C, Algorithm 1
"""

import numpy as np
from typing import Dict, List, Set, Tuple
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
    # Objective value J_m from Eq. (14)
    objective_value: float
    # Running diversity sums per edge: Sum_{i,j in M_e} S_ij * D_i * D_j
    edge_diversity_sums: Dict[int, float]


class GreedyNodeAssociation:
    """
    Greedy Node Association (GoA) Algorithm.

    Implements Algorithm 1 from the ShapeFL paper to associate each
    distributed computing node with an optimal edge aggregator.
    """

    def __init__(
        self,
        edge_aggregators: List[int],  # Set E of selected edge aggregators
        communication_costs: Dict[
            Tuple[int, int], float
        ],  # c_ne for each (node, edge) pair
        similarity_matrix: np.ndarray,  # S_ij: data distribution diversity
        data_sizes: Dict[int, int],  # D_n: data size of each node
        kappa_c: int = 10,  # kappa_c: edge epochs per cloud round (Algorithm 1 line 9)
        gamma: float = 2800.0,  # Trade-off weight
        B_e: int = 10,  # Max nodes per edge aggregator
    ):
        """
        Initialize the GoA algorithm.

        Args:
            edge_aggregators: List of node IDs selected as edge aggregators
            communication_costs: Dictionary mapping (node_id, edge_id) to cost
            similarity_matrix: N x N matrix where S[i,j] is diversity between nodes i and j
            data_sizes: Dictionary mapping node_id to data size
            kappa_c: Number of edge aggregation epochs per cloud round
            gamma: Trade-off weight between communication cost and diversity
            B_e: Maximum number of nodes that can associate with each edge aggregator
        """
        self.edge_aggregators = set(edge_aggregators)
        self.c_ne = communication_costs
        self.S = similarity_matrix
        self.D = data_sizes
        self.kappa_c = kappa_c
        self.gamma = gamma
        self.B_e = B_e

        # All nodes (including those that are edge aggregators)
        self.all_nodes = set(data_sizes.keys())

    @staticmethod
    def _comb2(d: float) -> float:
        """Compute C(d, 2) = d*(d-1)/2. Used for normalization in Eq. (12)."""
        return d * (d - 1) / 2.0 if d > 1 else 0.0

    def run(self) -> NodeAssociationResult:
        """
        Execute the GoA algorithm (Algorithm 1).

        Follows the paper exactly:
          - M_e <- empty, D_e <- 0  for each edge  (line 1)
          - N_a <- N  (all nodes, *including* edge aggregators)  (line 2)
        The greedy loop will naturally self-assign each edge aggregator
        because c_ne[(e,e)] = 0 gives the smallest Delta_J on the first
        iterations.

        Returns:
            NodeAssociationResult containing node-edge associations and J_m.
        """
        # Initialize (Algorithm 1, lines 1-2) — paper: M_e <- empty, N_a <- N
        M_e: Dict[int, Set[int]] = {e: set() for e in self.edge_aggregators}
        D_e: Dict[int, int] = {e: 0 for e in self.edge_aggregators}

        # Track running sum: Sum_{i,j in M_e} S_ij * D_i * D_j for each edge
        pair_sums: Dict[int, float] = {e: 0.0 for e in self.edge_aggregators}

        # Associations (empty at start)
        associations: Dict[int, int] = {}

        # Unassigned nodes = ALL nodes including edge aggregators (paper line 2)
        N_a = set(self.all_nodes)

        num_edges = len(self.edge_aggregators)

        # Main loop (Algorithm 1, lines 3-15)
        while len(N_a) > 0:
            best_node = None
            best_edge = None
            best_delta_J = float("inf")
            best_new_pairs_sum = 0.0

            # For each unassigned node (line 4)
            for n in N_a:
                D_n = self.D[n]

                # For each edge aggregator with capacity (line 5)
                for e in self.edge_aggregators:
                    if len(M_e[e]) >= self.B_e:
                        continue

                    # --- Communication cost term: kappa_c * c_ne (line 9, first term) ---
                    comm_cost = self.kappa_c * self.c_ne.get((n, e), float("inf"))

                    # --- Compute Delta_S_ne (line 8) ---
                    # New pairs involving node n with all existing nodes in M_e
                    new_pairs_sum = 0.0
                    for m in M_e[e]:
                        new_pairs_sum += self.S[n, m] * D_n * self.D[m]

                    # Old diversity term: pair_sums[e] / C(D_e, 2)
                    old_comb = self._comb2(D_e[e])
                    old_term = pair_sums[e] / old_comb if old_comb > 0 else 0.0

                    # New diversity term: (pair_sums[e] + new_pairs) / C(D_e + D_n, 2)
                    new_sum = pair_sums[e] + new_pairs_sum
                    new_de = D_e[e] + D_n
                    new_comb = self._comb2(new_de)
                    new_term = new_sum / new_comb if new_comb > 0 else 0.0

                    # Delta_S_ne = new diversity - old diversity
                    delta_S = new_term - old_term

                    # --- Delta_J_ne (line 9) ---
                    delta_J = comm_cost - self.gamma * (1.0 / num_edges) * delta_S

                    # Track the minimum Delta_J_ne (line 10)
                    if delta_J < best_delta_J:
                        best_delta_J = delta_J
                        best_node = n
                        best_edge = e
                        best_new_pairs_sum = new_pairs_sum

            if best_node is None:
                # No valid assignment found (all edges at capacity)
                print(
                    f"Warning: Could not assign {len(N_a)} nodes - all edges at capacity"
                )
                break

            # Associate the best node with the best edge (lines 11-14)
            pair_sums[best_edge] += best_new_pairs_sum
            M_e[best_edge].add(best_node)
            D_e[best_edge] += self.D[best_node]
            N_a.remove(best_node)
            associations[best_node] = best_edge

        # --- Compute final objective J_m (Eq. 14) ---
        # J_m = kappa_c * Sum c_ne  -  gamma/|E| * Sum diversity
        comm_total = 0.0
        for node, edge in associations.items():
            if node not in self.edge_aggregators:
                comm_total += self.kappa_c * self.c_ne.get((node, edge), 0)

        diversity_total = 0.0
        for e in self.edge_aggregators:
            comb = self._comb2(D_e[e])
            if comb > 0:
                diversity_total += pair_sums[e] / comb
        if num_edges > 0:
            diversity_total /= num_edges

        J_m = comm_total - self.gamma * diversity_total

        return NodeAssociationResult(
            associations=associations,
            edge_nodes=M_e,
            edge_data_sizes=D_e,
            objective_value=J_m,
            edge_diversity_sums=pair_sums,
        )


def run_goa(
    edge_aggregators: List[int],
    nodes: List[int],
    communication_costs: Dict[Tuple[int, int], float],
    similarity_matrix: np.ndarray,
    data_sizes: Dict[int, int],
    kappa_c: int = 10,
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
        kappa_c: Edge epochs per cloud round
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
        kappa_c=kappa_c,
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
    edge_aggs = [0, 2]

    # Random diversity matrix (symmetric)
    np.random.seed(42)
    S = np.random.rand(num_nodes, num_nodes)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 0)

    # Communication costs: c_ne[(n, e)], zero for self-association
    comm_costs = {}
    for n in range(num_nodes):
        for e in edge_aggs:
            if n == e:
                comm_costs[(n, e)] = 0.0
            else:
                comm_costs[(n, e)] = abs(n - e) * 100

    # Data sizes (180 samples per node)
    data_sizes = {i: 180 for i in range(num_nodes)}

    # Run GoA
    result = run_goa(
        edge_aggregators=edge_aggs,
        nodes=list(range(num_nodes)),
        communication_costs=comm_costs,
        similarity_matrix=S,
        data_sizes=data_sizes,
        kappa_c=10,
        gamma=2800.0,
        B_e=10,
    )

    print("\nResults:")
    print(f"Associations: {result.associations}")
    print(f"Edge nodes: {result.edge_nodes}")
    print(f"Edge data sizes: {result.edge_data_sizes}")
    print(f"Objective J_m: {result.objective_value:.2f}")
