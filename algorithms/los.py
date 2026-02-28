"""
Local Search Edge Selection (LoS) Algorithm
============================================
Implementation of Algorithm 2 from the ShapeFL paper.

Purpose: Select which nodes should become edge aggregators from the
         candidate set N_e to minimize the overall objective function J.

Objective (Eq. 19):
    J(E_s) = J_m(E_s) + Sum_{e in E_s} c_ec

Where J_m is the optimal value from the GoA node association (Eq. 14), which
includes BOTH communication cost (kappa_c * Sum c_ne) AND data distribution
diversity (-gamma * J_d).

The algorithm uses three local search operations:
- open(e): Add edge aggregator e to the current solution
- close(e): Remove edge aggregator e from the current solution
- swap(e, e'): Replace edge aggregator e' with e in the current solution

Reference: Paper Section IV-C, Algorithm 2 (page 2606)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

from .goa import GreedyNodeAssociation, NodeAssociationResult


@dataclass
class EdgeSelectionResult:
    """Result of the LoS algorithm."""

    # Set of selected edge aggregator IDs
    selected_edges: Set[int]
    # Node association result from GoA
    node_associations: NodeAssociationResult
    # Final objective value J
    objective_value: float


class LocalSearchEdgeSelection:
    """
    Local Search Edge Selection (LoS) Algorithm.

    Implements Algorithm 2 from the ShapeFL paper to select
    the optimal set of edge aggregators.
    """

    def __init__(
        self,
        candidate_edges: List[int],  # N_e: candidate edge aggregator nodes
        all_nodes: List[int],  # N: all computing nodes
        communication_costs_ne: Dict[Tuple[int, int], float],  # c_ne: node to edge cost
        communication_costs_ec: Dict[int, float],  # c_ec: edge to cloud cost
        similarity_matrix: np.ndarray,  # S_ij: data distribution diversity
        data_sizes: Dict[int, int],  # D_n: data size of each node
        kappa_c: int = 10,  # Edge aggregation epochs per cloud round
        gamma: float = 2800.0,  # Trade-off weight
        B_e: int = 10,  # Max nodes per edge
        T_max: int = 30,  # Max iterations
    ):
        """
        Initialize the LoS algorithm.

        Args:
            candidate_edges: List of node IDs that can be edge aggregators
            all_nodes: List of all node IDs
            communication_costs_ne: Cost from node n to edge e
            communication_costs_ec: Cost from edge e to cloud server
            similarity_matrix: N x N similarity matrix
            data_sizes: Data size per node
            kappa_c: Edge aggregation epochs per cloud round
            gamma: Trade-off weight
            B_e: Max nodes per edge aggregator
            T_max: Maximum iterations for local search
        """
        self.N_c = set(candidate_edges)
        self.N = set(all_nodes)
        self.c_ne = communication_costs_ne
        self.c_ec = communication_costs_ec
        self.S = similarity_matrix
        self.D = data_sizes
        self.kappa_c = kappa_c
        self.gamma = gamma
        self.B_e = B_e
        self.T_max = T_max

    def compute_objective_J(
        self, edge_set: Set[int]
    ) -> Tuple[float, Optional[NodeAssociationResult]]:
        """
        Compute the objective function J(E_s) for a given edge aggregator set.

        From Equation (19) in the paper:
        J(E_s) = J_m(E_s) + Sum_{e in E_s} c_ec

        J_m is the optimal value of Eq. (14), returned by GoA, which already
        includes both the communication cost and the -gamma * diversity term.

        Note: c_ec is NOT multiplied by kappa_c. It's a per-cloud-round cost
        (edges submit to cloud once per cloud round).

        Args:
            edge_set: Set of edge aggregator IDs

        Returns:
            Tuple of (objective_value, node_association_result)
        """
        if len(edge_set) == 0:
            return float("inf"), None

        # Check feasibility: can all nodes be assigned?
        total_capacity = len(edge_set) * self.B_e
        if total_capacity < len(self.N):
            return float("inf"), None

        # Run GoA to get optimal node associations for this edge set
        goa = GreedyNodeAssociation(
            edge_aggregators=list(edge_set),
            communication_costs=self.c_ne,
            similarity_matrix=self.S,
            data_sizes=self.D,
            kappa_c=self.kappa_c,
            gamma=self.gamma,
            B_e=self.B_e,
        )
        association_result = goa.run()

        # Check if all nodes were assigned
        if len(association_result.associations) < len(self.N):
            return float("inf"), association_result

        # J_m from GoA (Eq. 14): includes comm cost AND diversity term
        J_m = association_result.objective_value

        # Edge-to-cloud communication cost: Sum c_ec (Eq. 19, no kappa_c multiplier)
        edge_cloud_cost = sum(self.c_ec.get(e, 0) for e in edge_set)

        # Total objective
        J = J_m + edge_cloud_cost

        return J, association_result

    def initialize_random(self, num_edges: int = 3) -> Set[int]:
        """
        Initialize with a random feasible solution.

        Args:
            num_edges: Number of initial edge aggregators

        Returns:
            Initial edge aggregator set
        """
        num_edges = min(num_edges, len(self.N_c))
        return set(np.random.choice(list(self.N_c), num_edges, replace=False))

    def run(self, initial_edges: Optional[Set[int]] = None) -> EdgeSelectionResult:
        """
        Execute the LoS algorithm (Algorithm 2).

        Args:
            initial_edges: Initial set of edge aggregators (random if None)

        Returns:
            EdgeSelectionResult with optimal edge selection
        """
        # Initialize (Algorithm 2, line 1)
        if initial_edges is None:
            E_s = self.initialize_random()
        else:
            E_s = set(initial_edges)

        J_current, assoc_current = self.compute_objective_J(E_s)

        print(f"Initial: {len(E_s)} edges, J = {J_current:.2f}")

        # Main loop (Algorithm 2, lines 2-19)
        for t in range(self.T_max):
            improved = False

            # 'open' operation (Algorithm 2, lines 3-7)
            E_not_selected = self.N_c - E_s
            for e in E_not_selected:
                E_new = E_s | {e}
                J_new, assoc_new = self.compute_objective_J(E_new)

                if J_new < J_current:
                    E_s = E_new
                    J_current = J_new
                    assoc_current = assoc_new
                    improved = True
                    print(f"  [open] Added edge {e}, J = {J_current:.2f}")
                    break

            if improved:
                continue

            # 'close' operation (Algorithm 2, lines 8-12)
            if len(E_s) > 1:  # Keep at least one edge
                for e in list(E_s):
                    E_new = E_s - {e}
                    J_new, assoc_new = self.compute_objective_J(E_new)

                    if J_new < J_current:
                        E_s = E_new
                        J_current = J_new
                        assoc_current = assoc_new
                        improved = True
                        print(f"  [close] Removed edge {e}, J = {J_current:.2f}")
                        break

            if improved:
                continue

            # 'swap' operation (Algorithm 2, lines 13-18)
            E_not_selected = self.N_c - E_s
            for e_new in E_not_selected:
                for e_old in list(E_s):
                    E_new = (E_s - {e_old}) | {e_new}
                    J_new, assoc_new = self.compute_objective_J(E_new)

                    if J_new < J_current:
                        E_s = E_new
                        J_current = J_new
                        assoc_current = assoc_new
                        improved = True
                        print(f"  [swap] {e_old} -> {e_new}, J = {J_current:.2f}")
                        break
                if improved:
                    break

            if not improved:
                print(f"Converged at iteration {t+1}")
                break

        return EdgeSelectionResult(
            selected_edges=E_s,
            node_associations=assoc_current,
            objective_value=J_current,
        )


def run_los(
    candidate_edges: List[int],
    all_nodes: List[int],
    communication_costs_ne: Dict[Tuple[int, int], float],
    communication_costs_ec: Dict[int, float],
    similarity_matrix: np.ndarray,
    data_sizes: Dict[int, int],
    kappa_c: int = 10,
    gamma: float = 2800.0,
    B_e: int = 10,
    T_max: int = 30,
    initial_edges: Optional[Set[int]] = None,
) -> EdgeSelectionResult:
    """
    Convenience function to run the LoS algorithm.

    Args:
        candidate_edges: Nodes that can be edge aggregators
        all_nodes: All node IDs
        communication_costs_ne: Node-to-edge communication costs
        communication_costs_ec: Edge-to-cloud communication costs
        similarity_matrix: N x N similarity matrix
        data_sizes: Data size per node
        kappa_c: Edge aggregation epochs per cloud round
        gamma: Trade-off weight
        B_e: Max nodes per edge
        T_max: Max iterations
        initial_edges: Initial edge set (optional)

    Returns:
        EdgeSelectionResult
    """
    los = LocalSearchEdgeSelection(
        candidate_edges=candidate_edges,
        all_nodes=all_nodes,
        communication_costs_ne=communication_costs_ne,
        communication_costs_ec=communication_costs_ec,
        similarity_matrix=similarity_matrix,
        data_sizes=data_sizes,
        kappa_c=kappa_c,
        gamma=gamma,
        B_e=B_e,
        T_max=T_max,
    )
    return los.run(initial_edges=initial_edges)


if __name__ == "__main__":
    # Test the LoS algorithm
    print("Testing LoS Algorithm")
    print("=" * 50)

    # Sample: 8 nodes, all candidates for edge
    num_nodes = 8
    candidate_edges = list(range(num_nodes))

    # Random diversity matrix
    np.random.seed(42)
    S = np.random.rand(num_nodes, num_nodes)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 0)

    # Communication costs
    c_ne = {}
    for n in range(num_nodes):
        for e in candidate_edges:
            if n == e:
                c_ne[(n, e)] = 0.0
            else:
                c_ne[(n, e)] = abs(n - e) * 50 + np.random.rand() * 20

    c_ec = {e: (e + 1) * 100 for e in candidate_edges}

    # Data sizes
    data_sizes = {i: 180 for i in range(num_nodes)}

    # Run LoS
    result = run_los(
        candidate_edges=candidate_edges,
        all_nodes=list(range(num_nodes)),
        communication_costs_ne=c_ne,
        communication_costs_ec=c_ec,
        similarity_matrix=S,
        data_sizes=data_sizes,
        kappa_c=10,
        gamma=2800.0,
        B_e=10,
        T_max=30,
    )

    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"Selected edges: {result.selected_edges}")
    print(f"Objective value: {result.objective_value:.2f}")
    print(f"Node associations: {result.node_associations.associations}")
