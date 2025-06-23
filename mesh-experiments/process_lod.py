from typing import List

import torch
from asset_rep import MeshData


def optimize_mesh(mesh: MeshData, target_faces = 0.5) -> MeshData:
    if type(target_faces) is float:
        target_faces = int(mesh.triangles.shape[0] * target_faces)
    while mesh.triangles.shape[0] > target_faces:
        # print(f"Current mesh has {mesh.triangles.shape[0]} triangles, computing edge collapses...")
        print(f"Current mesh has {mesh.triangles.shape[0]} triangles, target is {target_faces}.")
        pos_quadrics = compute_position_quadrics(mesh)
        edges, costs, v_opt = compute_edge_costs(mesh, pos_quadrics)
        cheapest_edges = get_edge_collapse_batch(mesh, edges, costs)
        if cheapest_edges.shape[0] == 0:
            print(f"[Warning] No valid edges found for collapse. Stopping optimization.")
            break
        collapse_edges = cheapest_edges[:mesh.triangles.shape[0] - target_faces]
        final_edges = edges[collapse_edges]
        v_stars = v_opt[collapse_edges]
        for edge, v_star in zip(final_edges, v_stars):
            collapse_edge(mesh, edge, v_star)
        mesh.remove_degens()
    mesh.remove_orphans()
    mesh.validate()


def get_edge_collapse_batch(mesh, edges, costs, batch_size: int = 32):
    cheapest_edges = torch.argsort(costs)[:batch_size * 2]  # Get twice the batch size to ensure we have enough
    unique_edges, counts = torch.unique(edges[cheapest_edges], return_counts=True)
    repeated_indices = unique_edges[counts > 1]
    for i, idx in enumerate(repeated_indices):
        repeats = torch.where((edges[cheapest_edges][:, 0] == idx) | (edges[cheapest_edges][:, 1] == idx))[0][1:]
        cheapest_edges[repeats] = -1
    cheapest_edges = cheapest_edges[cheapest_edges >= 0]
    valid_edges = valid_collapse_mask(mesh, edges[cheapest_edges])
    cheapest_edges = cheapest_edges[valid_edges]
    if cheapest_edges.shape[0] > batch_size:
        cheapest_edges = cheapest_edges[:batch_size]
    else:
        print(f"[Warning] Only {cheapest_edges.shape[0]} edges available for collapse, less than batch size {batch_size}.")
    return cheapest_edges


def valid_collapse_mask(mesh: MeshData, candidate_edges: torch.Tensor):
    # candidate_edges: (B, 2) -- indices into mesh.positions
    pos_per_tri = mesh.polyvert_attrs[mesh.triangles, 0]  # (F, 3)
    is_valid = torch.zeros(candidate_edges.shape[0], dtype=torch.bool, device=candidate_edges.device)
    for k, (i, j) in enumerate(candidate_edges):
        # For each edge, find triangles that contain both i and j as vertex indices
        shared = ((pos_per_tri == i).any(dim=1)) & ((pos_per_tri == j).any(dim=1))
        count = shared.sum().item()
        is_valid[k] = (count == 2)
    return is_valid


def compute_position_quadrics(mesh: MeshData) -> torch.Tensor:
    Qgeom = torch.zeros((mesh.positions.shape[0], 4, 4), device=mesh.positions.device)

    # Gather triangle vertex positions
    tri_positions = mesh.positions[mesh.polyvert_attrs[mesh.triangles][:, :, 0]]  # shape: [F, 3, 3]
    v0, v1, v2 = tri_positions[:, 0], tri_positions[:, 1], tri_positions[:, 2]

    # Compute triangle normals
    n = torch.linalg.cross(v1 - v0, v2 - v0)  # shape: [F, 3]
    n_norm = n.norm(dim=1, keepdim=True) + 1e-8
    n = n / n_norm  # normalize normals

    # Skip degenerate triangles
    valid_mask = n_norm.squeeze() >= 1e-6
    if not valid_mask.all():
        print(f"[Warning] Skipping {(~valid_mask).sum().item()} degenerate triangles.")

    n = n[valid_mask]
    v0 = v0[valid_mask]
    triangles = mesh.triangles[valid_mask]

    # Compute plane equations
    d = -torch.einsum('ij,ij->i', n, v0)  # shape: [F]
    plane = torch.cat([n, d.unsqueeze(1)], dim=1)  # shape: [F, 4]

    # Compute quadric matrices
    Kp = plane.unsqueeze(2) @ plane.unsqueeze(1)  # outer product, shape: [F, 4, 4]

    # Symmetry and eigenvalue checks
    asym = (Kp - Kp.transpose(1, 2)).abs().amax(dim=(1, 2))
    if (asym > 1e-5).any():
        print(f"[Warning] {torch.sum(asym > 1e-5).item()} quadrics are not symmetric. Max asym: {asym.max().item():.6f}")

    eigvals = torch.linalg.eigvalsh(Kp)
    if (eigvals < -1e-5).any():
        print(f"[Warning] {torch.sum((eigvals < -1e-5).any(dim=1)).item()} quadrics have negative eigenvalues.")

    # Accumulate quadrics for each vertex
    for i in range(3):  # Iterate over triangle vertices
        indices = mesh.polyvert_attrs[triangles[:, i], 0]
        Qgeom.index_add_(0, indices, Kp)

    return Qgeom


def compute_edge_costs(mesh: MeshData, pos_quadrics: torch.Tensor):
    edges = torch.cat([
        mesh.triangles[:, [0, 1]],
        mesh.triangles[:, [1, 2]],
        mesh.triangles[:, [2, 0]],
    ], dim=0)  # shape: [3F, 2]
    pos_edges = mesh.polyvert_attrs[:, 0][edges]
    sorted_edges = torch.sort(pos_edges, dim=1)[0]
    pos_edges = torch.unique(sorted_edges, dim=0)  # shape: [E, 2]
    Qi = pos_quadrics[pos_edges[:, 0]]  # [E, 4, 4]
    Qj = pos_quadrics[pos_edges[:, 1]]  # [E, 4, 4]
    Q_sum = Qi + Qj                     # [E, 4, 4]

    def _compute_collapse_costs(Q_sum: torch.Tensor):
        A = Q_sum[:, :3, :3]
        b = -Q_sum[:, :3, 3]
        pinvA = torch.linalg.pinv(A)  # Moore-Penrose pseudoinverse
        v_opt = torch.einsum('eij,ej->ei', pinvA, b)  # [E, 3]

        ones = torch.ones(pos_edges.shape[0], 1, device=mesh.positions.device)
        v4 = torch.cat([v_opt, ones], dim=1)  # [E, 4]
        costs = torch.einsum('bi,bij,bj->b', v4, Q_sum, v4)  # [E,]
        return costs, v_opt
    costs, v_opt = _compute_collapse_costs(Q_sum)
    return pos_edges, costs, v_opt


def collapse_edge(mesh: MeshData, edge: torch.Tensor, v_opt: torch.Tensor) -> MeshData:
    pi, pj = edge
    new_position = mesh.merge_positions(pi, pj, v_opt)

    relevant_polyverts = torch.logical_or(mesh.polyvert_attrs[:, 0] == pi, mesh.polyvert_attrs[:, 0] == pj)
    mesh.polyvert_attrs[relevant_polyverts, 0] = new_position

