"""
Console reporting for particle-overlap diagnostics.

The reference particle loop performs:

    contact_neighbors = [j for j, ov in overlaps if ov]

    if contact_neighbors:
        print(
            f" [Frame {frame}] Particle {i0} OVERLAPS with "
            f"{len(contact_neighbors)} neighbor(s): {contact_neighbors}"
        )

    if i0 == viz_particle:
        print(
            f"    [Visualization particle {viz_particle}]"
        )

Important compatibility behavior
--------------------------------
- Normal Python truth-value testing is used for each overlap result.
- Neighbour order is preserved.
- Duplicate neighbour IDs are preserved.
- Neighbour IDs are returned unchanged.
- No Boolean conversion, sorting, or deduplication is performed.
- The visualization marker is printed even when no overlap exists.
"""

from __future__ import annotations


def collect_overlapping_neighbors(
    overlaps,
):
    """
    Return neighbour IDs whose overlap result is truthy.

    Parameters
    ----------
    overlaps
        Ordered iterable of ``(neighbor_id, overlap_result)`` pairs.

    Returns
    -------
    list
        Ordered neighbour IDs selected by the exact reference expression:

            [j for j, ov in overlaps if ov]
    """
    return [
        j
        for j, overlap in overlaps
        if overlap
    ]


def report_particle_overlap_diagnostic(
    frame,
    i0,
    overlaps,
    viz_particle,
):
    """
    Print the reference overlap messages for one central particle.

    Returns
    -------
    None
        The reference inline reporting block has no return value.
    """
    contact_neighbors = (
        collect_overlapping_neighbors(
            overlaps
        )
    )

    if contact_neighbors:
        print(
            f" [Frame {frame}] Particle {i0} OVERLAPS with "
            f"{len(contact_neighbors)} neighbor(s): "
            f"{contact_neighbors}"
        )

    if i0 == viz_particle:
        print(
            f"    [Visualization particle {viz_particle}]"
        )
