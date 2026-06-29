"""
Projected overlap area for one selected pair of polygonal faces.

FACE OVERLAP COMPUTATION OVERVIEW
==================================
This module computes the overlap area between two faces after projection.

Finds the intersection area of two polygonal faces when:
1. Central face stays in its original plane
2. Neighbor face rotated to be parallel to central
3. Both projected onto central face's plane
4. Intersection area computed in 2D

DEFINITION OF A_ij
==================
Given two finite polygonal faces in global 3D coordinates:

    Pc : central particle face
    Pj : neighbor particle face

The overlap area A_ij is computed as:

1. Compute plane data (center, normal, basis vectors)
   For both Pc and Pj

2. Rotate Pj about its OWN centroid
   Until its plane is parallel to Pc's plane
   Uses minimum rotation (smallest angle)

3. Project both polygons onto Pc's plane
   Orthogonal projection
   Both now coplanar

4. Express both in same 2D coordinate system
   Use Pc's basis vectors (u, v)
   Both become 2D polygons

5. Clean up 2D polygons
   Deduplicate points (remove duplicates)
   Order counter-clockwise (CCW)

6. Create Shapely polygons
   Validate geometry
   Make valid if needed

7. Compute intersection area
   Shapely polygon intersection
   Returns scalar area

MATHEMATICAL FOUNDATION
=======================
Projection orthogonal means:
    Project point p onto plane (point, normal)
    p_proj = p - ((p - point) \dot normal) * normal

This gives closest point on plane.

Minimum rotation:
    Given two unit vectors (normals n1, n2)
    Find rotation R such that R*n1 equals n2
    R is unique (or R and R + 180°)
    Choose minimum rotation angle

Intersection area:
    Two coplanar polygons
    Shapely computes intersection
    Returns area of overlapping region

ALGORITHM STEPS (15 TOTAL)
==========================
1. Validate inputs (as_3d_polygon)
2. Compute central plane data (plane_data)
3. Compute neighbor original plane data (plane_data)
4. Find minimum rotation to parallelize normals
5. Rotate neighbor about its centroid
6. Verify rotation (plane_data on rotated)
7. Project both to central plane
8. Compute projection residuals
9. Express in central's 2D basis
10. Deduplicate 2D points
11. Order points counter-clockwise
12. Check 2D areas
13. Create Shapely polygons
14. Compute intersection area
15. Assemble result record

ERROR HANDLING
==============

Returns zero_overlap_result():
    status = "failed"
    reason = "max_overlap_face_area_failed"
    error = exception message

Allows graceful degradation:
    Problematic pairs don't crash workflow
    Logged and continued

DIAGNOSTIC OUTPUT
=================
If debug=True:

    Prints at each major step:
    1. Starting message
    2. Input shape checks
    3. Plane data (center, normal, basis)
    4. Normal alignment before/after
    5. Centroid shifts
    6. Projection residuals
    7. 2D polygon areas
    8. Final result

Example debug output:

    [max_overlap_face_area] Starting projected face-overlap calculation.
    [max_overlap_face_area] Pc shape: (4, 3)
    [max_overlap_face_area] Pj shape: (4, 3)
    [max_overlap_face_area] Central face centroid cC: [0.5 0.5 0.5]
    [max_overlap_face_area] Normal dot before rotation nC·nJ: 0.12345678
    [max_overlap_face_area] Max |Pc_projected distance to Pc plane|: 1.23e-14
    [max_overlap_face_area] Pc2 shoelace area: 1.00000000
    [max_overlap_face_area] Shapely Pc2 area: 1.00000000


INTEGRATION
===========
Called by:
    max_overlap_all_neighbours()
    For each neighbor pair

Receives:
    Pc, Pj: face vertex arrays (both (V, 3))

Returns:
    Result dict (success or failure)
"""

from __future__ import annotations
import numpy as np

# Import geometry helpers
from .face_overlap_area import polygon_area_2d
from .face_overlap_coordinates import project_to_2d
from .face_overlap_deduplication import deduplicate_points_2d
from .face_overlap_geometry import as_3d_polygon
from .face_overlap_intersection import polygon_intersection_area
from .face_overlap_ordering import order_points_ccw
from .face_overlap_parallelization import minimum_rotation_to_parallelize_normals
from .face_overlap_plane import plane_data
from .face_overlap_polygon import make_valid_polygon
from .face_overlap_projection import project_points_to_plane
from .face_overlap_results import successful_overlap_result, zero_overlap_result


def max_overlap_face_area(Pc, Pj, *, debug=False):
    """
    Compute the projected overlap area between two selected 3D faces.

    FUNCTION PURPOSE
    ================
    Computes overlap area by projecting two faces onto same plane.

    Algorithm:
    1. Rotate neighbor face to be parallel to central
    2. Project both onto central's plane
    3. Convert to 2D coordinates
    4. Compute intersection area

    WORKFLOW
    ========
    Input: Two 3D faces (vertex arrays)
    
    Output: Result dict with:
        - Aij: overlap area (or 0.0 if failed)
        - status: "ok" or "failed"
        - diagnostic: residuals, angles, areas

    Returns immediately on any exception.
    No propagation of errors.

    Parameters
    ----------
    Pc : ndarray, shape (N, 3)
        Vertices of central-particle face in global coordinates.

    Pj : ndarray, shape (M, 3)
        Vertices of neighbor-particle face in global coordinates.

    debug : bool, optional
        Enable detailed diagnostic output.
        Prints step-by-step progress.
        Useful for debugging problem cases.
        Default: False

    Returns
    -------
    dict
        Successful or failed overlap-computation result.
        
        On success (status="ok"):
            Contains overlap area (Aij) and diagnostics
        
        On failure (status="failed"):
            Aij=0.0, includes reason and error
        
        Always returns dict (never raises exception)
    """

    # Create diagnostic output function
    # Prints only if debug=True
    def dbg(message):
        if debug: print(f"[max_overlap_face_area] {message}")

    try:
        # =========================================================================
        # STEP 1: VALIDATE AND STANDARDIZE INPUTS
        # =========================================================================
        # Ensure both inputs are valid 3D polygons
        Pc = as_3d_polygon(Pc, "Pc")
        Pj = as_3d_polygon(Pj, "Pj")

        dbg("Starting projected face-overlap calculation.")
        dbg(f"Pc shape: {Pc.shape}"); dbg(f"Pj shape: {Pj.shape}")

        # =====================================================================
        # STEP 2: COMPUTE REFERENCE PLANE DATA FOR Pc (CENTRAL)
        # =====================================================================
        # Central face stays in its original plane
        # This becomes the reference plane for projection
        cC, nC, uC, vC = plane_data(Pc, "Pc")
        dbg(f"Central face centroid cC: {cC}"); dbg(f"Central face normal nC: {nC}")
        dbg(f"Central in-plane basis uC: {uC}"); dbg(f"Central in-plane basis vC: {vC}")

        # =====================================================================
        # STEP 3: COMPUTE ORIGINAL PLANE DATA FOR Pj (NEIGHBOR)
        # =====================================================================
        # Get original neighbor plane (before rotation)

        cJ, nJ, _, _ = plane_data(Pj, "Pj")
        dbg(f"Neighbour face centroid cJ: {cJ}"); dbg(f"Neighbour face normal nJ: {nJ}")
        dbg(f"Normal dot before rotation nC·nJ: {float(np.dot(nC, nJ)):.8f}")

        # =====================================================================
        # STEP 4: FIND MINIMUM ROTATION TO PARALLELIZE NORMALS
        # =====================================================================
        # Find rotation that makes neighbor's plane parallel to central's
        # Smallest rotation angle needed
        rotation = minimum_rotation_to_parallelize_normals(n_from=nJ, n_reference=nC)
        dbg(f"Rotation matrix R:\n{rotation}")

        # =====================================================================
        # STEP 5: ROTATE Pj ABOUT ITS OWN CENTROID
        # =====================================================================
        # Rotate neighbor about its centroid
        # This makes planes parallel without changing centroid location
        Pj_centered = Pj - cJ                          # translate to origin
        Pj_rotated = Pj_centered @ rotation.T + cJ     # rotate then translate back
        cJ_after = Pj_rotated.mean(axis=0)             # Recompute centroid (should be same, but numerical precision)

        dbg(f"Neighbour centroid before rotation: {cJ}"); dbg(f"Neighbour centroid after rotation : {cJ_after}")
        dbg(f"Centroid shift after rotation     : {np.linalg.norm(cJ_after - cJ):.6e}")

        # =====================================================================
        # STEP 6: VERIFY ROTATION RESULTED IN PARALLEL PLANES
        # =====================================================================
        # Compute plane data for rotated neighbor

        cJ_rot, nJ_rot, _, _ = plane_data(Pj_rotated, "Pj_rotated")
        # Compute parallelism: |nC · nJ_rot| should be 1
        parallel_score = abs(float(np.dot(nC, nJ_rot)))

        dbg(f"Neighbour normal after rotation nJ_rot: {nJ_rot}")
        dbg(f"|nC · nJ_rot| after rotation: {parallel_score:.8f}")

        # Warn if not perfectly parallel
        if parallel_score < 1.0 - 1.0e-6:
            dbg(f"WARNING: Rotated neighbour face is not perfectly parallel to central face. |dot|={parallel_score:.8f}")

        # Preserve the original unused assignment.
        _ = cJ_rot

        # =====================================================================
        # STEP 7: PROJECT BOTH POLYGONS ONTO CENTRAL'S PLANE
        # =====================================================================
        # Orthogonal projection onto central's plane
        # Both become coplanar
        Pc_projected = project_points_to_plane(Pc, plane_point=cC, plane_normal=nC)
        Pj_projected = project_points_to_plane(Pj_rotated, plane_point=cC, plane_normal=nC)

        # =====================================================================
        # STEP 8: CALCULATE PROJECTION RESIDUALS
        # =====================================================================
        # Check accuracy of projection
        dist_Pc_to_plane = (Pc_projected - cC) @ nC
        dist_Pj_to_plane = (Pj_projected - cC) @ nC

        max_dist_Pc = float(np.max(np.abs(dist_Pc_to_plane)))
        max_dist_Pj = float(np.max(np.abs(dist_Pj_to_plane)))

        dbg(f"Max |Pc_projected distance to Pc plane|: {max_dist_Pc:.6e}")
        dbg(f"Max |Pj_projected distance to Pc plane|: {max_dist_Pj:.6e}")

        # Warn if residuals large
        if max_dist_Pc > 1.0e-8: dbg(f"WARNING: Pc projection residual is larger than expected: {max_dist_Pc:.6e}")
        if max_dist_Pj > 1.0e-8: dbg(f"WARNING: Pj projection residual is larger than expected: {max_dist_Pj:.6e}")

        # =====================================================================
        # STEP 9: EXPRESS BOTH POLYGONS IN SAME 2D BASIS
        # =====================================================================
        # Convert 3D coplanar coordinates to 2D in plane's basis
        # Both polygons now in same 2D coordinate system
        Pc2 = project_to_2d(Pc_projected, origin=cC, u=uC, v=vC)
        Pj2 = project_to_2d(Pj_projected, origin=cC, u=uC, v=vC)

        dbg(f"Pc2 before cleanup:\n{Pc2}"); dbg(f"Pj2 before cleanup:\n{Pj2}")

        # =====================================================================
        # STEP 10: DEDUPLICATE AND ORDER THE 2D POLYGON POINTS
        # =====================================================================
        # Remove duplicate points (numerical precision)
        # Order counter-clockwise (CCW)
        Pc2 = deduplicate_points_2d(Pc2, tol=1.0e-10)
        Pj2 = deduplicate_points_2d(Pj2, tol=1.0e-10)

        Pc2 = order_points_ccw(Pc2)
        Pj2 = order_points_ccw(Pj2)

        dbg(f"Pc2 after cleanup:\n{Pc2}"); dbg(f"Pj2 after cleanup:\n{Pj2}")

        # =====================================================================
        # STEP 11: CHECK 2D POLYGON AREAS
        # =====================================================================
        # Compute areas using shoelace formula
        Pc2_area_shoelace = polygon_area_2d(Pc2)
        Pj2_area_shoelace = polygon_area_2d(Pj2)

        # Raise error if areas too small
        dbg(f"Pc2 shoelace area: {Pc2_area_shoelace:.8f}"); dbg(f"Pj2 shoelace area: {Pj2_area_shoelace:.8f}")

        if Pc2_area_shoelace <= 1.0e-14: raise ValueError(f"Projected central polygon has near-zero area: {Pc2_area_shoelace}.")
        if Pj2_area_shoelace <= 1.0e-14: raise ValueError(f"Projected neighbour polygon has near-zero area: {Pj2_area_shoelace}.")

        # =====================================================================
        # STEP 12: CONSTRUCT VALID SHAPELY POLYGONAL GEOMETRIES
        # =====================================================================
        # Create Shapely polygon objects
        # Validate and fix geometry if needed
        poly_C = make_valid_polygon(Pc2, "Pc2", debug=debug)
        poly_J = make_valid_polygon(Pj2, "Pj2", debug=debug)

        dbg(f"Shapely Pc2 area: {poly_C.area:.8f}"); dbg(f"Shapely Pj2 area: {poly_J.area:.8f}")

        # =====================================================================
        # STEP 13 & 14: COMPUTE INTERSECTION AREA
        # =====================================================================
        # Find intersection of two polygons
        # Compute area of overlapping region
        Aij = polygon_intersection_area(poly_C, poly_J, debug_print=dbg)

        # =====================================================================
        # STEP 15: ASSEMBLE SUCCESSFUL RESULT
        # =====================================================================
        # Package all results into standardized record
        return successful_overlap_result(Aij=Aij, poly_C_area=poly_C.area, poly_J_area=poly_J.area, parallel_score=parallel_score, cJ_after=cJ_after, cJ=cJ, max_dist_Pc=max_dist_Pc, max_dist_Pj=max_dist_Pj)

    # =========================================================================
    # EXCEPTION HANDLING: CATCH ALL ERRORS
    # =========================================================================
    # If ANY error occurs at any step, return failed result
    except Exception as exc:
        if debug: print(f"[max_overlap_face_area ERROR] {exc}")
        return zero_overlap_result(reason="max_overlap_face_area_failed", error=exc)
