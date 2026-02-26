"""
Scene Geometry and Ray Intersection Utilities for LiDAR Simulation

This module provides a ray-casting framework for simulating LiDAR interactions
with 3D scene geometry. It contains the following components:

Material properties (reflectivity, retro-reflectivity) for radiometric modeling.
Ray-geometry intersection classes for common primitives (Sphere, Plane,
AxisAlignedBox, Triangle).
A Scene container that manages multiple objects and performs closest-hit ray
queries.

The intersection methods return RayHit dataclasses containing intersection
distance [m], world-space point [m], surface normal [unit], and material
properties. These results are consumed by the Lidar class in lidar.py to
compute received power, detection probability, and range measurements.
"""

from dataclasses import dataclass

import numpy as np

from .math_utils import _as_vector3, _normalize, eps


@dataclass(frozen=True)
class Material:
    """
    Surface material properties for LiDAR radiometric simulation.

    This is a frozen dataclass, meaning instances are immutable after creation.
    Reflectivity values are automatically clamped to [0, 1] inside __post_init__.

    :param reflectivity:       Lambertian diffuse reflectivity coefficient, range [0, 1].
    :param retro_reflectivity: Retro-reflective (corner-cube) component, range [0, 1].
    :param name:               Descriptive label used for debugging and metadata logging.
    """
    reflectivity: float = 0.5         # [0..1] Lambertian diffuse reflectivity coefficient
    retro_reflectivity: float = 0.0   # [0..1] retro-reflective (corner-cube) component
    name: str = "default"             # descriptive label for debugging / metadata logging

    def __post_init__(self):
        # Clamp reflectivity to the physically valid range [0, 1].
        # object.__setattr__ is required here because the dataclass is frozen (immutable).
        object.__setattr__(self, "reflectivity", float(np.clip(self.reflectivity, 0.0, 1.0)))
        object.__setattr__(self, "retro_reflectivity", float(np.clip(self.retro_reflectivity, 0.0, 1.0)))


@dataclass
class RayHit:
    """
    Result of a successful ray-geometry intersection query.

    Returned by SceneObject.intersect() when the ray hits a surface.

    :param distance:  Distance along the ray from origin to intersection point [m].
    :param point:     3D world-space coordinates of the hit point [m].
    :param normal:    Outward surface normal at the hit point [unit].
    :param material:  Material properties at the hit surface.
    :param object_id: Identifier of the intersected SceneObject.
    """
    distance: float          # [m]  distance along ray from origin to intersection point
    point: np.ndarray        # [m]  3D world-space coordinates of the hit point
    normal: np.ndarray       # [unit] outward surface normal at the hit point
    material: Material       # material properties at the hit surface
    object_id: str = ""      # identifier of the intersected SceneObject


class SceneObject:
    """
    Abstract base class for ray-traceable geometry primitives.

    Subclasses must override intersect() to implement ray-geometry math.
    Each object carries a Material (for radiometric modelling) and an
    object_id string (for tracking which object was hit in simulation output).
    """
    def __init__(self, material=None, object_id=None):
        self.material = material if material is not None else Material()  # default Lambertian 0.5
        self.object_id = str(object_id) if object_id is not None else self.__class__.__name__  # fallback to class name

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Compute ray-geometry intersection (abstract method).

        :param origin:    Ray origin in world space [m].
        :param direction: Ray direction (will be normalized internally) [unit].
        :param t_min:     Minimum valid distance [m]; hits closer are ignored.
        :param t_max:     Maximum valid distance [m]; hits farther are ignored.

        :return: RayHit if intersection is found within [t_min, t_max], else None.
        :raises NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError


class Sphere(SceneObject):
    """
    Solid sphere defined by centre and radius.

    Uses the standard quadratic formula for ray-sphere intersection.

    :param center:    Centre position of the sphere in world space [m].
    :param radius:    Radius of the sphere [m].
    :param material:  Surface material properties (defaults to Material()).
    :param object_id: Identifier string for this object.
    """
    def __init__(self, center, radius, material=None, object_id=None):
        super().__init__(material=material, object_id=object_id)
        self.center = _as_vector3(center, "center")  # [m] sphere centre in world space
        self.radius = float(radius)                   # [m] sphere radius
        if self.radius <= eps:
            raise ValueError("Sphere radius must be > 0.")

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Ray-sphere intersection via quadratic formula.

        Quadratic setup (direction is unit length, so a = 1):
            oc = origin - centre
            b  = dot(oc, direction)
            c  = dot(oc, oc) - r^2
            discriminant = b^2 - c
        Two solutions: t_near = -b - sqrt(disc),  t_far = -b + sqrt(disc).
        The closest t in [t_min, t_max] is returned.

        :return: RayHit or None.
        """
        origin = _as_vector3(origin, "origin")
        direction = _normalize(_as_vector3(direction, "direction"))
        oc = origin - self.center                                 # vector from centre to ray origin
        b = float(np.dot(oc, direction))                          # half b coefficient
        c = float(np.dot(oc, oc) - self.radius * self.radius)     # c coefficient
        disc = b * b - c                                          # discriminant
        if disc < 0.0:
            return None  # ray misses the sphere entirely

        sqrt_disc = np.sqrt(disc)
        t_near = -b - sqrt_disc  # entry point (closer)
        t_far = -b + sqrt_disc   # exit point (farther)

        # Select the closest valid intersection within [t_min, t_max].
        t_hit = None
        if t_min <= t_near <= t_max:
            t_hit = t_near           # prefer the entry point
        elif t_min <= t_far <= t_max:
            t_hit = t_far            # fall back to exit point (ray originates inside sphere)
        if t_hit is None:
            return None              # both intersections are out of the valid range

        point = origin + t_hit * direction           # world-space intersection point
        normal = _normalize(point - self.center)     # outward radial normal
        return RayHit(distance=float(t_hit), point=point, normal=normal, material=self.material, object_id=self.object_id)


class Plane(SceneObject):
    """
    Infinite plane defined by a point and a surface normal.

    Uses the parametric line-plane intersection formula.

    :param point:     Any point lying on the plane [m].
    :param normal:    Outward surface normal of the plane [unit].
    :param material:  Surface material properties (defaults to Material()).
    :param object_id: Identifier string for this object.
    """
    def __init__(self, point, normal, material=None, object_id=None):
        super().__init__(material=material, object_id=object_id)
        self.point = _as_vector3(point, "point")                # [m] any point on the plane
        self.normal = _normalize(_as_vector3(normal, "normal"))  # [unit] outward surface normal

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Ray-plane intersection.

        Formula:  t = dot(plane_point - origin, normal) / dot(direction, normal)
        If |denom| < eps the ray is parallel to the plane.
        The returned normal is flipped so it always faces the incoming ray.

        :return: RayHit or None.
        """
        origin = _as_vector3(origin, "origin")
        direction = _normalize(_as_vector3(direction, "direction"))
        denom = float(np.dot(direction, self.normal))  # dot(ray, normal)
        if abs(denom) < eps:
            return None  # ray is parallel to the plane; no intersection

        t_hit = float(np.dot(self.point - origin, self.normal) / denom)  # parametric distance along ray
        if not (t_min <= t_hit <= t_max):
            return None  # intersection is outside the valid range

        point = origin + t_hit * direction                                # world-space hit point
        out_normal = self.normal if denom < 0.0 else -self.normal          # flip normal to face the ray
        return RayHit(distance=t_hit, point=point, normal=out_normal, material=self.material, object_id=self.object_id)


class AxisAlignedBox(SceneObject):
    """
    Axis-aligned bounding box (AABB) defined by min and max corner coordinates.

    Uses the slab method: intersect the ray against three pairs of
    axis-perpendicular planes and compute the overlap of the resulting intervals.

    :param min_corner: Minimum (x, y, z) corner of the box [m].
    :param max_corner: Maximum (x, y, z) corner of the box [m].
    :param material:   Surface material properties (defaults to Material()).
    :param object_id:  Identifier string for this object.
    """
    def __init__(self, min_corner, max_corner, material=None, object_id=None):
        super().__init__(material=material, object_id=object_id)
        self.min_corner = _as_vector3(min_corner, "min_corner")  # [m] minimum (x,y,z) corner
        self.max_corner = _as_vector3(max_corner, "max_corner")  # [m] maximum (x,y,z) corner
        if np.any(self.max_corner <= self.min_corner):
            raise ValueError("max_corner must be strictly greater than min_corner on all axes.")

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Ray-AABB intersection via the slab method.

        For each axis the ray is intersected with the two bounding planes:
            t0 = (min - origin) / direction
            t1 = (max - origin) / direction
        The overall entry/exit distance is t_enter = max(t_small) and
        t_exit = min(t_big). If t_exit < t_enter the ray misses the box.

        The hit-face normal is determined by checking which face the
        intersection point is closest to (with a small tolerance for
        edge cases).

        :return: RayHit or None.
        """
        origin = _as_vector3(origin, "origin")
        direction = _normalize(_as_vector3(direction, "direction"))

        # Compute slab intersection distances for each axis.
        # Suppress divide-by-zero warnings for rays parallel to a face (direction ~= 0 produces inf).
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dir = 1.0 / direction                      # reciprocal of each direction component
            t0 = (self.min_corner - origin) * inv_dir  # distances to min-face slabs
            t1 = (self.max_corner - origin) * inv_dir  # distances to max-face slabs

        # Ensure t_small <= t_big on each axis (swap when ray direction is negative).
        t_small = np.minimum(t0, t1)
        t_big = np.maximum(t0, t1)
        t_enter = float(np.max(t_small))  # ray enters box when it enters the last slab
        t_exit = float(np.min(t_big))     # ray exits box when it exits the first slab

        if t_exit < t_enter:
            return None  # ray misses the box (exit before entry)
        if t_exit < t_min or t_enter > t_max:
            return None  # intersection interval is entirely outside [t_min, t_max]

        # Choose the closest valid intersection distance.
        t_hit = t_enter if t_enter >= t_min else t_exit
        if not (t_min <= t_hit <= t_max):
            return None  # no valid hit after clamping

        point = origin + t_hit * direction  # world-space hit point

        # Determine which face was hit by checking proximity to each bounding plane.
        normal = np.zeros(3, dtype=float)
        tol = 1e-8  # [m] tolerance for face identification
        if abs(point[0] - self.min_corner[0]) < tol:
            normal = np.array([-1.0, 0.0, 0.0])   # hit the -x face
        elif abs(point[0] - self.max_corner[0]) < tol:
            normal = np.array([1.0, 0.0, 0.0])    # hit the +x face
        elif abs(point[1] - self.min_corner[1]) < tol:
            normal = np.array([0.0, -1.0, 0.0])   # hit the -y face
        elif abs(point[1] - self.max_corner[1]) < tol:
            normal = np.array([0.0, 1.0, 0.0])    # hit the +y face
        elif abs(point[2] - self.min_corner[2]) < tol:
            normal = np.array([0.0, 0.0, -1.0])   # hit the -z face
        elif abs(point[2] - self.max_corner[2]) < tol:
            normal = np.array([0.0, 0.0, 1.0])    # hit the +z face
        else:
            # Fallback for edge/corner hits: pick the axis whose slab boundary
            # is closest to the computed t_enter value.
            face_idx = int(np.argmin(np.abs(t_small - t_enter)))
            normal = np.zeros(3, dtype=float)
            normal[face_idx] = np.sign(direction[face_idx]) * -1.0  # outward-facing normal

        return RayHit(distance=t_hit, point=point, normal=normal, material=self.material, object_id=self.object_id)


class Triangle(SceneObject):
    """
    Single triangle primitive defined by three vertices.

    Uses the Moller-Trumbore algorithm for fast ray-triangle intersection
    with barycentric coordinate checks.

    :param v0:        First vertex of the triangle [m].
    :param v1:        Second vertex of the triangle [m].
    :param v2:        Third vertex of the triangle [m].
    :param material:  Surface material properties (defaults to Material()).
    :param object_id: Identifier string for this object.
    """
    def __init__(self, v0, v1, v2, material=None, object_id=None):
        super().__init__(material=material, object_id=object_id)
        self.v0 = _as_vector3(v0, "v0")  # [m] first vertex
        self.v1 = _as_vector3(v1, "v1")  # [m] second vertex
        self.v2 = _as_vector3(v2, "v2")  # [m] third vertex
        # Pre-compute the face normal for later use in intersection and shading.
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        normal = np.cross(edge1, edge2)  # area-weighted normal (cross product of edges)
        if np.linalg.norm(normal) <= eps:
            raise ValueError("Triangle vertices must be non-collinear.")
        self.face_normal = _normalize(normal)  # [unit] outward face normal

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Ray-triangle intersection via the Moller-Trumbore algorithm.

        Steps:
        1. Compute pvec = cross(direction, edge2), det = dot(edge1, pvec).
           If |det| < eps the ray is parallel to the triangle plane.
        2. Compute barycentric coordinate u; reject if u is outside [0, 1].
        3. Compute barycentric coordinate v; reject if v < 0 or u+v > 1.
        4. Compute parametric distance t; reject if t is outside [t_min, t_max].
        5. Flip the normal to face the incoming ray.

        :return: RayHit or None.
        """
        origin = _as_vector3(origin, "origin")
        direction = _normalize(_as_vector3(direction, "direction"))

        edge1 = self.v1 - self.v0  # edge from v0 to v1
        edge2 = self.v2 - self.v0  # edge from v0 to v2
        pvec = np.cross(direction, edge2)           # used to compute determinant and u
        det = float(np.dot(edge1, pvec))            # determinant of the 3x3 system
        if abs(det) < eps:
            return None  # ray is parallel to triangle plane

        inv_det = 1.0 / det
        tvec = origin - self.v0                                # vector from v0 to ray origin
        u = float(np.dot(tvec, pvec) * inv_det)                # barycentric coord u
        if u < 0.0 or u > 1.0:
            return None  # hit is outside the triangle (u dimension)

        qvec = np.cross(tvec, edge1)                           # used to compute v and t
        v = float(np.dot(direction, qvec) * inv_det)           # barycentric coord v
        if v < 0.0 or (u + v) > 1.0:
            return None  # hit is outside the triangle (v dimension)

        t_hit = float(np.dot(edge2, qvec) * inv_det)           # parametric distance along ray
        if not (t_min <= t_hit <= t_max):
            return None  # intersection is outside the valid range

        point = origin + t_hit * direction                                          # world-space hit point
        normal = self.face_normal if np.dot(direction, self.face_normal) < 0.0 else -self.face_normal  # flip to face ray
        return RayHit(distance=t_hit, point=point, normal=normal, material=self.material, object_id=self.object_id)


class Scene:
    """
    Container and closest-hit ray casting manager for scene geometry.

    Holds a list of SceneObject instances. cast_ray() tests the ray against
    every object and returns the nearest RayHit (or None).
    """
    def __init__(self, objects=None):
        self.objects = list(objects) if objects is not None else []  # list of SceneObject instances

    def add(self, obj):
        """Add a SceneObject to the scene. Raises TypeError for non-SceneObject inputs."""
        if not isinstance(obj, SceneObject):
            raise TypeError("Scene accepts SceneObject instances.")
        self.objects.append(obj)

    def cast_ray(self, origin, direction, max_range=float("inf"), min_range=eps):
        """
        Cast a ray through the scene and return the closest intersection.

        Iterates over all scene objects, progressively narrowing t_max to the
        current closest hit distance so that far objects are culled early.

        :param origin:    Ray origin [m].
        :param direction: Ray direction (will be normalized) [unit].
        :param max_range: Maximum query distance [m].
        :param min_range: Minimum query distance [m].

        :return: Closest RayHit, or None if the ray hits nothing.
        """
        origin = _as_vector3(origin, "origin")
        direction = _normalize(_as_vector3(direction, "direction"))

        best_hit = None
        best_dist = float(max_range)  # shrinks as closer hits are found
        for obj in self.objects:
            # Pass current best_dist as t_max so objects farther away are skipped quickly.
            hit = obj.intersect(origin, direction, t_min=min_range, t_max=best_dist)
            if hit is None:
                continue
            if hit.distance < best_dist:
                best_dist = hit.distance  # tighten the search range
                best_hit = hit

        return best_hit  # closest intersection, or None
