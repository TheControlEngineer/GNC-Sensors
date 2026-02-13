"""
Scene geometry and ray intersection utilities for LiDAR simulation.

This module provides a ray casting framework for simulating LiDAR interactions with
3D scene geometry. It includes:
Material properties (reflectivity, retro reflectivity) for radiometric modeling
Ray geometry intersection classes for common primitives (Sphere, Plane, AxisAlignedBox, Triangle)
A Scene container that manages multiple objects and performs closest hit ray queries

The intersection methods return RayHit dataclasses containing intersection distance [m],
world space point [m], surface normal [unit], and material properties.
These are consumed by the Lidar class in lidar.py to compute received power,
detection probability, and range measurements.
"""

from dataclasses import dataclass

import numpy as np

from .math_utils import _as_vector3, _normalize, eps


@dataclass(frozen=True)
class Material:
    """
    Surface material properties for LiDAR radiometric simulation.

    Frozen dataclass: instances are immutable after creation.
    Reflectivity values are automatically clamped to [0, 1] in __post_init__.
    """
    reflectivity: float = 0.5         # [0..1] Lambertian diffuse reflectivity coefficient
    retro_reflectivity: float = 0.0   # [0..1] retro reflective (corner cube) component
    name: str = "default"             # descriptive label for debugging / metadata logging

    def __post_init__(self):
        # Clamp reflectivity values to physically valid range [0, 1].
        # Uses object.__setattr__ because the dataclass is frozen (immutable).
        object.__setattr__(self, "reflectivity", float(np.clip(self.reflectivity, 0.0, 1.0)))
        object.__setattr__(self, "retro_reflectivity", float(np.clip(self.retro_reflectivity, 0.0, 1.0)))


@dataclass
class RayHit:
    """
    Result of a successful ray geometry intersection query.

    Returned by SceneObject.intersect() when the ray hits a surface.
    """
    distance: float          # [m]  distance along ray from origin to intersection point
    point: np.ndarray        # [m]  3D world space coordinates of the hit point
    normal: np.ndarray       # [unit] outward surface normal at the hit point
    material: Material       # material properties at the hit surface
    object_id: str = ""      # identifier of the intersected SceneObject


class SceneObject:
    """
    Abstract base class for ray traceable geometry primitives.

    Subclasses must override intersect() to implement raygeometry math.
    Each object carries a Material (for radiometric modelling) and an
    object_id string (for tracking which object was hit in simulation output).
    """
    def __init__(self, material=None, object_id=None):
        self.material = material if material is not None else Material()  # default Lambertian 0.5
        self.object_id = str(object_id) if object_id is not None else self.__class__.__name__  # fallback to class name

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Compute rayÃ¢â‚¬â€œgeometry intersection (abstract).

        :param origin:    ray origin in world space [m]
        :param direction: ray direction (will be normalized internally) [unit]
        :param t_min:     minimum valid distance [m]; hits closer are ignored
        :param t_max:     maximum valid distance [m]; hits farther are ignored

        :return: RayHit if intersection within [t_min, t_max], else None
        :raises NotImplementedError: must be overridden by subclasses
        """
        raise NotImplementedError


class Sphere(SceneObject):
    """
    Solid sphere defined by centre and radius.

    Uses the standard quadratic formula for rayÃ¢â‚¬â€œsphere intersection.
    """
    def __init__(self, center, radius, material=None, object_id=None):
        super().__init__(material=material, object_id=object_id)
        self.center = _as_vector3(center, "center")  # [m] sphere centre in world space
        self.radius = float(radius)                   # [m] sphere radius
        if self.radius <= eps:
            raise ValueError("Sphere radius must be > 0.")

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        RayÃ¢â‚¬â€œsphere intersection via quadratic formula.

        Quadratic setup (direction is unit length, so a = 1):
            oc = origin Ã¢Ë†â€™ centre
            b  = dot(oc, direction)
            c  = dot(oc, oc) Ã¢Ë†â€™ rÃ‚Â²
            discriminant = bÃ‚Â² Ã¢Ë†â€™ c
        Two solutions: t_near = Ã¢Ë†â€™b Ã¢Ë†â€™ Ã¢Ë†Å¡(disc),  t_far = Ã¢Ë†â€™b + Ã¢Ë†Å¡(disc).
        The closest t in [t_min, t_max] is returned.

        :return: RayHit or None
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

        point = origin + t_hit * direction           # world space intersection point
        normal = _normalize(point - self.center)     # outward radial normal
        return RayHit(distance=float(t_hit), point=point, normal=normal, material=self.material, object_id=self.object_id)


class Plane(SceneObject):
    """
    Infinite plane defined by a point and a surface normal.

    Uses the parametric line plane intersection formula.
    """
    def __init__(self, point, normal, material=None, object_id=None):
        super().__init__(material=material, object_id=object_id)
        self.point = _as_vector3(point, "point")                # [m] any point on the plane
        self.normal = _normalize(_as_vector3(normal, "normal"))  # [unit] outward surface normal

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Ray plane intersection.

        Formula:  t = dot(plane_point Ã¢Ë†â€™ origin, normal) / dot(direction, normal)
        If |denom| < eps the ray is parallel to the plane.
        The returned normal is flipped so it always faces the incoming ray.

        :return: RayHit or None
        """
        origin = _as_vector3(origin, "origin")
        direction = _normalize(_as_vector3(direction, "direction"))
        denom = float(np.dot(direction, self.normal))  # dot(ray, normal)
        if abs(denom) < eps:
            return None  # ray is parallel to the plane; no intersection

        t_hit = float(np.dot(self.point - origin, self.normal) / denom)  # parametric distance along ray
        if not (t_min <= t_hit <= t_max):
            return None  # intersection is outside the valid range

        point = origin + t_hit * direction                                # world space hit point
        out_normal = self.normal if denom < 0.0 else -self.normal          # flip normal to face the ray
        return RayHit(distance=t_hit, point=point, normal=out_normal, material=self.material, object_id=self.object_id)


class AxisAlignedBox(SceneObject):
    """
    Axis aligned bounding box (AABB) defined by min and max corner coordinates.

    Uses the slab method: intersect the ray against three pairs of axis perpendicular
    planes and compute the overlap of the resulting intervals.
    """
    def __init__(self, min_corner, max_corner, material=None, object_id=None):
        super().__init__(material=material, object_id=object_id)
        self.min_corner = _as_vector3(min_corner, "min_corner")  # [m] minimum (x,y,z) corner
        self.max_corner = _as_vector3(max_corner, "max_corner")  # [m] maximum (x,y,z) corner
        if np.any(self.max_corner <= self.min_corner):
            raise ValueError("max_corner must be strictly greater than min_corner on all axes.")

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Ray AABB intersection via the slab method.

        For each axis the ray is intersected with the two bounding planes:
            t0 = (min Ã¢Ë†â€™ origin) / direction
            t1 = (max Ã¢Ë†â€™ origin) / direction
        The overall entry/exit distance is t_enter = max(t_small) and t_exit = min(t_big).
        If t_exit < t_enter the ray misses the box.

        The hit face normal is determined by checking which face the intersection
        point is closest to (with a small tolerance for edge cases).

        :return: RayHit or None
        """
        origin = _as_vector3(origin, "origin")
        direction = _normalize(_as_vector3(direction, "direction"))

        # Compute slab intersection distances for each axis.
        # Suppress warnings for rays parallel to a face (direction component Ã¢â€°Ë† 0 Ã¢â€ â€™ inf).
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dir = 1.0 / direction
            t0 = (self.min_corner - origin) * inv_dir  # distances to min face slabs
            t1 = (self.max_corner - origin) * inv_dir  # distances to max face slabs

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

        point = origin + t_hit * direction  # world space hit point

        # Determine which face was hit by proximity to each bounding plane.
        normal = np.zeros(3, dtype=float)
        tol = 1e-8  # [m] tolerance for face identification
        if abs(point[0] - self.min_corner[0]) < tol:
            normal = np.array([-1.0, 0.0, 0.0])   # hit Ã¢Ë†â€™x face
        elif abs(point[0] - self.max_corner[0]) < tol:
            normal = np.array([1.0, 0.0, 0.0])    # hit +x face
        elif abs(point[1] - self.min_corner[1]) < tol:
            normal = np.array([0.0, -1.0, 0.0])   # hit Ã¢Ë†â€™y face
        elif abs(point[1] - self.max_corner[1]) < tol:
            normal = np.array([0.0, 1.0, 0.0])    # hit +y face
        elif abs(point[2] - self.min_corner[2]) < tol:
            normal = np.array([0.0, 0.0, -1.0])   # hit Ã¢Ë†â€™z face
        elif abs(point[2] - self.max_corner[2]) < tol:
            normal = np.array([0.0, 0.0, 1.0])    # hit +z face
        else:
            # Fallback for edge/corner hits: pick the axis whose slab boundary
            # is closest to the computed t_enter value.
            face_idx = int(np.argmax(np.abs(t_small - t_enter)))
            normal = np.zeros(3, dtype=float)
            normal[face_idx] = np.sign(direction[face_idx]) * -1.0  # outward facing normal

        return RayHit(distance=t_hit, point=point, normal=normal, material=self.material, object_id=self.object_id)


class Triangle(SceneObject):
    """
    Single triangle primitive defined by three vertices.

    Uses the MÃƒÂ¶ller Trumbore algorithm for fast ray triangle intersection
    with barycentric coordinate checks.
    """
    def __init__(self, v0, v1, v2, material=None, object_id=None):
        super().__init__(material=material, object_id=object_id)
        self.v0 = _as_vector3(v0, "v0")  # [m] first vertex
        self.v1 = _as_vector3(v1, "v1")  # [m] second vertex
        self.v2 = _as_vector3(v2, "v2")  # [m] third vertex
        # Pre compute face normal for later use in intersection and shading.
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        normal = np.cross(edge1, edge2)  # area weighted normal (cross product of edges)
        if np.linalg.norm(normal) <= eps:
            raise ValueError("Triangle vertices must be non collinear.")
        self.face_normal = _normalize(normal)  # [unit] outward face normal

    def intersect(self, origin, direction, t_min=eps, t_max=float("inf")):
        """
        Ray triangle intersection via MÃƒÂ¶ller Trumbore algorithm.

        Steps:
        1. Compute pvec = cross(direction, edge2), det = dot(edge1, pvec).
           If |det| < eps the ray is parallel to the triangle plane.
        2. Compute barycentric coordinate u; reject if u Ã¢Ë†â€° [0, 1].
        3. Compute barycentric coordinate v; reject if v < 0 or u+v > 1.
        4. Compute parametric distance t; reject if t Ã¢Ë†â€° [t_min, t_max].
        5. Flip normal to face the incoming ray.

        :return: RayHit or None
        """
        origin = _as_vector3(origin, "origin")
        direction = _normalize(_as_vector3(direction, "direction"))

        edge1 = self.v1 - self.v0  # edge from v0 to v1
        edge2 = self.v2 - self.v0  # edge from v0 to v2
        pvec = np.cross(direction, edge2)           # used to compute determinant and u
        det = float(np.dot(edge1, pvec))            # determinant of the 3Ãƒâ€”3 system
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

        point = origin + t_hit * direction                                          # world space hit point
        normal = self.face_normal if np.dot(direction, self.face_normal) < 0.0 else -self.face_normal  # face ray
        return RayHit(distance=t_hit, point=point, normal=normal, material=self.material, object_id=self.object_id)


class Scene:
    """
    Container and closest hit ray casting manager for scene geometry.

    Holds a list of SceneObject instances. cast_ray() tests the ray against
    every object and returns the nearest RayHit (or None).
    """
    def __init__(self, objects=None):
        self.objects = list(objects) if objects is not None else []  # list of SceneObject instances

    def add(self, obj):
        """Add a SceneObject to the scene. Raises TypeError for non SceneObject inputs."""
        if not isinstance(obj, SceneObject):
            raise TypeError("Scene accepts SceneObject instances.")
        self.objects.append(obj)

    def cast_ray(self, origin, direction, max_range=float("inf"), min_range=eps):
        """
        Cast a ray through the scene and return the closest intersection.

        Iterates over all scene objects, progressively narrowing t_max to the
        current closest hit distance so that far objects are culled early.

        :param origin:    ray origin [m]
        :param direction: ray direction (will be normalized) [unit]
        :param max_range: maximum query distance [m]
        :param min_range: minimum query distance [m]

        :return: closest RayHit, or None if the ray hits nothing
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
