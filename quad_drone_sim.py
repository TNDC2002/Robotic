"""
Minimal quad-style thrust on a flat body in PyBullet.

Physics uses a simple box collision shape; rendering uses the Blender
OBJ mesh in ``drone_blender_model/``. Four forces are applied at the top
corners of the box, each along the body +Z axis (like rotors). A simple
PD loop holds altitude and damps roll/pitch so the drone can hover.

Run (after activating .venv):
  python quad_drone_sim.py
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile

import pybullet as p
import pybullet_data
import time


def _urdf_path_ok_for_loader(path: str) -> bool:
    """PyBullet's URDF loader often fails on Windows when the path is not ASCII."""
    try:
        path.encode("ascii")
    except UnicodeEncodeError:
        return False
    return os.path.isfile(path)


def resolve_drone_mesh_path() -> str:
    """Return an ASCII-safe path to drone.obj for PyBullet's mesh loader."""
    here = os.path.dirname(os.path.abspath(__file__))
    obj = os.path.normpath(os.path.join(here, "drone_blender_model", "drone.obj"))
    mtl = os.path.normpath(os.path.join(here, "drone_blender_model", "drone.mtl"))
    if not os.path.isfile(obj):
        raise FileNotFoundError(f"Drone mesh not found: {obj}")
    if _urdf_path_ok_for_loader(obj):
        return obj

    cache_dir = os.path.join(tempfile.gettempdir(), "robotic_drone_mesh")
    os.makedirs(cache_dir, exist_ok=True)
    cached_obj = os.path.join(cache_dir, "drone.obj")
    cached_mtl = os.path.join(cache_dir, "drone.mtl")
    shutil.copy2(obj, cached_obj)
    if os.path.isfile(mtl):
        shutil.copy2(mtl, cached_mtl)
    return cached_obj


def create_drone_visual_shape() -> int:
    mesh_path = resolve_drone_mesh_path()
    mesh_dir = os.path.dirname(mesh_path)
    if mesh_dir:
        p.setAdditionalSearchPath(mesh_dir)
    mesh_scale = (MESH_SCALE, MESH_SCALE, MESH_SCALE)
    vis_orn = p.getQuaternionFromEuler(MESH_VIS_EULER)
    return p.createVisualShape(
        p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=mesh_scale,
        visualFrameOrientation=vis_orn,
    )


def load_ground_plane() -> None:
    """Ground: prefer plane.urdf from pybullet_data; fall back to infinite plane.

    PyBullet's URDF importer commonly fails on non-ASCII paths (e.g. OneDrive
    folders with accented names), even when the file exists. In that case we
    use a static GEOM_PLANE (no checker texture, same physics).
    """
    data_dir = pybullet_data.getDataPath()
    if data_dir:
        p.setAdditionalSearchPath(data_dir)
        plane_path = os.path.normpath(os.path.join(data_dir, "plane.urdf"))
        if _urdf_path_ok_for_loader(plane_path):
            p.loadURDF(plane_path)
            return
    plane_col = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, plane_col)


# Flat short rectangle (half-extents in meters): long in X, narrow in Y, thin in Z
HALF_EXTENTS = (0.22, 0.10, 0.02)
MASS = 0.45
HOVER_Z = 0.85
DT = 1.0 / 240.0
# Wall-clock pause per physics frame in GUI mode (0 = run as fast as possible).
GUI_STEP_SLEEP_S = 0.01
REALTIME_STEP_SLEEP_S = DT

# Blender OBJ export (meters after scale): ~0.44 x 0.53 x 0.08 with +90 deg about X
MESH_SCALE = 0.07
MESH_VIS_EULER = (math.pi / 2.0, 0.0, 0.0)


def motor_corners_local(hx: float, hy: float, hz: float) -> list[tuple[float, float, float]]:
    """Four rotor attachment points on the top face (body frame, origin at COM)."""
    return [
        (+hx, +hy, hz),
        (+hx, -hy, hz),
        (-hx, +hy, hz),
        (-hx, -hy, hz),
    ]


def world_from_body(pos: tuple[float, float, float], orn: tuple[float, float, float, float], local: tuple[float, float, float]):
    """Transform body-frame point to world frame."""
    wp, _ = p.multiplyTransforms(pos, orn, local, (0.0, 0.0, 0.0, 1.0))
    return wp


def body_z_in_world(orn: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Unit thrust direction (body +Z) expressed in world coordinates."""
    tip = world_from_body((0.0, 0.0, 0.0), orn, (0.0, 0.0, 1.0))
    length = math.sqrt(tip[0] ** 2 + tip[1] ** 2 + tip[2] ** 2) or 1.0
    return (tip[0] / length, tip[1] / length, tip[2] / length)


def scale_vec(v: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def allocate_motor_thrusts(
    thrust_sum: float,
    mix_r: float,
    mix_p: float,
    signs_xy: list[tuple[float, float]],
    max_motor: float,
) -> list[float]:
    """Four motor thrusts (N) whose *sum* matches thrust_sum before saturation.

    Roll/pitch mixing uses coefficients that sum to zero per axis, so
    base_each + mix_r*sy + mix_p*sx sums to thrust_sum. Clipping each motor
    to a cap breaks that sum and can leave total thrust below weight; we
    re-scale upward (within max_motor) so the requested total is recovered.
    """
    base_each = thrust_sum / 4.0
    raw = [base_each + mix_r * sy + mix_p * sx for sx, sy in signs_xy]

    for _ in range(24):
        clamped = [min(max_motor, max(0.0, t)) for t in raw]
        s = sum(clamped)
        if thrust_sum <= 1e-6:
            return clamped
        if abs(s - thrust_sum) <= 1e-3 * max(1.0, thrust_sum):
            return clamped
        if s <= 1e-9:
            return clamped
        raw = [t * (thrust_sum / s) for t in clamped]

    return [min(max_motor, max(0.0, t)) for t in raw]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Quad drone hover sim (PyBullet GUI).")
    parser.add_argument(
        "--sleep",
        type=float,
        default=GUI_STEP_SLEEP_S,
        metavar="SEC",
        help=f"Seconds to pause each frame for visibility (default {GUI_STEP_SLEEP_S}; 0 = max speed)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help=f"Match sim clock (~{1.0 / DT:.0f} Hz); overrides --sleep",
    )
    args = parser.parse_args()
    step_sleep_s = REALTIME_STEP_SLEEP_S if args.realtime else max(0.0, args.sleep)

    cid = p.connect(p.GUI)
    p.setGravity(0.0, 0.0, -9.81)
    load_ground_plane()
    p.setTimeStep(DT)

    hx, hy, hz = HALF_EXTENTS
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=HALF_EXTENTS)
    vis = create_drone_visual_shape()
    # Start level so body +Z is world +Z (rotor normal up); avoid landing flat then losing vertical thrust.
    start = (0.0, 0.0, 0.55)
    start_orn = (0.0, 0.0, 0.0, 1.0)
    drone = p.createMultiBody(
        baseMass=MASS,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=start,
        baseOrientation=start_orn,
    )
    p.changeDynamics(drone, -1, linearDamping=0.05, angularDamping=0.08)
    p.changeDynamics(drone, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)

    corners_body = motor_corners_local(hx, hy, hz)
    # Corner layout: 0=+X+Y, 1=+X-Y, 2=-X+Y, 3=-X-Y (for mixing roll/pitch)
    signs_xy = [(+1.0, +1.0), (+1.0, -1.0), (-1.0, +1.0), (-1.0, -1.0)]

    g = 9.81
    # PD gains (tuned for this mass/size)
    kp_z, kd_z = 18.0, 4.5
    kp_rp, kd_rp = 0.45, 0.08
    # Per-motor ceiling: must exceed hover share (mg/4) plus room for attitude mixing.
    max_motor = MASS * g * 0.65

    print("Quad drone sim — hover hold at target Z. Keys: W/S target height, Q/E roll trim, A/D pitch trim, ESC quit.")
    target_z = HOVER_Z
    roll_trim = pitch_trim = 0.0

    while p.isConnected(cid):
        keys = p.getKeyboardEvents()
        if ord("w") in keys and keys[ord("w")] & p.KEY_IS_DOWN:
            target_z += 0.25 * DT
        if ord("s") in keys and keys[ord("s")] & p.KEY_IS_DOWN:
            target_z -= 0.25 * DT
        if ord("q") in keys and keys[ord("q")] & p.KEY_IS_DOWN:
            roll_trim += 0.4 * DT
        if ord("e") in keys and keys[ord("e")] & p.KEY_IS_DOWN:
            roll_trim -= 0.4 * DT
        if ord("a") in keys and keys[ord("a")] & p.KEY_IS_DOWN:
            pitch_trim += 0.4 * DT
        if ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:
            pitch_trim -= 0.4 * DT
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            break

        pos, orn = p.getBasePositionAndOrientation(drone)
        lin_vel, ang_vel = p.getBaseVelocity(drone)
        roll, pitch, _yaw = p.getEulerFromQuaternion(orn)

        roll_e = roll - roll_trim
        pitch_e = pitch - pitch_trim
        z_e = target_z - pos[2]
        vz_e = lin_vel[2]

        # Total thrust command (N); mixer preserves sum before saturation, see allocate_motor_thrusts.
        thrust_sum = MASS * g + kp_z * z_e - kd_z * vz_e
        thrust_sum = max(0.0, thrust_sum)

        # Roll: left/right (Y). Pitch: front/back (X) — opposite sign vs roll (see env).
        mix_r = -kp_rp * roll_e - kd_rp * ang_vel[0]
        mix_p = +kp_rp * pitch_e + kd_rp * ang_vel[1]

        thrusts = allocate_motor_thrusts(thrust_sum, mix_r, mix_p, signs_xy, max_motor)


        # for testing - dont remove this part
        thrusts = [1.10405, 1.10405, 1.104, 1.104]
        print(thrusts)


        thrust_dir = body_z_in_world(orn)
        for corner_body, t in zip(corners_body, thrusts):
            world_point = world_from_body(pos, orn, corner_body)
            force = scale_vec(thrust_dir, t)
            p.applyExternalForce(drone, -1, force, world_point, p.WORLD_FRAME)

        if step_sleep_s > 0.0:
            time.sleep(step_sleep_s)
        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
