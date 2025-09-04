
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-body A4 (no articulation)."""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidPrimView
#from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# --------------------------------------------------------------------------------------
# Configuration: spawn as ONE rigid body (no joints/articulation)
# --------------------------------------------------------------------------------------

A4_RIGID_CFG = RigidObjectCfg(
    prim_path="/home/fom/Documents/ISAAC5/",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/fom/Documents/ISAAC5/AASSEM4D_with_joints.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        # note: no articulation_props here — we’re not using an articulation
        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),    # start 0.5 m above ground
        rot=(0.0, 0.0, 0.0, 1.0),  # identity quaternion (x,y,z,w)
        joint_pos=(0.0, 0.0),
        joint_vel=(0.0, 0.0)
    ),
)
"""Configuration for the A4 as a single rigid body (no articulation)."""

# --------------------------------------------------------------------------------------
# Motor-force “controller”: applies world-space forces at motor loci
# --------------------------------------------------------------------------------------

@dataclass
class QuadMotorForcesParams:
    # Motor positions in the BODY frame (meters)
    # A4 ~ 0.046 m motor-to-motor; half-arm ~ 0.023 m (tweak as needed)
    arm: float = 0.23
    # +1 / -1 spin directions for yaw reaction torque (if you want to add torques)
    spin_dirs: Tuple[int, int, int, int] = (+1, -1, +1, -1)
    # Max thrust per motor (Newtons) at command = 1.0  (tune for your model)
    max_thrust_N: float = 0.5
    # Optional: torque coefficient for yaw reaction (N·m per N of thrust)
    torque_coeff: float = 0.005
    # For computing thrust, use quadratic mapping F = (cmd^2) * max_thrust  (common for BLDC)
    cmd_to_thrust_map = "quadratic_map"

    yaw_force_dist = 0.01


    def motor_offsets_body(self) -> Tuple[Tuple[float, float, float], ...]:
        a = self.arm
        return (
            (+a, +a, 0.0),  # m1
            (-a, +a, 0.0),  # m2
            (-a, -a, 0.0),  # m3
            (+a, -a, 0.0),  # m4
        )




class QuadMotorForcesController:
    """Applies thrust forces at motor loci on a single rigid body.

    Usage (per step, per env):
        ctrl.apply_forces(rigid_view, env_indices, motor_cmds01)
    where motor_cmds01 is a (N_env, 4) tensor/ndarray of normalized commands in [0,1].
    """

    def __init__(self, params: QuadMotorForcesParams = QuadMotorForcesParams()):
        self.p = params
        self._offsets_body = self.p.motor_offsets_body()

    def _cmd_to_thrust(self, u: float) -> float:
        u = max(0.0, min(1.0, float(u)))  # clamp
        if self.p.cmd_to_thrust_map == "quadratic_map":
            return (u * u) * self.p.max_thrust_N
        else:
            return u * self.p.max_thrust_N  # linear mapping

    def apply_forces(
        self,
        rigid_view,
        env_ids,
        motor_cmds01: Sequence[Sequence[float]],
        add_reaction_torque: bool = True,
        ):
        """Apply per-motor forces at world positions.

        Args:
            rigid_view: isaaclab view for the rigid object (e.g., RigidObjectView or RigidPrimView)
            env_ids: indices of the environments to update (list/ndarray/Tensor)
            motor_cmds01: shape (len(env_ids), 4), normalized [0..1]
            add_reaction_torque: if True, apply yaw torques from rotor drag
        """
        # 1) Query world poses for the body (positions and orientations)
        #    Depending on the view class, the API is usually:
        #       world_pos, world_quat = rigid_view.get_world_poses(env_ids)
        world_pos, world_quat = rigid_view.get_world_poses(env_ids)

        # Expect array/tensor shapes: (N, 3) and (N, 4)
        # Convert quat to rotation matrices; Isaac Lab usually offers a helper in math utils.
        # If not, implement a small quat->R; here we assume a helper exists:
        from isaaclab.utils.math import quat_to_matrix  # typical location; adjust if needed
        R = quat_to_matrix(world_quat)  # (N, 3, 3)

        N = len(env_ids)
        # Precompute world-space application points and forces
        world_points = []
        world_forces = []
        world_torques = [] if add_reaction_torque else None

        for i in range(N):
            Ri = R[i]              # (3,3)
            pi = world_pos[i]      # (3,)

            cmds = motor_cmds01[i]
            # body z-axis (thrust direction) in world frame = Ri @ [0,0,1]
            thrust_dir_world = Ri @ (0.0, 0.0, 1.0)

            for m in range(4):
                # body-frame motor offset -> world position
                r_b = self._offsets_body[m]
                r_w = Ri @ r_b  # rotate offset into world
                p_app = (pi[0] + r_w[0], pi[1] + r_w[1], pi[2] + r_w[2])
                # thrust magnitude
                Fm = self._cmd_to_thrust(cmds[m])
                F_vec = (thrust_dir_world[0] * Fm,
                         thrust_dir_world[1] * Fm,
                         thrust_dir_world[2] * Fm)

                world_points.append(p_app)
                world_forces.append(F_vec)

                if add_reaction_torque:
                    # Yaw reaction torque around body z (world z after rotation)
                    tz = self.p.torque_coeff * Fm * self.p.spin_dirs[m]
                    
                    # Pick any unit vector perpendicular to thrust_dir_world
                    if np.allclose(thrust_dir_world, [0,0,1]):
                        perp = np.array([1.0, 0.0, 0.0])
                    else:
                        perp = np.cross(thrust_dir_world, [0,0,1])
                        perp /= np.linalg.norm(perp)

                    # Force magnitude to produce torque: |F| * d = tau
                    F_mag = tz / self.p.yaw_force_dist

                    # Two points along perpendicular direction
                    p1 = p_app + 0.5 * self.p.yaw_force_dist * perp
                    p2 = p_app - 0.5 * self.p.yaw_force_dist * perp

                    F1 = F_mag * perp
                    F2 = -F1

                    world_points.extend([p1, p2])
                    world_forces.extend([F1, F2])

        # 2) Apply all forces (and optional torques) at positions
        # Isaac Lab views generally have one of the following:
        #   rigid_view.apply_forces_at_positions(forces, positions, env_ids, is_global=True)
        #   rigid_view.apply_forces(forces, env_ids, is_global=True)  # at COM
        # Choose the “at positions” variant to create the proper moments automatically.
        rigid_view.apply_forces_at_positions(
            forces=world_forces,
            positions=world_points,
            env_ids=env_ids,
            is_global=True,
        )

        