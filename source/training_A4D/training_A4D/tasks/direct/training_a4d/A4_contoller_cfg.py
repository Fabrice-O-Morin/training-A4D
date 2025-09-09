
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import torch
import math
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg #, RigidPrimView
#from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import omni
from pxr import UsdGeom, PhysxSchema, UsdPhysics
from scipy.spatial.transform import Rotation



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




class A4ForcesController:
    """Applies thrust forces at motor loci on a single rigid body.

    Usage (per step, per env):
        ctrl.apply_forces(rigid_view, env_indices, motor_cmds01)
    where motor_cmds01 is a (N_env, 4) tensor/ndarray of normalized commands in [0,1].
    """

    def __init__(self, params: QuadMotorForcesParams = QuadMotorForcesParams()):
        self.p = params
        self._offsets_body = self.p.motor_offsets_body()


    def _cmd_to_thrust(self, x):
        y = x.float() 
        z  = torch.clamp(y, min=0, max=1.0)   # clamp between 0 and 1
        if self.p.cmd_to_thrust_map == "quadratic_map":
            return (z ** 2) * self.p.max_thrust_N  # quadratic model of the thrust value as a function of input
        else:
            return z * self.p.max_thrust_N  # linear mapping


    def apply_forces(
        self,
        drones,
        body_index,
        env_ids,
        scene,
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

        # Query world poses for the body (positions and orientations)
        xform = [UsdGeom.Xformable(p) for p in drones]
        world_transform = [xformi.ComputeLocalToWorldTransform(0.0) for xformi in xform]
        world_pos  = [wt.ExtractTranslation() for wt in world_transform]
        world_quat_temp = [wt.ExtractRotationQuat() for wt in world_transform]
        world_quat = [np.array([w.GetReal(), *w.GetImaginary()]) for w in world_quat_temp]
        
        R = Rotation.from_quat(world_quat, scalar_first=True)  # (N, 3, 3)

        # Precompute world-space application points and forces
        world_points = []
        world_forces = []
        world_torques = [] 


        art = scene.articulations["Agent"]


        # Some verifications if need be
        #scrutinize_object(art, "scene.articulations")
        #scrutinize_object(art, "world_pos")
        #print(f"len(world_pos)", len(world_pos))
        #print(f"type(world_pos)", type(world_pos), "\n")
        #print(f"world_pos\n",world_pos, "\n\n")


        cmds = motor_cmds01  
        # Verification of motor commands (actions from policy)
        #print(f"cmds type = {type(cmds)}")  # Tensor num_envs,4
        #print(f"cmds =\n{cmds}")
        #print(f"\ncmds shape = {cmds.shape}")  # Tensor num_envs,4


        # body z-axis (thrust direction) in world frame = Ri @ [0,0,1]
        thrust_dir_world = R.apply([0.0, 0.0, 1.0])                    # should be shape (num_envs,3)
        # Verification
        #print(f"\nthrust_dir_world = \n", thrust_dir_world)
        #print(f"\nthrust_dir_world.shape = ", thrust_dir_world.shape)  


        for m in range(4):
            # body-frame motor offset -> world position
            r_b = np.array(self._offsets_body[m])                      # should be shape (3,)
            r_w = R.apply(r_b)  # rotate offset into world       -       should be shape (num_envs,3)
            p = np.array([[v[0], v[1], v[2]] for v in world_pos])      # should be shape (num_envs,3)
            p_app = p + r_w
            
            # Verification if necessary
            #print(f"\nr_b.shape = {r_b.shape}")
            #print(f"r_b = self._offsets_body[m] = {self._offsets_body[m]}\n")  #
            #print(f"r_w.shape = {r_w.shape}")
            #print(f"r_w = {r_w}\n")  #
            #print(f"p_app =\n", p_app, "\n")

            
            # thrust magnitude
            Fm = self._cmd_to_thrust(cmds)                            # should be shape (num_envs,4)
            thrust_dir_world_batched = torch.tensor(thrust_dir_world, device = "cuda:0").unsqueeze(1).repeat(1, 4, 1).clone()
                                                                      # should be shape (num_envs,4,3)
            F_vec = thrust_dir_world_batched * Fm.unsqueeze(-1)       # should be shape (num_envs,4,3)


            # Verification
            #print(f"\ncmds.shape = {cmds.shape}")                    # should be (num_envs, 4)
            #print(f"cmds =\n", cmds, "\n")
            #print(f"\nthrust_dir_world_batched.shape = {thrust_dir_world_batched.shape}\n") 
            #print(f"\nthrust_dir_world_batched =\n{thrust_dir_world_batched}\n")
                                                                      # should be shape (num_envs,4)
            #print(f"\nFm.shape = {Fm.shape}")                         # should be shape (num_envs,4)
            #print(f"Fm =\n", Fm, "\n")
            #print(f"\nF_vec.shape = {F_vec.shape}")                   # should be shape (num_envs,4,3)
            #print(f"F_vec =\n", F_vec, "\n")




            # Determine torque
            #torque = np.cross(p_app - local_com, F_vec)
            #torque = np.cross(p_app, F_vec)   # wrong formula, only needed to see if subsequent code runs   
            
            world_points.append(p_app)
            world_forces.append(F_vec)
            

            if add_reaction_torque:
                # Reaction torque around body z (world z after rotation) from rotation of the rotor. No yaw without it.
                tz = self.p.torque_coeff * F_vec * self.p.spin_dirs[m]
                
                # Pick any unit vector perpendicular to thrust_dir_world
                if np.allclose(thrust_dir_world, [0,0,1]):
                    perp = np.array([1.0, 0.0, 0.0])
                else:
                    perp = np.cross(thrust_dir_world, [0,0,1])
                    perp /= np.linalg.norm(perp)

                # Force magnitude to produce torque: |F| * d = torque
                F_mag = tz / self.p.yaw_force_dist

                # Two points along perpendicular direction
                p1 = p_app + 0.5 * self.p.yaw_force_dist * perp
                p2 = p_app - 0.5 * self.p.yaw_force_dist * perp

                F1 = F_mag * perp
                F2 = -F1

                # Apply force at position
                #rb_api.ApplyForceAtPos().Set([(F1, p1), (F2, p2)])
                
                world_points.extend([p1, p2])
                world_forces.extend([F1, F2])


        # Apply total force and torque
        art.set_external_force_and_torque(
                                           world_forces,
                                           torch.zeros(16, 3),
                                           [body_index],
                                           env_ids
                                         )




def scrutinize_object(obj, name: str):
    print(f"\n\ndir(object) for {name}\n", dir(obj))
    print(f"\n\nobject.__dict__ for {name})\n", dir(obj),"\n\n")
        

        