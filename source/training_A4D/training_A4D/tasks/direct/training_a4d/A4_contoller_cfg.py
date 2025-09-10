
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
from .utils import get_centers_of_mass

#import omni.physx
from pxr import UsdGeom, PhysxSchema, UsdPhysics, Usd, Gf
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


    def _cmd_to_thrust(self, x): #-> torch.tensor 
        y = x.float() 
        z  = torch.clamp(y, min=0, max=1.0)   # clamp between 0 and 1
        if self.p.cmd_to_thrust_map == "quadratic_map":
            return (z ** 2) * self.p.max_thrust_N  # quadratic model of the thrust value as a function of input
        else:
            return z * self.p.max_thrust_N  # linear mapping


    def apply_forces(
        self,
        drones,
        env_ids,
        body_index,
        nba,
        art,
        blk,
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
            #print(f"r.shape is ", (torch.tensor(p_app)).shape)

            
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


            world_points.append(torch.tensor(p_app))  # List of numpy arrays
            world_forces.append(F_vec)                # List of torch tensors
            

            # To compute the torque, we need the center of mass of the object that combines the three rigid bodies:
            # For that, we need:           
            #            the center of mass of each rigid body
            #            the mass ratios of these bodies to their sum (to apply a weighted sum to the individual centers of mass)
            # Once these are known, a utility function does the job to compute the relevant COM
            
            get_centers_of_mass(blk)

            COM_list = []
            for prim_list in blk:
                masses = np.array([])
                block_mass = np.array(0.0)
                centers_of_mass = []
                for j in range(3): 
                    prim =  prim_list[j]
                    mass_prim = prim.GetAttribute('physics:mass').Get()# physx_iface.get_rigid_body_mass(prim_path)
                    a,b,c = prim.GetAttribute('physics:centerOfMass').Get() # physx_iface.get_rigid_body_center_of_mass(prim_path) # returns tuple of 3 floats
                    com_world = [a,b,c]

                    # Verification
                    #print(f"mass of prim = ", mass_prim)
                    #print(type(prim.GetAttribute('physics:centerOfMass').Get()))
                    #print(type(a), type(b), type(c))
                    #print(f"COM vec3f is \n",prim.GetAttribute('physics:centerOfMass').Get())
                    #print(f"com_world is \n", com_world, "\n")

                    masses = np.append(masses, mass_prim)
                    block_mass += mass_prim
                    centers_of_mass.append(com_world)         

                    #xform = UsdGeom.Xformable(prim)
                    #world_transform = xform.ComputeLocalToWorldTransform(0) # or (Usd.TimeCode.Default())
                    #local_transform = world_transform.GetInverse() 

                    #com_local_h = Gf.Vec4d(com_world[0], com_world[1], com_world[2], 1.0) * local_transform
                    #com_local = Gf.Vec3d(com_local_h[0], com_local_h[1], com_local_h[2])

                mass_prim = np.array(masses)
                centers_of_mass = np.array(centers_of_mass)
                prim_COM = np.average(centers_of_mass, axis=0, weights=mass_prim) #np.dot(mass_prim, centers_of_mass)   
                prim_COM /= block_mass
                COM_list.append(torch.tensor(prim_COM))

                #print("torch.tensor(prim_COM) ", torch.tensor(prim_COM))

            COMs = torch.stack(COM_list, dim=0).to("cuda:0")  # shape (n,3)
            # Verification
            #print(f"\ntensorized COMs.shape is ", COMs.shape)
            #print(f"\ntensorized COMs.device is ", COMs.device)

            # Now compute torque
            r = (torch.tensor(p_app, dtype=torch.float32, device="cuda:0") - COMs).unsqueeze(1).expand(-1, 4, -1)  # should be (n, 4, 3)
            torques = torch.cross(r, F_vec, dim=-1)
            world_torques.append(torques)
            

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

        #print(f"test of torque = 0\n", world_torques[0])
       
        ext_F       =  torch.stack(world_forces, dim=1).to("cuda:0").reshape(16, 16, 3)  # shape (n,m,3)
        sum_ext_F       = torch.sum(ext_F, dim=1)
        ext_torques = torch.stack(world_torques, dim=1).to("cuda:0").reshape(16, 16, 3)  # shape (n,m,3)
        sum_ext_torques = torch.sum(ext_torques, dim=1)
        

        # Format in the way articulation method expects
        artF = torch.zeros((len(env_ids), nba, 3), dtype=torch.float32, device="cuda:0")
        artF[:, m-1, :] = sum_ext_F.float()
        artT = torch.zeros((len(env_ids), nba, 3), dtype=torch.float32, device="cuda:0")
        artT[:, m-1, :] = sum_ext_torques.float()
        artnba=torch.arange(nba, device="cuda:0")
        
        # Verification
        #print(f"ext_torques.shape = ", ext_torques.shape)
        #print(f"ext_F.shape = ",ext_F.shape)
        #print(f"sum_Fext.shape = ", sum_ext_F.shape)
        #print(f"sum_ext_torques.shape = ", sum_ext_torques.shape)
        #print(f"env_ids = ", env_ids.shape)
        #print(f"artF.shape = ", artF.shape, "type = ", artF.dtype, " and device = ", artF.device)
        #print(f"artT.shape = ", artT.shape, "type = ", artT.dtype, " and device = ", artT.device)
        #print(f"env_ids.shape = ", env_ids.shape, "type = ", env_ids.dtype, " and device = ", env_ids.device)
        #print(f"artnba.shape = ", artnba.shape, "type = ", artnba.dtype, " and device = ", artnba.device)

        # Apply total force and torque
        art.set_external_force_and_torque(
                                           forces = artF,
                                           torques = artT,
                                           body_ids = artnba,#None,#range(nba),
                                           env_ids = env_ids
                                         )




def scrutinize_object(obj, name: str):
    print(f"\n\ndir(object) for {name}\n", dir(obj))
    print(f"\n\nobject.__dict__ for {name})\n", dir(obj),"\n\n")
        

        