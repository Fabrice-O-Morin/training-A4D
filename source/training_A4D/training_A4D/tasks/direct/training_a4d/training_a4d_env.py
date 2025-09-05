# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import gymnasium as gym
import torch
import omni
from pxr import UsdPhysics

from collections.abc import Sequence

import isaaclab.sim as sim_utils
#from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
#from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
#from isaaclab.scene import InteractiveSceneCfg
#from isaaclab.sim import SimulationCfg
#from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

#from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
#from isaaclab.assets import RigidObjectView
#from isaaclab.envs.mdp.terrains import TerrainImporterCfg
#from isaaclab.terrains import TerrainImporterCfg
from .training_a4d_env_cfg import TrainingA4dEnvCfg


class TrainingA4dEnv(DirectRLEnv):
    #cfg: TrainingA4dEnvCfg

    def __init__(self, cfg: TrainingA4dEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self.num_envs = cfg.scene.num_envs   is set AUTOMATICALLY, do NOT set it manually

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        self.control = cfg.control 

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        #self.set_debug_vis(self.cfg.debug_vis)



    def _setup_scene(self):  # This is called BEFORE CLONING
        
        # scene is automatically added. The rest needs manually adding. PROBABLY FALSE

        # Add A4
        #self.A4 = self.scene.add(self.cfg.A4_RIGID_CFG)
        

        



        #joints = []
        #joint_api = UsdPhysics.Joint(joint)
        #for prim in self.stage.Traverse():
        #    if prim.IsA(UsdPhysics.Joint):
        #        x = str(prim.GetPath())
        #        if x.startswith(A4_path): #TO-DO: set the proper path
        #            joints.append(prim) 
        #            if  x == self.cfg.holder_dof_name: # a RigidPrimView would be more convenient for more than two joints and three components
        #                self._holder_dof_idx = prim  
        #            elif x == self.cfg.drone_dof_name: 
        #                self._drone_dof_idx =  prim
        #print("Joints:", [j.GetPath() for j in joints])

        #for joint in joints:
        #    body0 = joint_api.GetBody0Rel().GetTargets()
        #    body1 = joint_api.GetBody1Rel().GetTargets()
        #    print(f"Joint {joint.GetPath()} connects {body0} <-> {body1}")


        # add ground plane (if it does not exist in the scene)
        #spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Add terrain
        #self.terrain = self.scene.add(self.cfg.terrain)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation to scene
        # self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)



    def get_all_descendants(self,prim):
        descendants = []
        for child in prim.GetChildren():
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child))
        return descendants



    def post_reset(self):
        super().post_reset()

        self.stage = omni.usd.get_context().get_stage()

        A4_path = "World/envs/env_0/A4" #self.cfg.A4_RIGID_CFG.prim_path #"{ENV_REGEX_NS}/A4"
        rigid_bodies = []    # List of rigid bodies in agent A4
        total_mass = 0.0
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if str(prim.GetPath()).startswith(A4_path): 
                    rigid_bodies.append(prim)
                    mass_api = UsdPhysics.MassAPI(prim)
                    mass_attr = mass_api.GetMassAttr()
                    if mass_attr.HasAuthoredValue(): total_mass += mass_attr.Get()
        print("Rigid bodies:", [rb.GetPath() for rb in rigid_bodies])
        self._total_mass = total_mass
        print(f"self._total_mass = ", total_mass)

        self._total_weight = self._total_mass * self._gravity_magnitude

        self.drone_bodies = [prim for prim in self.stage.Traverse() if prim.GetName() == "tn__MODELSimpleDrone11_sQI"] # List of drone bodies in agent A4
        print("Drone bodies:", [rb.GetPath() for rb in self.drone_bodies])

        for env_id in range(self.num_envs):
            prim_path = f"/World/envs/env_{env_id}/A4"
            prim = stage.GetPrimAtPath(prim_path)

            # Apply PhysX API
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb_api.CreateCollisionGroupAttr().Set(env_id)
            rb_api.CreateCollisionMaskAttr().Set(1 << env_id)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()


    def _apply_action(self) -> None:   
        #self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)
        self.control.apply_forces( 
            self.drone_bodies,    # ?
            motor_cmds01, # Sequence[Sequence[float]]           # TO-DO
            add_reaction_torque = True
            )
        


    def _get_observations(self) -> dict: # TO-DO
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    


    def _get_rewards(self) -> torch.Tensor: # TO-DO
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # TO-DO
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out



    def _reset_idx(self, env_ids: Sequence[int] | None): # TO-DO
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        

    def _set_debug_vis_impl(self, debug_vis: bool): 
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)



@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward