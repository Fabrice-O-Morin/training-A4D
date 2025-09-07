# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import gymnasium as gym
import torch
import omni
from pxr import Usd, UsdPhysics, PhysxSchema

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
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
    cfg: TrainingA4dEnvCfg

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
        self._total_weight = self._total_mass * self._gravity_magnitude

        self.control = cfg.control 

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        #self.set_debug_vis(self.cfg.debug_vis)



    def _setup_scene(self):  # This is called BEFORE CLONING
        """Setup the scene for the environment.

        This function is responsible for creating the scene objects and setting up the scene for the environment.
        The scene creation can happen through :class:`isaaclab.scene.InteractiveSceneCfg` or through
        directly creating the scene objects and registering them with the scene manager.

        The implementation of this function is left to the derived classes. If the environment does not require
        any explicit scene setup, the function can be left empty.

        NOTE: This method is called in the middle of __init__ witzh the following two lines:
            self.scene = InteractiveScene(self.cfg.scene)
            self._setup_scene()
        """

        # add ground plane (if it does not exist in the scene)
        #spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Add terrain
        #self.terrain = self.scene.add(self.cfg.terrain)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        #self.scene.clone_environments()#,copy_from_source=False)
        #print("Namespace after cloning:", self.scene.env_ns)  # should be like /World/envs/env_0
        
        self.stage = omni.usd.get_context().get_stage()
        
        # Add articulation to scene 
        self._agent = Articulation(cfg=self.cfg.A4_RIGID_CFG)
        self.scene.articulations["Agent"] = self._agent
        print(f"\n\nScene attribute articulations = {self.scene.articulations}")
        #self.print_hierarchy(self.stage.GetPseudoRoot())
        

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])


        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        A4_path = "/World/envs/env_0/A4" #self.cfg.A4_RIGID_CFG.prim_path #"{ENV_REGEX_NS}/A4"
        rigid_bodies = []    # List of rigid bodies in agent A4
        total_mass = 0.0
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith(A4_path): 
                #print(f"prim_path = {str(prim.GetPath())}\nStarts with A4? {str(prim.GetPath()).startswith(A4_path)}")
                #print(f"\n{prim}   has applied schemas {prim.GetAppliedSchemas()}")
                #print(f"prim.HasAPI(UsdPhysics.RigidBodyAPI) is {prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_bodies.append(prim)
                if prim.HasAPI(UsdPhysics.MassAPI):
                    mass_api = UsdPhysics.MassAPI(prim)
                    mass_attr = mass_api.GetMassAttr()
                    if mass_attr.HasAuthoredValue(): total_mass += mass_attr.Get()                    
        print(  f"\nRigid bodies:\n"  +  "".join([f"{rb.GetPath()}\n" for rb in rigid_bodies])  )
        self._total_mass = total_mass
        print(f"\nself._total_mass = ", total_mass)

        # collect the prims where forces will be applied
        self.drone_bodies = [prim for prim in self.stage.Traverse() if "tn__MODELSimpleDrone11_sQI" in str(prim.GetName())] # List of drone bodies in agent A4
        print("\n\nDrone bodies:", [rb.GetPath() for rb in self.drone_bodies], f"\n \n \n")
        # TO-DO: move this after the cloning, otherwise useless





    #def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Resets all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        #super.reset()



    def print_stage_traverse(self):
        print(f"\n\n\nSTAGE TRAVERSE\n")
        for prim in self.stage.Traverse():
            print("Prim:", prim.GetPath(), "Type:", prim.GetTypeName())
        print(f"END STAGE TRAVERSE\n \n \n")


    
    def get_all_descendants(self,prim):
        descendants = []
        for child in prim.GetChildren():
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child))
        return descendants



    def print_hierarchy(self, prim, indent=0):
        print("  " * indent + prim.GetName())
        for child in prim.GetChildren():
            self.print_hierarchy(child, indent + 1)
    


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
        """
        self.actions = actions.clone()
        #self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._agent_weight * (self._actions[:, 0] + 1.0) / 2.0
        #self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]


    def _apply_action(self) -> None:   
        """Apply actions to the simulator.
        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """
        #self._agent.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        self.control.apply_forces( 
            self.drone_bodies,    # ?
            motor_cmds01, # Sequence[Sequence[float]]                   # TO-DO
            add_reaction_torque = True
            )
        


    def _get_observations(self) -> dict:     #{"policy": VecEnvObs}           # TO-DO
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment.
        """
        desired_pos_b, _ = subtract_frame_transforms(
            self._agent.data.root_pos_w, self._agent.data.root_quat_w, self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._agent.data.root_lin_vel_b,
                self._agent.data.root_ang_vel_b,
                self._agent.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    


    def _get_rewards(self) -> torch.Tensor:                             # TO-DO
        """Compute and return the rewards for the environment.
        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        lin_vel = torch.sum(torch.square(self._agent.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._agent.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._agent.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._agent.data.root_pos_w[:, 2] < 0.1, self._agent.data.root_pos_w[:, 2] > 2.0)
        return died, time_out



    #def _reset_idx(self, env_ids: Sequence[int] | None): # TO-DO
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments based on specified indices.
        Args:
            env_ids: List of environment ids which must be reset
        """
        print(f"\n\n\nRESETTING")
        #self.print_hierarchy(self.stage.GetPseudoRoot())

        # Set collision filtering
        #for env_id in env_ids:
        #    prim_path = f"/World/envs/env_{env_id}/A4"
        #    prim = self.stage.GetPrimAtPath(prim_path)
        #    # Apply PhysX API
        #    rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        #    rb_api.CreateCollisionGroupAttr().Set(env_id)
        #    rb_api.CreateCollisionMaskAttr().Set(1 << env_id)


        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._agent._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._agent.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._agent.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._agent.data.default_joint_pos[env_ids]
        joint_vel = self._agent.data.default_joint_vel[env_ids]
        default_root_state = self._agent.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._agent.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._agent.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._agent.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        
## DEBUG_VIS section
#
#    def _set_debug_vis_impl(self, debug_vis: bool): 
#        # create markers if necessary for the first time
#        if debug_vis:
#            if not hasattr(self, "goal_pos_visualizer"):
#                marker_cfg = CUBOID_MARKER_CFG.copy()
#                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
#                # -- goal pose
#                marker_cfg.prim_path = "/Visuals/Command/goal_position"
#                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
#            # set their visibility to true
#            self.goal_pos_visualizer.set_visibility(True)
#        else:
#            if hasattr(self, "goal_pos_visualizer"):
#                self.goal_pos_visualizer.set_visibility(False)
#
#    def _debug_vis_callback(self, event):
#        # update the markers
#        self.goal_pos_visualizer.visualize(self._desired_pos_w)




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