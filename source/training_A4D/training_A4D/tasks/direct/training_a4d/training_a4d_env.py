# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import gymnasium as gym
import torch

import omni
import omni.isaac.dynamic_control as dc
from pxr import  UsdPhysics, UsdGeom, Usd, Gf #, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
#from isaaclab.markers import VisualizationMarkers
#from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms #, sample_uniform
#from isaaclab.assets import RigidObjectView


from .training_a4d_env_cfg import TrainingA4dEnvCfg

from .utils import print_hierarchy
#import training_a4d.tasks.direct.training_a4d.utils


DEBUG = False

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

            self.number_of_bodies_in_articulation = 3

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
            if DEBUG: print(f"\n\nENTERING _SETUP_SCENE\n\n")

            self.stage = omni.usd.get_context().get_stage()

            # add ground plane (if it does not exist in the scene)
            #spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

            # Add terrain
            #self.terrain = self.scene.add(self.cfg.terrain)
            #self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            #self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            #self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            
            # Add articulation to scene if it is to be cloned
            self.agent = Articulation(cfg=self.cfg.A4_RIGID_CFG)
            self.scene.articulations["Agent"] = self.agent
            
            # clone and replicate
            self.scene.clone_environments(copy_from_source=False)


            # we need to explicitly filter collisions for CPU simulation
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=[])


            # add lights
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)


            # List of rigid bodies in one agent A4 in env_0
            A4_path = "/World/envs/env_0/A4/ASSEM4D_with_joints" 
            self.rigid_bodies = []    # List of rigid bodies in agent A4
            total_mass = 0.0
            for prim in self.stage.Traverse():
                if str(prim.GetPath()).startswith(A4_path): 
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        self.rigid_bodies.append(prim)
                    if prim.HasAPI(UsdPhysics.MassAPI):
                        mass_api = UsdPhysics.MassAPI(prim)
                        mass_attr = mass_api.GetMassAttr()
                        if mass_attr.HasAuthoredValue(): total_mass += mass_attr.Get()                    
            if DEBUG: print(  f"\nRigid bodies:\n"  +  "".join([f"{rb.GetPath()}\n" for rb in self.rigid_bodies])  )
            self._total_mass = total_mass
            if DEBUG: print(f"\nself._total_mass = ", total_mass, "\n")

            # Verification: List of joints in agent A4 in env_0
            self.joints = []    
            for prim in self.stage.Traverse():
                if str(prim.GetPath()).startswith(A4_path): 
                    if UsdPhysics.Joint(prim):
                        self.joints.append(prim)                  
            if DEBUG: print(  f"\nJoints:\n"  +  "".join([f"{rb.GetPath()}\n" for rb in self.joints])  )

            # List of rigid bodies making up the drone block in all agents
            self.prims_drone_block = []
            nam_obj = "tn__Drone_body_with_flangesAssem4d1_ik0xp0"
            for prim in self.stage.Traverse():
                if str(prim.GetName()) == nam_obj:
                    prim_block = prim.GetChildren()  
                    self.prims_drone_block.append(prim_block)
            if DEBUG:
                print(  f"\nPrims of all drone blocks:")
                for rb in self.prims_drone_block:       
                    print(  ["".join(f"{x.GetName()}\n") for x in rb]   )     

            # Verification: collect the prims where forces will be applied
            self.name_drone = "tn__Drone_body_with_flangesAssem4d1_ik0xp0" #"tn__MODELSimpleDrone11_sQI" 
            #nam0 = "/World/envs/env_.*/A4/ASSEM4D_with_joints/ASSEM4D/tn__Drone_body_with_flangesAssem4d1_ik0xp0"
            self.drone_bodies_prims = [prim for prim in self.stage.Traverse() 
                                if ( (self.name_drone == str(prim.GetName())) and (prim.HasAPI(UsdPhysics.RigidBodyAPI)) ) ] # List of drone bodies in agent A4
            if DEBUG: print("\n\nDrone bodies as prims:\n", "".join([f"{str(prim.GetPath())}\n" for prim in self.drone_bodies_prims]), f"\n")
        

            # Verification of hierarchy
            #print_hierarchy(self.stage.GetPseudoRoot())

            self.art = self.scene.articulations["Agent"]
            self.env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

            self.first_reset = True

            if DEBUG: print(f"\n\nEND SETUP_SCENE\n\n\n")






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




    

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
            """Pre-process actions before stepping through the physics.

            This function is responsible for pre-processing the actions before stepping through the physics.
            It is called before the physics stepping (which is decimated).

            Args:
                actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
            """
            if DEBUG: print("pre-physics")
            #self._actions = actions.clone()
            self._actions = actions.clone().clamp(-2.0, 1.0)
            COMs = self.art.data.body_com_pos_w        #  (num_envs, nba, 3)   with nba = number of bodies in articulation
            if DEBUG:
                COMs = COMs[:, 2, :].float() #  (num_envs, 3)
                print(f"self._actions = {self._actions}\n")
                print("COMs =\n", COMs)




    def _apply_action(self) -> None:   
            """Apply actions to the simulator.
            This function is responsible for applying the actions to the simulator. It is called at each
            physics time-step.
            """
            if DEBUG: print("apply action")
            self.control.apply_forces( 
                self.drone_bodies_prims,
                self.env_ids,
                self.drone_body_index_in_articulation,
                self.number_of_bodies_in_articulation,
                self.art,
                self.device,
                self._actions, # Sequence[Sequence[float]]                   # TO-DO
                add_reaction_torque = False
                )
            self.art.write_data_to_sim()
            
        



    def _get_observations(self) -> dict:     #{"policy": VecEnvObs}          
            """Compute and return the observations for the environment.

            Returns:
                The observations for the environment.
            """
            if DEBUG: print("get observations")
            desired_pos_b, _ = subtract_frame_transforms(
                self.agent.data.root_pos_w, self.agent.data.root_quat_w, self._desired_pos_w
            )                                                           # expected (num_envs, 3)
            obs = torch.cat(
                [
                    self.agent.data.root_lin_vel_b,
                    self.agent.data.root_ang_vel_b,
                    self.agent.data.projected_gravity_b,
                    desired_pos_b,
                ],
                dim=-1,
            )
            observations = {"policy": obs}                              # expected (num_envs, 12)

            if self.first_reset:
                 pass
            else:
                assert not torch.isnan(obs).any(), "WARNING ISNAN in obs"
                assert not torch.isinf(obs).any(), "WARNING ISINF in obs"
                assert torch.linalg.norm(desired_pos_b) < 10**3, "EXPLODING desired_pos_b"
                if DEBUG: 
                    COMs = self.art.data.body_com_pos_w        #  (num_envs, nba, 3)   with nba = number of bodies in articulation
                    COMs = COMs[:, 2, :].float() #  (num_envs, 3)
                    print("COMs =\n", COMs)

            return observations
    



    def _get_rewards(self) -> torch.Tensor:                           
            """Compute and return the rewards for the environment.
            Returns:
                The rewards for the environment. Shape is (num_envs,).
            """
            if DEBUG: print("get rewards")
            # For info:
            # lin_vel_reward_scale = -0.05
            # ang_vel_reward_scale = -0.01
            # distance_to_goal_reward_scale = 15.0
            lin_vel = torch.sum(torch.square(self.art.data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(self.art.data.root_ang_vel_b), dim=1)
            distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.art.data.root_pos_w, dim=1)
            distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
            rewards = {
                "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,  # expected  (num_envs, )
                "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,  # expected  (num_envs, )
                "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt, # expected  (num_envs, )
            }
            stak = torch.stack(list(rewards.values()))                # expected  (3, num_envs)
            reward = torch.sum(stak, dim=0)                           # expected  (num_envs, )
            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
            if self.first_reset:
                 pass
            else:
                if DEBUG:
                    COMs = self.art.data.body_com_pos_w        #  (num_envs, nba, 3)   with nba = number of bodies in articulation
                    COMs = COMs[:, 2, :].float() #  (num_envs, 3)
                    print("COMs =\n", COMs)
                    print(f"self._actions = {self._actions}\n")
            return reward




    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            if DEBUG: print("get dones")
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            died = torch.logical_or(self.art.data.root_pos_w[:, 2] < 0.0, self.art.data.root_pos_w[:, 2] > 40.0)
            if not ((~died).all): print(f"\n\nSome agent died ", died, "\n\n")        # expected (num_envs, )
            if self.first_reset:
                 pass
            else:
                if DEBUG:
                    COMs = self.art.data.body_com_pos_w        #  (num_envs, nba, 3)   with nba = number of bodies in articulation
                    COMs = COMs[:, 2, :].float() #  (num_envs, 3)
                    print("COMs =\n", COMs)
                    print(f"self._actions = {self._actions}\n")
            return died, time_out




    #def _reset_idx(self, env_ids: Sequence[int] | None): # TO-DO
    def _reset_idx(self, env_ids: torch.Tensor | None):
            """Reset environments based on specified indices.
            Args:
                env_ids: List of environment ids which must be reset
            """
            if DEBUG: print(f"reset_idx for env_ids = {env_ids}")
            art = self.scene.articulations["Agent"]
            indices, _ = art.find_bodies(self.name_drone)
            self.drone_body_index_in_articulation = indices[0]
            

            # Set collision filtering
            #for env_id in env_ids:
            #    prim_path = f"/World/envs/env_{env_id}"
            #    prim = self.stage.GetPrimAtPath(prim_path)
            #    # Apply PhysX API
            #    rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            #    rb_api.CreateCollisionGroupAttr().Set(env_id)
            #    rb_api.CreateCollisionMaskAttr().Set(1 << env_id)


            if env_ids is None:
                env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
            else:
                 pass #print(f"\nReset for env_ids = {env_ids}\n")
            
            # Checking for NaN or inf
            if self.first_reset:
                 pass
            else:
                assert not torch.isnan(self.actions).any(), "WARNING ISNAN in actions"
                assert not torch.isinf(self.actions).any(), "WARNING ISINF in actions"
                assert not torch.isnan(art.data.body_com_pos_w).any(), "WARNING ISNAN in COM position"
                assert not torch.isinf(art.data.body_com_pos_w).any(), "WARNING ISNINF in COM position"
            self.first_reset = False

            # Logging the results of the previous episode
            # For info, the keys are:
            # lin_vel
            # ang_vel
            # distance_to_goal
            final_distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[env_ids] - art.data.root_pos_w[env_ids], dim=1
                ).mean()

            extras = dict()
            for key in self._episode_sums.keys():                                 # _episode_sums is initialised in __init__
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])   # whatever key, _episode_sums[key][env_ids] is (num_envs,)
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
                
            self.extras["log"] = dict()
            self.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
            extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
            self.extras["log"].update(extras)

            super()._reset_idx(env_ids)

            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            #if len(env_ids) == self.num_envs:  
            #    self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))


            # Now (re-)initialize task-specific parameters
            self._actions[env_ids] = 0.0                      # (num_envs, 4)    ou         (num_envs, num_motors)
            # Sample new targets
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
            # Add offset due to cloning
            # if using terrain:
            #self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            # otherwise, if using groundPlane
            self._desired_pos_w[env_ids, :2] += self.scene.env_origins[env_ids, :2] + self.art.data.default_root_state[env_ids, :2]    # TO-DO is in the local env plane.
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(1.0, 2.0)
            
            # Reset agent state
            # joints
            joint_pos = art.data.default_joint_pos[env_ids]
            joint_vel = art.data.default_joint_vel[env_ids]
            art.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel, env_ids=env_ids)  
            
            # articulation root
            default_root_state = art.data.default_root_state[env_ids]
            default_root_state[:,:3] += self.scene.env_origins[env_ids]
            #print("default_root_state =\n", default_root_state)
            assert len(env_ids) == len(default_root_state), "length of env_ids not matching default_root_state in reset function" 
            art.write_root_pose_to_sim(default_root_state[:,:7], env_ids)
            art.write_root_velocity_to_sim(default_root_state[:,7:], env_ids)

            if self.first_reset:
                 pass
            else:
                if DEBUG:
                    COMs = self.art.data.body_com_pos_w        #  (num_envs, nba, 3)   with nba = number of bodies in articulation
                    COMs = COMs[:, 2, :].float() #  (num_envs, 3)
                    print("COMs =\n", COMs)
                    print(f"self._actions = {self._actions}\n")

            #print(f"END RESET\n\n\n")
            #breakpoint()

            


        
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