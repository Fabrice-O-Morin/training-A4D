# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from .A4_contoller_cfg import A4ForcesController



#class EnvWindow():
#    """Window manager for the Quadcopter environment."""
#
#    def __init__(self, env: TrainingA4dEnv, window_name: str = "IsaacLab_for_A4"):
#        """Initialize the window.
#        Args:
#            env: The environment object.
#            window_name: The name of the window.
#        """
#        # initialize base window
#        super().__init__(env, window_name)
#        # add custom UI elements
#        with self.ui_window_elements["main_vstack"]:
#            with self.ui_window_elements["debug_frame"]:
#                with self.ui_window_elements["debug_vstack"]:
#                    # add command manager visualization
#                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class TrainingA4dSceneCfg(InteractiveSceneCfg):
    
    # scene
    # See the meaning of the args here: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveSceneCfg
    num_envs=16 
    env_spacing=3.0 
    clone_in_fabric=False
    replicate_physics=True 
    filter_collisions=True

    

   



@configclass
class TrainingA4dEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 4
    observation_space = 12
    state_space = 0

    scene = TrainingA4dSceneCfg()

    A4_RIGID_CFG = ArticulationCfg(
        #prim_path = "{ENV_REGEX_NS}/A4",
        prim_path = "/World/envs/env_.*/A4",
        #prim_path = "/World/envs/env_0",
        spawn = sim_utils.UsdFileCfg(
            usd_path = f"/home/fom/Documents/ISAAC5/ASSEM4Dv2.usda",
            copy_from_source=False,
        ),
        actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # apply to all joints
            stiffness=0.0,            # no spring-like behavior
            damping=0.0,              # no damping either
            )
        },
        #rigid_props = { ".*": sim_utils.RigidBodyPropertiesCfg(
        #        disable_gravity = False,
        #        max_depenetration_velocity = 10.0,
        #        enable_gyroscopic_forces = True,
        #        )},
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),        # start 0.5 m above ground
            rot=(0.0, 0.0, 0.0, 1.0),  # identity quaternion (x,y,z,w)
            joint_pos={"Axle_to_holder": 0.0, "Axle_to_drone_block": 0.0},
            #joint_vel,
            #lin_vel,
            #ang_vel,
        ), # see: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationCfg.InitialStateCfg
    )

    ARTICULATIONS = [A4_RIGID_CFG]



     # simulation
    sim: SimulationCfg = SimulationCfg(dt = 1/100, 
                                       device = "cuda:0",
                                       render_interval = decimation
                                       )
   

   # terrain / ground properties
    terrain = TerrainImporterCfg(
        prim_path="/World/GroundPlane",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # custom parameters/scales
    thrust_to_weight = 1.9
    moment_scale = 0.01
    # - controllable joint
    holder_dof_name = "axle_to_holder"
    drone_dof_name  = "axle_to_drone"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    control = A4ForcesController()

    