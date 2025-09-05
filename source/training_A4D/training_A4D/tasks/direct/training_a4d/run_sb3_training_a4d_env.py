"""
Standalone SB3 training for DirectRLEnv TrainingA4Env
Works fully inside Isaac Lab repo (source/standalone/workflows/).

So, do:
cp run_sb3_training_a4d_env.py ~/IsaacLab/source/isaaclab_tasks/training_a4d/
Then from isaaclab root folder ~/IsaacLab/:
./isaaclab.sh -p source/isaaclab_tasks/training_a4d/run_sb3_training_a4d_env.py
"""

import torch
from stable_baselines3 import PPO

# Import ISAACLAB modules here (after Kit is bootstrapped)
from isaaclab.envs import DirectRLEnv
    

# Bootstrap Isaac Lab runtime
#import isaaclab.app  # <- starts Omniverse Kit environment
# Explicitly load physics extensions
#import omni
#omni.kit.app.get_app().load_extension("omni.physx")  # loads PhysX

 

def main():
    """
    Main training function.
    All imports that rely on omni.* happen after Kit is bootstrapped.
    """

    from .training_a4d_env_cfg import TrainingA4dEnvCfg  

    # Instantiate your environment
    env = DirectRLEnv(TrainingA4dEnvCfg())

    # Create SB3 agent (PPO here, but SAC, TD3 etc. also work)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cuda"  # or "cpu"
    )

    # Train
    model.learn(total_timesteps=1_000)

    # Save model
    model.save("ppo_sb3_a4d")



if __name__ == "__main__":
    main()
#isaaclab.app.run_app(main)
