import torch
from isaaclab.envs import DirectRLEnv
from .training_a4d_env_cfg import TrainingA4dEnvCfg   

from skrl.envs.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.models.torch import DeterministicMixin, GaussianMixin
from skrl.resources.models.torch import Model
from skrl.resources.preprocessing import RunningStandardScaler


# 1. Define the environment
env = DirectRLEnv(TrainingA4dEnvCfg())
env = wrap_env(env)  # makes it compatible with skrl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Define models (policy and value)
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_actions)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), torch.zeros(self.num_actions, device=self.device)


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"])


policy = Policy(env.observation_space, env.action_space, device)
value = Value(env.observation_space, env.action_space, device)


# 3. Configure and create the PPO agent
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 2
cfg["discount_factor"] = 0.99
cfg["learning_rate"] = 1e-3
cfg["state_preprocessor"] = RunningStandardScaler

agent = PPO(models={"policy": policy, "value": value}, memory=None, cfg=cfg, env=env, device=device)


# 4. Trainer
trainer = SequentialTrainer(cfg={"timesteps": 1_000}, env=env, agents=agent)

# 5. Train
trainer.train()
