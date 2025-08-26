from stable_baselines3 import PPO
from env import WaterChlorinationEnv
from scenarios import load_scenario
from gymnasium.wrappers import NormalizeObservation
import gymnasium
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
import numpy as np
import os
import multiprocessing as mp
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
import csv
import torch
import glob


class StatsCallback(BaseCallback):
    def __init__(self, log_dir: str = "./logs", save_freq: int = 10000, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.save_freq = save_freq
        self.step_log_path = os.path.join(log_dir, "per_step_log.csv")
        self.episode_log_path = os.path.join(log_dir, "per_episode_log.csv")

        self.episode_rewards = []
        self.episode_idx = 0

        # Per-step CSV
        if not os.path.exists(self.step_log_path):
            with open(self.step_log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep",
                    "reward",
                    "action_magnitude",
                    "obs_mean",
                    "cl2_violation_upper",
                    "cl2_violation_lower",
                    "cl2_total_deviation",
                    "cl_penalty",
                    "fairness_penalty",
                    "control_cost",
                    "smoothness_penalty",
                    "infection_risk_penalty",
                    "unscaled_reward"
                ])

        # Per-episode CSV
        if not os.path.exists(self.episode_log_path):
            with open(self.episode_log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward", "timestep"])

    def _on_training_start(self) -> None:
        # Now self.training_env is guaranteed to exist
        self.episode_rewards = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])
        obs = self.locals.get("new_obs", [])
        dones = self.locals.get("dones", [])

        for i in range(len(rewards)):
            reward = float(self.training_env.unnormalize_reward(np.array([rewards[i]])))
            observation = self.training_env.unnormalize_obs(obs[i])
            self.episode_rewards[i].append(reward)

            info = infos[i] if i < len(infos) else {}
            action = actions[i] if i < len(actions) else np.zeros_like(obs[0])
            # observation = observation if i < len(obs) else np.zeros_like(obs[0])

            cl2_upper = info.get("cl2_violation_upper", 0.0)
            cl2_lower = info.get("cl2_violation_lower", 0.0)
            cl2_total = info.get("cl2_total_deviation", 0.0)

            act_mag = float(np.linalg.norm(action))
            obs_mean = float(np.mean(observation))

            cl_penalty = info.get("cl_penalty", 0.0)
            fairness_penalty = info.get("fairness_penalty", 0.0)
            control_cost = info.get("control_cost", 0.0)
            smoothness_penalty = info.get("smoothness_penalty", 0.0)
            infection_risk_penalty = info.get("infection_risk_penalty", 0.0)
            unscaled_reward = info.get("unscaled_reward", reward)

            self.logger.record("step/reward", reward)
            self.logger.record("step/action_magnitude", act_mag)
            self.logger.record("step/obs_mean", obs_mean)
            self.logger.record("step/cl2_violation_upper", cl2_upper)
            self.logger.record("step/cl2_violation_lower", cl2_lower)
            self.logger.record("step/cl2_total_deviation", cl2_total)
            self.logger.record("step/cl_penalty", cl_penalty)
            self.logger.record("step/fairness_penalty", fairness_penalty)
            self.logger.record("step/control_cost", control_cost)
            self.logger.record("step/smoothness_penalty", smoothness_penalty)
            self.logger.record("step/infection_risk_penalty", infection_risk_penalty)
            self.logger.record("step/unscaled_reward", unscaled_reward)

            # Step-level CSV log
            with open(self.step_log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    reward,
                    act_mag,
                    obs_mean,
                    cl2_upper,
                    cl2_lower,
                    cl2_total,
                    cl_penalty,
                    fairness_penalty,
                    control_cost,
                    smoothness_penalty,
                    infection_risk_penalty,
                    unscaled_reward
                ])

            if dones[i]:
                total_reward = sum(self.episode_rewards[i])
                self.episode_rewards[i] = []

                self.logger.record(f"episode/total_reward_{i}", total_reward)

                with open(self.episode_log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode_idx, total_reward, self.num_timesteps])

                if self.verbose:
                    print(f"[Episode {self.episode_idx}] Total reward: {total_reward:.2f}")

                self.episode_idx += 1

        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.log_dir, f"model_step_{self.num_timesteps}.zip")
            self.model.save(model_path)
            if self.verbose:
                print(f"[Callback] Saved model at {self.num_timesteps} → {model_path}")

        return True

    def _on_training_end(self):
        final_path = os.path.join(self.log_dir, "model_final.zip")
        self.model.save(final_path)
        if self.verbose:
            print(f"[Callback] Final model saved to {final_path}")


class SurrogateTrainingDataCallback(BaseCallback):
    def __init__(self, log_dir: str = "./logs", save_freq: int = 10000, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.save_freq = save_freq
        self.csv_path = os.path.join(log_dir, "surrogate_training_data.csv")

        # Write header if file doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["scenario_id", "obs", "action", "reward"])

        self._last_scenario_ids = None
        self._pending_lines = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions", [])
        obs = self.locals.get("new_obs", [])
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        scenario_ids = []
        for info in infos:
            scenario_id = info.get("scenario_id", -1)
            scenario_ids.append(scenario_id)

        for i in range(len(rewards)):
            sid = scenario_ids[i]
            reward = float(self.training_env.unnormalize_reward(np.array([rewards[i]])))
            observation = self.training_env.unnormalize_obs(obs[i])
            obs_str = ";".join(f"{x}" for x in observation)
            action_str = ";".join(f"{x}" for x in actions[i])
            reward_val = reward

            self._pending_lines.append([sid, obs_str, action_str, reward_val])

            if dones[i]:
                with open(self.csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self._pending_lines)
                    writer.writerow([])

                self._pending_lines = []

        if self.num_timesteps % self.save_freq == 0 and self._pending_lines:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self._pending_lines)
            self._pending_lines = []

        return True

    def _on_training_end(self) -> None:
        if self._pending_lines:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self._pending_lines)
            self._pending_lines = []


def evaluate_model_on_env(model, env, num_episodes=5):
    episode_rewards = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

def make_env(sc):
    def _init():
        # return RemoveScadaDataWrapper(WaterChlorinationEnv(**sc))
        return WaterChlorinationEnv(**sc)
    return _init

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class RemoveScadaDataWrapper(gymnasium.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info.pop("scada_data", None)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.pop("scada_data", None)
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    sids = range(0, 1)
    scenario_configs = [load_scenario(i) for i in sids]
    print(f'envs: {list(sids)}') 

    env_fns = [make_env(sc) for sc in scenario_configs]

    vec_env = SubprocVecEnv(env_fns)
    print('loaded envs')

    # Train
    stats_cb = StatsCallback(log_dir="./ppo/logs_quad_exp", verbose=1)
    surrogate_data_cb = SurrogateTrainingDataCallback(log_dir="./ppo/logs_quad_exp", verbose=1)
    callback = CallbackList([stats_cb, surrogate_data_cb])

    # policy_kwargs = dict(
    #     lstm_hidden_size=128,
    #     n_lstm_layers=1,
    #     shared_lstm=False,
    #     enable_critic_lstm=True,
    #     net_arch=dict(pi=[64], vf=[64]),
    #     activation_fn=torch.nn.Tanh,
    #     ortho_init=True,
    #     log_std_init=np.log(1.0),
    # )

    policy_kwargs = dict(
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(pi=[128, 64], vf=[256, 128, 64]),
        activation_fn=torch.nn.Tanh,
        ortho_init=False,
        log_std_init=np.log(1.0),
    )

    # model = RecurrentPPO(
    #     "MlpLstmPolicy", 
    #     vec_env, 
    #     policy_kwargs=policy_kwargs, 
    #     verbose=1, 
    #     tensorboard_log="./ppo/tb", 
    #     ent_coef=.01,
    #     n_steps=32,
    #     batch_size=256
    # )

    save_files = sorted(glob.glob(os.path.join('ppo', 'logs_quad', 'model_step*')), reverse=True, key=lambda x: int(x.split('_')[-1][:-4]))

    model = RecurrentPPO(
        "MlpLstmPolicy",
        VecMonitor(VecNormalize(DummyVecEnv(env_fns))),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo/tb",
        ent_coef=0.01,
        learning_rate=3e-4,
        n_steps=864,
        batch_size=864,
    )
    model.set_parameters(save_files[0])


    # model = RecurrentPPO.load(save_files[0], env=VecMonitor(VecNormalize(DummyVecEnv(env_fns))))

    actor_params = sum(p.numel() for p in model.policy.mlp_extractor.policy_net.parameters())
    critic_params = sum(p.numel() for p in model.policy.mlp_extractor.value_net.parameters())
    actor_lstm_params = sum(p.numel() for p in model.policy.lstm_actor.parameters())
    critic_lstm_params = sum(p.numel() for p in model.policy.lstm_critic.parameters())
    total_actor_params = actor_params + actor_lstm_params
    total_critic_params = critic_params + critic_lstm_params
    print(f"actor params: {total_actor_params}")
    print(f"critic params: {total_critic_params}")

    # promotes some sort of action in beg.
    with torch.no_grad():
        model.policy.action_net.bias.data.fill_(3.0)

    model.learn(total_timesteps=10000000, callback=callback, progress_bar=True, log_interval=1)

    for i, sc in enumerate(scenario_configs):
        env = NormalizeObservation(WaterChlorinationEnv(**sc))
        mean_reward, std_reward = evaluate_model_on_env(model, env, num_episodes=2)
        print(f"Env {i+1}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        env.close()
        del env

    # with WaterChlorinationEnv(**load_scenario(1)) as env:
    #     obs, info = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         obs, reward, terminated, truncated, info = env.step(np.zeros((5,)))
    #         done = terminated or truncated
    #         total_reward += reward
