import copy

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Dict as DictSpace
from mjlab.utils.spaces import Space


class RslRlVecEnvWrapper(VecEnv):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
    sapg_cfg: dict | None = None,
  ):
    self.env = env
    self.clip_actions = clip_actions
    self._sapg_embedding = None
    self._observation_space = copy.deepcopy(self.env.observation_space)

    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    if sapg_cfg is not None:
      block_size = int(sapg_cfg["expl_coef_block_size"])
      if self.num_envs % block_size:
        raise ValueError(
          f"num_envs {self.num_envs} must be divisible by block size {block_size}"
        )
      num_blocks = self.num_envs // block_size
      self._sapg_embedding = torch.linspace(
        50.0, 0.0, num_blocks, device=self.device
      ).repeat_interleave(block_size)
      embedding_size = (
        1
        if "learn_param" in sapg_cfg["expl_type"]
        else int(sapg_cfg["expl_reward_coef_embd_size"])
      )
      self._sapg_embedding = self._sapg_embedding[:, None].repeat(1, embedding_size)
      if isinstance(self._observation_space, DictSpace):
        for group in ("actor", "critic"):
          space = self._observation_space.spaces[group]
          space.shape = (
            *space.shape[:-1],
            space.shape[-1] + self._sapg_embedding.shape[-1],
          )
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset.
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self._observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  # Properties.

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def get_observations(self) -> TensorDict:
    obs_dict = self.unwrapped.observation_manager.compute()
    return self._add_sapg_embedding(TensorDict(obs_dict, batch_size=[self.num_envs]))

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    return self._add_sapg_embedding(
      TensorDict(obs_dict, batch_size=[self.num_envs])
    ), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    term_or_trunc = terminated | truncated
    assert isinstance(rew, torch.Tensor)
    assert isinstance(term_or_trunc, torch.Tensor)
    dones = term_or_trunc.to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return (
      self._add_sapg_embedding(TensorDict(obs_dict, batch_size=[self.num_envs])),
      rew,
      dones,
      extras,
    )

  def close(self) -> None:
    return self.env.close()

  # Private methods.

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return

    from mjlab.utils.spaces import Box, batch_space

    self.unwrapped.single_action_space = Box(
      shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
    )
    self.unwrapped.action_space = batch_space(
      self.unwrapped.single_action_space, self.num_envs
    )

  def _add_sapg_embedding(self, obs: TensorDict) -> TensorDict:
    if self._sapg_embedding is None:
      return obs
    for group in ("actor", "critic"):
      if group in obs:
        tail = self._sapg_embedding.to(dtype=obs[group].dtype)
        obs[group] = torch.cat((obs[group], tail), dim=-1)
    return obs
