import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
from prettytable import PrettyTable

from mjlab.envs import types
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionManager, ActionTermCfg
from mjlab.managers.command_manager import (
  CommandManager,
  CommandTermCfg,
  NullCommandManager,
)
from mjlab.managers.curriculum_manager import (
  CurriculumManager,
  CurriculumTermCfg,
  NullCurriculumManager,
)
from mjlab.managers.event_manager import EventManager, EventTermCfg
from mjlab.managers.metrics_manager import (
  MetricsManager,
  MetricsTermCfg,
  NullMetricsManager,
)
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlab.managers.recorder_manager import (
  NullRecorderManager,
  RecorderManager,
  RecorderTermCfg,
)
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.utils import random as random_utils
from mjlab.utils.logging import print_info
from mjlab.utils.spaces import Box
from mjlab.utils.spaces import Dict as DictSpace
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
  """Configuration for a manager-based RL environment.

  This config defines all aspects of an RL environment: the physical scene,
  observations, actions, rewards, terminations, and optional features like
  commands and curriculum learning.

  The environment step size is ``sim.mujoco.timestep * decimation``. For example,
  with a 2ms physics timestep and decimation=10, the environment runs at 50Hz.
  """

  # Base environment configuration.

  decimation: int
  """Number of physics simulation steps per environment step. Higher values mean
  coarser control frequency. Environment step duration = physics_dt * decimation."""

  scene: SceneCfg
  """Scene configuration defining terrain, entities, and sensors. The scene
  specifies ``num_envs``, the number of parallel environments."""

  observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
  """Observation groups configuration. Each group (e.g., "actor", "critic") contains
  observation terms that are concatenated. Groups can have different settings for
  noise, history, and delay."""

  actions: dict[str, ActionTermCfg] = field(default_factory=dict)
  """Action terms configuration. Each term controls a specific entity/aspect
  (e.g., joint positions). Action dimensions are concatenated across terms."""

  events: dict[str, EventTermCfg] = field(
    default_factory=lambda: {
      "reset_scene_to_default": EventTermCfg(
        func=reset_scene_to_default,
        mode="reset",
      )
    }
  )
  """Event terms for domain randomization and state resets. Default includes
  ``reset_scene_to_default`` which resets entities to their initial state.
  Can be set to empty to disable all events including default reset."""

  seed: int | None = None
  """Random seed for reproducibility. If None, a random seed is used. The actual
  seed used is stored back into this field after initialization."""

  sim: SimulationCfg = field(default_factory=SimulationCfg)
  """Simulation configuration including physics timestep, solver iterations,
  contact parameters, and NaN guarding."""

  viewer: ViewerConfig = field(default_factory=ViewerConfig)
  """Viewer configuration for rendering (camera position, resolution, etc.)."""

  # RL-specific configuration.

  episode_length_s: float = 0.0
  """Duration of an episode (in seconds).

  Episode length in steps is computed as:
    ceil(episode_length_s / (sim.mujoco.timestep * decimation))
  """

  rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  """Reward terms configuration."""

  terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
  """Termination terms configuration. If empty, episodes never reset. Use
  ``mdp.time_out`` with ``time_out=True`` for episode time limits."""

  commands: dict[str, CommandTermCfg] = field(default_factory=dict)
  """Command generator terms (e.g., velocity targets)."""

  curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
  """Curriculum terms for adaptive difficulty."""

  metrics: dict[str, MetricsTermCfg] = field(default_factory=dict)
  """Custom metric terms for logging per-step values as episode averages."""

  recorders: dict[str, RecorderTermCfg] = field(default_factory=dict)
  """Recorder terms for logging observations, actions, or other data during rollouts.
  If empty, a no-op manager is used with zero overhead."""

  is_finite_horizon: bool = False
  """Whether the task has a finite or infinite horizon. Defaults to False (infinite).

  - **Finite horizon (True)**: The time limit defines the task boundary. When reached,
    no future value exists beyond it, so the agent receives a terminal done signal.
  - **Infinite horizon (False)**: The time limit is an artificial cutoff. The agent
    receives a truncated done signal to bootstrap the value of continuing beyond the
    limit.
  """

  auto_reset: bool = True
  """Whether to automatically reset environments that terminate or time out.

  When True (default), ``step()`` resets done environments and returns post-reset
  observations. When False, ``step()`` returns the true terminal observation and the
  caller must explicitly call ``reset(env_ids=...)`` for done environments before the
  next ``step()``.

  Note: mjlab's bundled ``train.py`` goes through rsl_rl's ``OnPolicyRunner``, which
  does not drive manual resets. ``auto_reset=False`` is intended for users running
  their own training loop (or a wrapper that handles the reset between steps).
  """

  scale_rewards_by_dt: bool = True
  """Whether to multiply rewards by the environment step duration (dt).

  When True (default), reward values are scaled by step_dt to normalize cumulative
  episodic rewards across different simulation frequencies. Set to False for
  algorithms that expect unscaled reward signals (e.g., HER, static reward scaling).
  """


class ManagerBasedRlEnv:
  """Manager-based RL environment."""

  is_vector_env = True
  metadata = {
    "render_modes": [None, "rgb_array"],
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.config.version,
  }
  cfg: ManagerBasedRlEnvCfg

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    # Initialize base environment state.
    self.cfg = cfg
    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    self._sim_step_counter = 0
    self.extras = {}
    self.obs_buf = {}
    self.reward_manager: RewardManager | None = None
    self.reward_buf = torch.zeros(cfg.scene.num_envs, device=device)
    self._manual_reset_pending = torch.zeros(
      self.cfg.scene.num_envs, dtype=torch.bool, device=device
    )

    # Initialize scene and simulation.
    self.scene = Scene(self.cfg.scene, device=device)
    self.sim = Simulation(
      num_envs=self.scene.num_envs,
      cfg=self.cfg.sim,
      spec=self.scene.spec,
      variant_info=self.scene.collect_variant_info(),
      physics=self.scene.physics_cfgs,
      device=device,
    )
    self.scene.bind_physics(self.sim.physics)

    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )

    # Wire sensor context to simulation for sense_graph.
    if self.scene.sensor_context is not None:
      self.sim.set_sensor_context(self.scene.sensor_context)

    # Print environment info.
    print_info("")
    table = PrettyTable()
    table.title = "Base Environment"
    table.field_names = ["Property", "Value"]
    table.align["Property"] = "l"
    table.align["Value"] = "l"
    table.add_row(["Number of environments", self.num_envs])
    table.add_row(["Environment device", self.device])
    table.add_row(["Environment seed", self.cfg.seed])
    table.add_row(["Physics step-size", self.physics_dt])
    table.add_row(["Environment step-size", self.step_dt])
    print_info(table.get_string())
    print_info("")

    # Initialize RL-specific state.
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.scene.num_envs, device=device, dtype=torch.long
    )
    self.render_mode = render_mode
    self._offline_renderer: OffscreenRenderer | None = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRenderer(
        model=self.sim.mj_model,
        cfg=self.cfg.viewer,
        scene=self.scene,
        sim_model=self.sim.model,
        expanded_fields=self.sim.expanded_fields,
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt

    # Load all managers.
    self.load_managers()
    self.setup_manager_visualizers()

  # Properties.

  @property
  def num_envs(self) -> int:
    """Number of parallel environments."""
    return self.scene.num_envs

  @property
  def physics_dt(self) -> float:
    """Physics simulation step size."""
    return self.cfg.sim.mujoco.timestep

  @property
  def step_dt(self) -> float:
    """Environment step size (physics_dt * decimation)."""
    return self.cfg.sim.mujoco.timestep * self.cfg.decimation

  @property
  def device(self) -> str:
    """Device for computation."""
    return self.sim.device

  @property
  def max_episode_length_s(self) -> float:
    """Maximum episode length in seconds."""
    return self.cfg.episode_length_s

  @property
  def max_episode_length(self) -> int:
    """Maximum episode length in steps."""
    return math.ceil(self.max_episode_length_s / self.step_dt)

  @property
  def unwrapped(self) -> "ManagerBasedRlEnv":
    """Get the unwrapped environment (base case for wrapper chains)."""
    return self

  # Methods.

  def setup_manager_visualizers(self) -> None:
    self.manager_visualizers = {}
    if getattr(self.command_manager, "active_terms", None):
      self.manager_visualizers["command_manager"] = self.command_manager
    self.manager_visualizers["event_manager"] = self.event_manager
    self.manager_visualizers["reward_manager"] = self.reward_manager

  def load_managers(self) -> None:
    """Load and initialize all managers.

    Order is important! Event and command managers must be loaded first,
    then action and observation managers, then other RL managers.
    """
    # Event manager (required before everything else for domain randomization).
    self.event_manager = EventManager(self.cfg.events, self)
    print_info(f"[INFO] {self.event_manager}")

    self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

    # Command manager (must be before observation manager since observations
    # may reference commands).
    if len(self.cfg.commands) > 0:
      self.command_manager = CommandManager(self.cfg.commands, self)
    else:
      self.command_manager = NullCommandManager()
    print_info(f"[INFO] {self.command_manager}")

    # Action and observation managers.
    self.action_manager = ActionManager(self.cfg.actions, self)
    print_info(f"[INFO] {self.action_manager}")
    self.observation_manager = ObservationManager(self.cfg.observations, self)
    print_info(f"[INFO] {self.observation_manager}")

    # Other RL-specific managers.

    self.termination_manager = TerminationManager(self.cfg.terminations, self)
    print_info(f"[INFO] {self.termination_manager}")
    self.reward_manager = RewardManager(
      self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.reward_manager}")
    if len(self.cfg.curriculum) > 0:
      self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
    else:
      self.curriculum_manager = NullCurriculumManager()
    print_info(f"[INFO] {self.curriculum_manager}")
    if len(self.cfg.metrics) > 0:
      self.metrics_manager = MetricsManager(self.cfg.metrics, self)
    else:
      self.metrics_manager = NullMetricsManager()
    print_info(f"[INFO] {self.metrics_manager}")
    if len(self.cfg.recorders) > 0:
      self.recorder_manager = RecorderManager(self.cfg.recorders, self)
    else:
      self.recorder_manager = NullRecorderManager()
    print_info(f"[INFO] {self.recorder_manager}")

    # Configure spaces for the environment.
    self._configure_gym_env_spaces()

    # Initialize startup events if defined.
    if "startup" in self.event_manager.available_modes:
      self.event_manager.apply(mode="startup")

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    advance_curriculum: bool = False,
    options: dict[str, Any] | None = None,
  ) -> tuple[types.VecEnvObs, dict]:
    del options  # Unused.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    if seed is not None:
      self.seed(seed)
    self.extras["log"] = dict()
    self._reset_idx(env_ids, advance_curriculum=advance_curriculum)
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    self.recorder_manager.record_post_reset(env_ids)
    return self.obs_buf, self.extras

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """Run one environment step: apply actions, simulate, compute RL signals.

    When ``auto_reset=True`` (default), environments that terminate or time out are
    reset in place and the returned observation is the post-reset state. When
    ``auto_reset=False``, the reset is skipped and the returned observation is the
    terminal state; the caller must call ``reset(env_ids=...)`` for done envs before
    the next ``step()``.

    **Forward-call placement.** MuJoCo's ``mj_step`` runs forward kinematics *before*
    integration, so after stepping, derived quantities (``xpos``, ``xquat``,
    ``site_xpos``, ``cvel``, ``sensordata``) lag ``qpos``/``qvel`` by one physics
    substep. This method refreshes them before termination and reward evaluation;
    reset environments receive a second refresh after their state is written.

    .. note::

      Event and command authors do not need to call ``sim.forward()`` themselves.
      This method handles it. The only constraint is: do not read derived quantities
      (``root_link_pose_w``, ``body_link_vel_w``, etc.) in the same function that
      writes state (``write_root_state_to_sim``, ``write_joint_state_to_sim``,
      etc.). See :ref:`faq` for details.
    """
    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
      pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      raise RuntimeError(
        f"Environments {pending_ids.cpu().tolist()} must be reset via "
        "reset(env_ids=...) before calling step() again when auto_reset=False."
      )

    self.extras["log"] = dict()
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Refresh kinematics before termination and reward evaluation.
    self.sim.forward()
    self.sim.after_control_step()
    self.command_manager.observe_step()

    # Check terminations and compute rewards.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    assert self.reward_manager is not None
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # Capture completed episodes before any manager consumes/reset its state.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self.extras["episode"] = self.metrics_manager.episode_values(reset_env_ids)
      self.extras["episode"].update(
        {
          f"termination/{name}": self.termination_manager.get_term(name)[
            reset_env_ids
          ].clone()
          for name in self.termination_manager.active_terms
        }
      )
      self.recorder_manager.record_pre_reset(reset_env_ids)

    # Goal changes occur only after termination, reward, metric, and trace capture.
    self.command_manager.advance_step(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids, advance_curriculum=True)
      self.scene.write_data_to_sim()
      self.sim.forward()

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_post_reset(reset_env_ids)
    elif len(reset_env_ids) > 0:
      self._manual_reset_pending[reset_env_ids] = True

    self.recorder_manager.record_post_step()

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def get_observations(self) -> dict:
    return self.observation_manager.compute()

  def render(self) -> np.ndarray | None:
    if self.render_mode == "human" or self.render_mode is None:
      return None
    elif self.render_mode == "rgb_array":
      if self._offline_renderer is None:
        raise ValueError("Offline renderer not initialized")
      debug_callback = (
        self.update_visualizers if hasattr(self, "update_visualizers") else None
      )
      self._offline_renderer.update(self.sim.data, debug_vis_callback=debug_callback)
      return self._offline_renderer.render()
    else:
      raise NotImplementedError(
        f"Render mode {self.render_mode} is not supported. "
        f"Please use: {self.metadata['render_modes']}."
      )

  def close(self) -> None:
    if self._offline_renderer is not None:
      self._offline_renderer.close()
    self.recorder_manager.close()

  @staticmethod
  def seed(seed: int = -1) -> int:
    if seed == -1:
      seed = np.random.randint(0, 10_000)
    print_info(f"Setting seed: {seed}")
    random_utils.seed_rng(seed)
    return seed

  def update_visualizers(self, visualizer: DebugVisualizer) -> None:
    for mod in self.manager_visualizers.values():
      mod.debug_vis(visualizer)
    for sensor in self.scene.sensors.values():
      sensor.debug_vis(visualizer)

  # Private methods.

  def _configure_gym_env_spaces(self) -> None:
    from mjlab.utils.spaces import batch_space

    self.single_observation_space = DictSpace()
    for group_name, group_term_names in self.observation_manager.active_terms.items():
      has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
      group_dim = self.observation_manager.group_obs_dim[group_name]
      if has_concatenated_obs:
        assert isinstance(group_dim, tuple)
        self.single_observation_space.spaces[group_name] = Box(
          shape=group_dim, low=-math.inf, high=math.inf
        )
      else:
        assert not isinstance(group_dim, tuple)
        group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
        # Create a nested dict for this group.
        group_space = DictSpace()
        for term_name, term_dim, _term_cfg in zip(
          group_term_names, group_dim, group_term_cfgs, strict=False
        ):
          group_space.spaces[term_name] = Box(
            shape=term_dim, low=-math.inf, high=math.inf
          )
        self.single_observation_space.spaces[group_name] = group_space

    action_dim = sum(self.action_manager.action_term_dim)
    self.single_action_space = Box(shape=(action_dim,), low=-math.inf, high=math.inf)

    self.observation_space = batch_space(self.single_observation_space, self.num_envs)
    self.action_space = batch_space(self.single_action_space, self.num_envs)

  def training_state_dict(self) -> dict[str, Any]:
    """Return versioned state that affects future training transitions."""

    return {
      "schema_version": 2,
      "common_step_counter": self.common_step_counter,
      "sim_step_counter": self._sim_step_counter,
      "events": self.event_manager.training_state_dict(),
      "curriculum": self.curriculum_manager.training_state_dict(),
    }

  def load_training_state_dict(self, state: dict[str, Any]) -> None:
    """Restore training state before the first post-resume reset."""

    self.validate_training_state_dict(state)
    self.common_step_counter = int(state["common_step_counter"])
    self._sim_step_counter = int(state["sim_step_counter"])
    if state["schema_version"] >= 2:
      self.event_manager.load_training_state_dict(state["events"])
    self.curriculum_manager.load_training_state_dict(state["curriculum"])

  def validate_training_state_dict(self, state: dict[str, Any]) -> None:
    """Validate continuation state without mutating the environment."""

    if state.get("schema_version") not in (1, 2):
      raise ValueError("Unsupported environment training-state schema")
    if not isinstance(state.get("common_step_counter"), int):
      raise TypeError("common_step_counter must be an integer")
    if not isinstance(state.get("sim_step_counter"), int):
      raise TypeError("sim_step_counter must be an integer")
    if state["schema_version"] >= 2:
      if not isinstance(state.get("events"), dict):
        raise TypeError("events must be a dictionary")
      self.event_manager.validate_training_state_dict(state["events"])
    self.curriculum_manager.validate_training_state_dict(state["curriculum"])

  def _reset_idx(
    self,
    env_ids: torch.Tensor | None = None,
    *,
    advance_curriculum: bool = False,
  ) -> None:
    if advance_curriculum:
      self.curriculum_manager.compute(env_ids=env_ids)
    self.sim.reset(env_ids)
    self.scene.reset(env_ids)

    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )

    # NOTE: This is order sensitive.
    # observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # rewards manager.
    assert self.reward_manager is not None
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # metrics manager.
    info = self.metrics_manager.reset(env_ids)
    self.extras["log"].update(info)
    # curriculum manager.
    info = self.curriculum_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager.
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)
    # event manager.
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode length buffer.
    self.episode_length_buf[env_ids] = 0
    self._manual_reset_pending[env_ids] = False
