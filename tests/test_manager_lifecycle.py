from dataclasses import dataclass

import pytest
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import NullCurriculumManager
from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg


class MetricEnvironment:
  num_envs = 2
  device = "cpu"

  def __init__(self):
    self.values = torch.zeros(2)


def metric_value(env: MetricEnvironment) -> torch.Tensor:
  return env.values


def test_metric_reductions_are_non_consuming_and_reset_cleanly():
  env = MetricEnvironment()
  manager = MetricsManager(
    {
      name: MetricsTermCfg(func=metric_value, reduce=reduce)
      for name, reduce in {
        "last": "last",
        "sum": "sum",
        "mean": "mean",
        "min": "min",
        "max": "max",
      }.items()
    },
    env,
  )
  env.values[:] = torch.tensor([3.0, 5.0])
  manager.compute()
  env.values[:] = torch.tensor([1.0, 9.0])
  manager.compute()

  first = manager.episode_values()
  second = manager.episode_values()
  torch.testing.assert_close(first["last"], torch.tensor([1.0, 9.0]))
  torch.testing.assert_close(first["sum"], torch.tensor([4.0, 14.0]))
  torch.testing.assert_close(first["mean"], torch.tensor([2.0, 7.0]))
  torch.testing.assert_close(first["min"], torch.tensor([1.0, 5.0]))
  torch.testing.assert_close(first["max"], torch.tensor([3.0, 9.0]))
  torch.testing.assert_close(first["sum"], second["sum"])

  manager.reset(torch.tensor([0]))
  reset_values = manager.episode_values(torch.tensor([0]))
  assert reset_values["last"].item() == 0.0
  assert reset_values["min"].item() == float("inf")


@dataclass(kw_only=True)
class SpyCommandCfg(CommandTermCfg):
  def build(self, env):
    return SpyCommand(self, env)


class SpyCommand(CommandTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.target = torch.zeros(self.num_envs)
    self.observed_target = torch.zeros_like(self.target)

  @property
  def command(self):
    return self.target[:, None]

  def update_metrics(self) -> None:
    self.observed_target.copy_(self.target)

  def resample_command(self, env_ids: torch.Tensor) -> None:
    self.target[env_ids] += 1.0

  def update_command(self) -> None:
    pass


class CommandEnvironment:
  num_envs = 3
  device = "cpu"


def test_command_advances_only_after_observation_and_isolates_partial_reset():
  command = SpyCommand(
    SpyCommandCfg(resampling_time_range=(100.0, 100.0)), CommandEnvironment()
  )
  command.reset(torch.arange(3))
  before = command.target.clone()
  command.request_resample(torch.tensor([1]))
  command.observe_step()
  torch.testing.assert_close(command.target, before)
  torch.testing.assert_close(command.observed_target, before)

  command.advance_step(0.1)
  torch.testing.assert_close(command.target, torch.tensor([1.0, 2.0, 1.0]))
  command.reset(torch.tensor([0]))
  torch.testing.assert_close(command.target, torch.tensor([2.0, 2.0, 1.0]))


def test_null_curriculum_accepts_only_its_canonical_state():
  manager = NullCurriculumManager()
  state = {"schema_version": 1, "reported_state": {}, "terms": {}}
  manager.load_training_state_dict(state)
  assert manager.training_state_dict() == state
  with pytest.raises(ValueError):
    manager.load_training_state_dict({**state, "terms": {"unknown": {}}})
