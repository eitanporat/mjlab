"""Immutable authoring and fully bound runtime contracts for scene physics."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp


@dataclass(frozen=True)
class PhysicsContext:
  """All compiled state available while binding a physics extension."""

  host_model: mujoco.MjModel
  device_model: mjwarp.Model
  device_data: mjwarp.Data
  num_worlds: int
  device: str
  world_variants: Mapping[str, torch.Tensor]


class PhysicsExtensionCfg(abc.ABC):
  """Immutable scene-authoring configuration for one domain extension."""

  @abc.abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    """Author fixed-capacity model structure before compilation."""

  @abc.abstractmethod
  def build(self, context: PhysicsContext) -> PhysicsExtension:
    """Bind IDs and allocate a complete runtime before rollout."""


class PhysicsExtension:
  """Fully initialized runtime lifecycle."""

  def before_physics_step(self) -> None:
    pass

  def after_physics_step(self) -> None:
    pass

  def after_control_step(self) -> None:
    pass

  def reset(self, world_mask: wp.array) -> None:
    pass


__all__ = ["PhysicsContext", "PhysicsExtension", "PhysicsExtensionCfg"]
