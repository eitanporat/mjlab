"""Composable simulation physics."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjwarp
import warp as wp


class PhysicsExtension:
  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    pass

  def initialize(
    self, model: mujoco.MjModel, wp_model: mjwarp.Model, data: mjwarp.Data
  ) -> None:
    pass

  def before_step(self) -> None:
    pass

  def after_step(self) -> None:
    pass

  def after_control_step(self) -> None:
    pass

  def reset(self, mask: wp.array) -> None:
    pass


__all__ = ["PhysicsExtension"]
