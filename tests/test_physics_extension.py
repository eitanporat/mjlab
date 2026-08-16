from types import MappingProxyType

import pytest

from mjlab.physics import PhysicsContext, PhysicsExtension
from mjlab.scene.scene import Scene
from mjlab.sim.sim import Simulation


class Recorder(PhysicsExtension):
  def __init__(self):
    self.after_control_calls = 0

  def after_control_step(self) -> None:
    self.after_control_calls += 1


def test_physics_context_exposes_immutable_variant_assignments():
  variants = MappingProxyType({"object": object()})
  context = PhysicsContext(object(), object(), object(), 4, "cpu", variants)  # type: ignore[arg-type]

  assert context.world_variants is variants
  with pytest.raises(TypeError):
    context.world_variants["other"] = object()  # type: ignore[index]


def test_scene_rejects_mismatched_bound_physics_names():
  scene = Scene.__new__(Scene)
  scene._physics_cfgs = {"brick": object()}
  scene._physics = {}

  with pytest.raises(ValueError, match="names"):
    scene.bind_physics({"other": Recorder()})

  runtime = Recorder()
  scene.bind_physics({"brick": runtime})
  assert scene.physics == {"brick": runtime}


def test_after_control_step_dispatches_to_every_extension():
  first, second = Recorder(), Recorder()
  simulation = Simulation.__new__(Simulation)
  simulation.physics = {"first": first, "second": second}

  simulation.after_control_step()

  assert first.after_control_calls == 1
  assert second.after_control_calls == 1
