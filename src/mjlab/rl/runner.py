import os
import random
from pathlib import Path

import numpy as np
import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.checkpoint_metadata: dict[str, object] | None = None
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if not train_cfg[key].get("aux_value"):
          train_cfg[key].pop("aux_value", None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in (
            "rnn_type",
            "rnn_hidden_dim",
            "rnn_num_layers",
            "rnn_layer_norm",
          ):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    infos = {
      **(infos or {}),
      "env_state": self.env.unwrapped.training_state_dict(),
      "rng_state": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
      },
      "checkpoint_metadata": self.checkpoint_metadata,
    }
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
    restore_training_state: bool = True,
  ) -> dict:
    """Load current weights and optionally exact continuation state."""
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    infos = loaded_dict["infos"]
    if restore_training_state:
      self.env.unwrapped.validate_training_state_dict(infos["env_state"])

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    if restore_training_state:
      self.env.unwrapped.load_training_state_dict(infos["env_state"])
      rng = infos["rng_state"]
      random.setstate(rng["python"])
      np.random.set_state(rng["numpy"])
      torch.set_rng_state(rng["torch_cpu"])
      if torch.cuda.is_available() and rng["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda"][: torch.cuda.device_count()])
    self.env.unwrapped.reset(advance_curriculum=False)
    return infos
