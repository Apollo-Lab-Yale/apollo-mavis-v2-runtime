"""Reference policies (phase-08): a small torch MLP + a scripted fake.

``MLPPolicy`` is the dev/test policy implementing the core ``Policy``
protocol (delta_ee, state-only obs); the SAME ``MLPNet`` weights are what
the AsyncTrainer fine-tunes, so runtime hot-swap and trainer checkpoints
interoperate. ``ScriptedPolicy`` is a numpy-only deterministic policy for
unit tests (no GPU, NaN injectable, optional chunking).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from apollo_mavis_v2_core.interfaces.policy import Observation, PolicyOutput, PolicySpec


def resolve_device(preferred: str) -> str:
    """Preferred CUDA device when usable, else cpu (shared-box fallback)."""
    try:
        return preferred if (preferred == "cpu" or torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"


class MLPNet(torch.nn.Module):
    """2-layer MLP: observation.state -> delta_ee action block(s)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.state_dim, self.action_dim, self.hidden = state_dim, action_dim, hidden
        self.body = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def save_policy_bundle(path: str, net: MLPNet, spec_meta: dict[str, Any]) -> None:
    """state_dict.pt payload: weights + arch + spec metadata (no optimizer)."""
    torch.save(
        {
            "state_dict": net.state_dict(),
            "arch": {"state_dim": net.state_dim, "action_dim": net.action_dim,
                     "hidden": net.hidden},
            "spec": dict(spec_meta),
        },
        path,
    )


def load_policy_bundle(path: str, device: str = "cpu") -> tuple[MLPNet, dict[str, Any]]:
    """Load a bundle onto ``device``; raises on malformed/corrupt files."""
    payload = torch.load(path, map_location=device, weights_only=False)
    arch = payload["arch"]
    net = MLPNet(int(arch["state_dim"]), int(arch["action_dim"]), int(arch["hidden"]))
    net.load_state_dict(payload["state_dict"])  # validates keys/shapes
    net.to(device).eval()
    return net, dict(payload.get("spec", {}))


class MLPPolicy:
    """Core ``Policy`` over an ``MLPNet`` bundle (delta_ee, state-only)."""

    def __init__(self, spec: PolicySpec, net: MLPNet, device: str = "cpu") -> None:
        self.spec = spec
        self._device = device
        self._net = net.to(device).eval()
        self._version = spec.version
        self._lock_free_scale = 1.0  # kept for future tuning; deltas already small

    @classmethod
    def from_bundle(cls, path: str, version: int, device: str = "cpu") -> MLPPolicy:
        net, meta = load_policy_bundle(path, device)
        spec = PolicySpec(
            action_space=meta.get("action_space", "delta_ee"),
            action_frame=str(meta.get("action_frame", "")),
            action_names=list(meta.get("action_names", [])),
            state_names=list(meta.get("state_names", [])),
            camera_keys=list(meta.get("camera_keys", [])),
            version=version,
        )
        return cls(spec, net, device)

    def reset(self) -> None:  # no chunks/history to drop
        pass

    def act(self, obs: Observation) -> PolicyOutput:
        with torch.no_grad():
            x = torch.as_tensor(obs.state, dtype=torch.float32, device=self._device)
            a = self._net(x).cpu().numpy().astype(np.float32)
        return PolicyOutput(actions=a, version=self._version, t_mono=obs.t_mono)

    def load_weights(self, path: str) -> None:
        """Hot-swap: load onto a scratch net FIRST (never partially applied)."""
        net, _ = load_policy_bundle(path, self._device)
        self._net.load_state_dict(net.state_dict())
        self._net.eval()

    def set_version(self, version: int) -> None:
        self._version = int(version)
        object.__setattr__(self.spec, "version", int(version))


class ScriptedPolicy:
    """Deterministic numpy policy for unit tests (no torch).

    ``script(k) -> np.ndarray`` gives the k-th action; ``nan_at`` injects a
    NaN action at those call indices; ``chunk`` > 1 emulates a chunked policy
    (``chunk_remaining`` counts down; ``reset()`` drops the chunk).
    """

    def __init__(
        self,
        spec: PolicySpec,
        script=None,
        nan_at: frozenset[int] | set[int] = frozenset(),
        chunk: int = 1,
    ) -> None:
        self.spec = spec
        dim = len(spec.action_names)
        self._script = script or (lambda k: np.zeros(dim, dtype=np.float32))
        self._nan_at = set(nan_at)
        self._chunk = int(chunk)
        self.calls = 0
        self.resets = 0
        self._version = spec.version

    def reset(self) -> None:
        self.resets += 1

    def act(self, obs: Observation) -> PolicyOutput:
        k = self.calls
        self.calls += 1
        a = np.asarray(self._script(k), dtype=np.float32)
        if k in self._nan_at:
            a = np.full_like(a, np.nan)
        remaining = (self._chunk - 1) if self._chunk > 1 else 0
        return PolicyOutput(
            actions=a, version=self._version,
            t_mono=obs.t_mono if obs is not None else time.monotonic(),
            chunk_remaining=remaining,
        )

    def load_weights(self, path: str) -> None:
        pass

    def set_version(self, version: int) -> None:
        self._version = int(version)
        object.__setattr__(self.spec, "version", int(version))


__all__ = ["MLPNet", "MLPPolicy", "ScriptedPolicy", "save_policy_bundle",
           "load_policy_bundle", "resolve_device"]
