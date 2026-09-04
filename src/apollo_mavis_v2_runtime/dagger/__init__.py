"""DAgger / inference runtime package (12-dagger §1).

Heavy deps (torch, zmq, pyarrow) are imported lazily by the modules that
need them; importing this package stays cheap. See ``gate`` (takeover state
machine), ``loop`` (GatedPolicyExecutor + sessions), ``policy_runner``,
``recorder``, ``reloader``, ``client``, ``registry`` and the ``trainer``
subpackage (separate process).
"""

from .gate import TakeoverGateImpl

__all__ = ["TakeoverGateImpl"]
