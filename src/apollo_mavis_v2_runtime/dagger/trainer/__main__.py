"""``python -m apollo_mavis_v2_runtime.dagger.trainer --config <json> [--resume]``.

Separate OS process (12-dagger §7): spawned by the runtime's
``AsyncTrainerClientImpl`` with ``CUDA_VISIBLE_DEVICES`` set; crash-isolated
from the 100 Hz servo loop.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apollo_mavis_v2_runtime.dagger.trainer")
    parser.add_argument("--config", required=True, help="TrainerConfig JSON path")
    parser.add_argument("--resume", action="store_true",
                        help="reload trainer_state.pt of the newest version")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s trainer %(levelname)s %(name)s: %(message)s",
    )
    from ..client import load_trainer_config
    from .trainer import TrainerMain

    cfg = load_trainer_config(args.config)
    return TrainerMain(cfg, resume=args.resume).run()


if __name__ == "__main__":
    sys.exit(main())
