"""``python -m apollo_mavis_v2_runtime.tools.export_lerobot <repo_id> [--out DIR]``
— rebuild the LeRobot v3 export of an episode-directory dataset (10-frames §11.8;
04-runtime §10.6). The same code the REST job runs; the dataset directory is spelled
by the same ``DatasetStore.root_of`` the runtime uses (15-online-dagger §7, D5): the
generic root and the per-namespace roots come from the runtime config (``--config``
/ ``APOLLO_CONFIG``, else the defaults), or ``--root`` names ONE generic
``<root>/<ns>/<name>`` tree (no namespace map). A bare ``<name>`` resolves into
``--namespace``, else the config's ``datasets.default_namespace`` (``bc_demo``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ..recorder.datasets import DatasetStore
from ..recorder.export_lerobot import ExportError, ExportProgress, export_lerobot_v3


def _dataset_store(args: argparse.Namespace) -> DatasetStore:
    """The store that spells dataset directories: ``--root`` = one generic tree with
    no mapped namespaces, else the runtime config's generic root + namespace map."""
    from ..config import DatasetsConfig, load_runtime_config

    if args.root is not None:
        return DatasetStore(
            Path(args.root),
            default_namespace=args.namespace or DatasetsConfig().default_namespace,
            namespaces={},
        )
    cfg = load_runtime_config(args.config or os.environ.get("APOLLO_CONFIG"))
    return DatasetStore(
        cfg.datasets_root,
        default_namespace=args.namespace or cfg.datasets.default_namespace,
        namespaces=cfg.datasets.namespaces,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apollo_mavis_v2_runtime.tools.export_lerobot",
        description="Export an episode-directory dataset to LeRobot v3 (remux, no re-encode).",
    )
    parser.add_argument(
        "repo_id",
        help="<namespace>/<name>, or a bare name (-> <default namespace>/<name>)",
    )
    parser.add_argument(
        "--namespace", default=None,
        help="namespace a bare name resolves into (default: the config's "
             "datasets.default_namespace, bc_demo)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output dir (default <root>/exports/lerobot_v3)"
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="ONE generic <root>/<ns>/<name> datasets root, ignoring the config's namespace "
             "map (default: the runtime config's generic root + per-namespace roots)",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="runtime YAML (default $APOLLO_CONFIG)"
    )
    parser.add_argument("--video-file-mb", type=int, default=None)
    parser.add_argument("--data-file-mb", type=int, default=None)
    parser.add_argument(
        "--no-validate", action="store_true",
        help="skip the LeRobotDataset re-open (keeps torch out of the process)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    store = _dataset_store(args)
    repo_id = store.resolve(args.repo_id)
    root = store.root_of(repo_id)
    if not root.is_dir():
        print(f"error: no dataset at {root}", file=sys.stderr)
        return 2
    caps = {}
    if args.video_file_mb is None or args.data_file_mb is None:
        from ..config import ExportConfig

        defaults = ExportConfig()
        caps["video_file_mb"] = args.video_file_mb or defaults.video_file_mb
        caps["data_file_mb"] = args.data_file_mb or defaults.data_file_mb
    else:
        caps = {"video_file_mb": args.video_file_mb, "data_file_mb": args.data_file_mb}
    progress = ExportProgress(repo_id)
    try:
        result = export_lerobot_v3(
            root, repo_id, args.out, progress=progress, validate=not args.no_validate, **caps
        )
    except ExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - the CLI reports and exits non-zero
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    skipped = f" ({len(result.skipped)} skipped: export_ok false)" if result.skipped else ""
    print(
        f"exported {result.episodes} episodes / {result.frames} frames of {repo_id} "
        f"-> {result.path}{skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
