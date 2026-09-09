"""The shipped ``configs/*.yaml`` load and carry the lab facts (2026-09-04):
two control boxes (Manipulation Arm ``grip`` 192.168.1.201 with the force-capable
Gripper G2, Perception Arm ``view`` 192.168.2.219 with the microphone), camera
paths still placeholders, user-facing microphone label."""

from __future__ import annotations

from pathlib import Path

import pytest

from apollo_mavis_v2_runtime.config import (
    ControlConfig,
    DatasetNamespaceConfig,
    DatasetsConfig,
    GelloConfig,
    RuntimeConfig,
    load_runtime_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_mavis_v2_hardware_workcell_matches_the_lab():
    cfg = load_runtime_config(CONFIGS / "mavis_v2.yaml")
    hw = cfg.workcells["hardware"]
    assert hw.kind == "hardware" and hw.digital_twin_scene == "mavis_v2"
    by_id = {a.id: a for a in hw.arms}
    assert [a.id for a in hw.arms] == ["grip", "view"]  # Manipulation Arm first
    assert (by_id["grip"].ip, by_id["grip"].gripper) == ("192.168.1.201", "xarm_g2")
    assert (by_id["view"].ip, by_id["view"].gripper) == ("192.168.2.219", "none")
    assert by_id["view"].microphone is True and by_id["grip"].microphone is False
    # Phase-09b controller-side backstops (PROVISIONAL payloads until the tools are
    # weighed): Manipulation Arm = Gripper G2 + D435i + mount, Perception Arm = D435i +
    # RØDE NT-USB Mini + mount; collision sensitivity 3 on both, no reduced-mode box.
    assert (by_id["grip"].tcp_load_kg, by_id["grip"].tcp_load_cog_mm) == (0.95, (0.0, 0.0, 60.0))
    assert (by_id["view"].tcp_load_kg, by_id["view"].tcp_load_cog_mm) == (0.55, (0.0, 0.0, 90.0))
    assert by_id["grip"].collision_sensitivity == by_id["view"].collision_sensitivity == 3
    assert all(a.reduced_tcp_boundary_mm is None and a.expected_sn is None for a in hw.arms)
    assert [c.id for c in hw.cameras] == ["grip_wrist", "view_wrist"]
    # RealSense D435i colour over UVC: by USB serial (no by-id path), YUYV only.
    assert all(c.kind == "v4l2" and c.device_path is None and c.serial for c in hw.cameras)
    assert {c.fourcc for c in hw.cameras} == {"YUYV"}
    assert cfg.microphone.enabled and cfg.microphone.mic_id == "mic_view"
    assert cfg.microphone.label == "Perception Arm microphone"
    assert cfg.hardware_probe.enabled and cfg.hardware_probe.port == 502
    # Phase-09a: D435i COLOUR intrinsics per wrist camera (rs-enumerate-devices -c),
    # read-only monitor + twin overlay blocks.
    intr = {c.id: c.intrinsics for c in hw.cameras}
    assert (intr["grip_wrist"].fx, intr["grip_wrist"].fy) == (608.19, 608.23)
    assert (intr["grip_wrist"].cx, intr["grip_wrist"].cy) == (327.39, 247.90)
    assert (intr["view_wrist"].fx, intr["view_wrist"].fy) == (606.36, 606.38)
    assert (intr["view_wrist"].cx, intr["view_wrist"].cy) == (311.90, 249.45)
    mon = cfg.hardware_monitor
    assert (mon.enabled, mon.poll_hz, mon.stale_s, mon.reconnect_s) == (True, 10.0, 0.5, 2.0)
    ov = cfg.twin_overlay
    assert ov.enabled and ov.fps == 12.0 and ov.alpha == 0.5 and ov.stream_suffix == "_align"
    assert ov.tint_rgb == (255, 235, 140) and ov.edge_rgb == (255, 220, 60)
    assert ov.env_outline and ov.env_rgb == (90, 200, 250)
    assert ov.stale_tint_rgb == (170, 170, 170)
    assert ov.joint1_offset_rad == 0.0 and ov.rail_flip is False
    assert ov.rail_fallback_m == {"grip": 0.65, "view": 0.0}
    # Phase-09c/09d: first live runs at 10 % (both arms always in the session, so no
    # arm pre-selection key); D4 sweep margins; rail_flip lives here now (the overlay
    # key is an alias) and stays false until the first homing is checked against the
    # *_align overlay.
    hs = cfg.hardware_session
    assert hs.default_speed_scale == 1.0 and not hasattr(hs, "default_arms")  # 100 % (2026-09-08)
    assert hs.rail_flip is False and ov.rail_flip is False
    assert (hs.home_rail_inflation_m, hs.home_rail_step_m) == (0.025, 0.005)
    assert hs.bringup_timeout_s == 60.0
    assert cfg.tracker.backend == "fake"  # the live config switches to libsurvive
    sim = cfg.workcells["sim"]
    assert {a.id for a in sim.arms} == {"view", "grip"} and sim.sim_scene == "mavis_v2"


@pytest.mark.parametrize("name", ["mavis_v2.yaml", "sim.yaml"])
def test_shipped_configs_load(name):
    cfg = load_runtime_config(CONFIGS / name)
    assert cfg.port == 8765


@pytest.mark.parametrize("name", ["mavis_v2.yaml", "sim.yaml"])
def test_keyboard_translate_frame_defaults_to_world(name):
    """Operator decision 2026-09-08 evening (04-runtime §6): the keyboard translate
    keys act in the operator-fixed WORLD frame by default (W away from the operator,
    A to their left, E up); "camera" — that morning's default — and "base" stay
    selectable. Pinned on the model default AND on both shipped configs (mavis_v2.yaml
    states it, sim.yaml inherits it)."""
    assert ControlConfig().translate_frame == "world"
    assert load_runtime_config(CONFIGS / name).control.translate_frame == "world"


def test_shipped_paths_are_self_contained_under_the_workspace():
    """The tracked config anchors every path at ${APOLLO_HOME}/var (04-runtime §14.1):
    no ~/apollo, no absolute machine path. ${APOLLO_HOME} is inferred from the config
    file's own workspace (CONFIGS is inside apollo-mavis-v2-runtime)."""
    # comments may mention ~/apollo (the thing we moved away from); check values only
    values = "\n".join(line.split("#", 1)[0] for line in
                       (CONFIGS / "mavis_v2.yaml").read_text().splitlines())
    assert "~/apollo" not in values and "/home/" not in values
    cfg = load_runtime_config(CONFIGS / "mavis_v2.yaml")
    ws = CONFIGS.resolve().parents[1]  # <ws>/apollo-mavis-v2-runtime/configs -> <ws>
    for p in (cfg.profiles_dir, cfg.datasets_root, cfg.checkpoints_root,
              cfg.calibration_dir, cfg.tracker.libsurvive_config_path,
              cfg.gello.calibration_path):
        assert p.is_absolute() and str(p).startswith(str(ws / "var"))


def test_apollo_home_expansion_and_anchoring(tmp_path, monkeypatch):
    """${APOLLO_HOME} / bare-relative resolve against $APOLLO_HOME; ~ and absolute pass through."""
    monkeypatch.setenv("APOLLO_HOME", str(tmp_path))
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "profiles_dir: ${APOLLO_HOME}/var/profiles\n"
        "datasets_root: relative/datasets\n"
        "checkpoints_root: /absolute/ckpts\n"
        "calibration_dir: ~/cal\n"
    )
    cfg = load_runtime_config(cfg_file)
    assert cfg.profiles_dir == tmp_path / "var" / "profiles"   # ${APOLLO_HOME} expanded
    assert cfg.datasets_root == tmp_path / "relative" / "datasets"  # bare relative anchored
    assert cfg.checkpoints_root == Path("/absolute/ckpts")     # absolute untouched
    assert cfg.calibration_dir == Path("~/cal").expanduser()   # ~ still works


@pytest.mark.parametrize("name", ["mavis_v2.yaml", "sim.yaml"])
def test_shipped_dataset_namespace_roots(name):
    """Operator decision 2026-09-08 (15-online-dagger §0 item 6 / §7 D5): demonstrations
    live in ~/data/bc_demo/<name>, Online DAgger sessions in ~/data/online_dagger/<session>/
    {session.json,rollouts}; a bare dataset name resolves into bc_demo. Both shipped
    configs state the block, and it equals the model defaults (so a config without the
    block behaves the same). ``~`` is expanded like datasets_root; the generic root stays
    inside the workspace."""
    cfg = load_runtime_config(CONFIGS / name)
    ds = cfg.datasets
    assert ds.default_namespace == "bc_demo"
    assert list(ds.namespaces) == ["bc_demo", "online_dagger"]
    home = Path.home()
    assert ds.namespaces["bc_demo"].root == home / "data" / "bc_demo"
    assert ds.namespaces["bc_demo"].subdir is None
    assert ds.namespaces["online_dagger"].root == home / "data" / "online_dagger"
    assert ds.namespaces["online_dagger"].subdir == "rollouts"
    assert all(ns.root.is_absolute() and "~" not in str(ns.root) for ns in ds.namespaces.values())
    od = home / "data" / "online_dagger"
    assert ds.namespaces["online_dagger"].dataset_dir("s1") == od / "s1" / "rollouts"
    assert ds.namespaces["bc_demo"].dataset_dir("pick") == home / "data" / "bc_demo" / "pick"
    assert ds == DatasetsConfig() == RuntimeConfig().datasets  # the block IS the defaults
    ws = CONFIGS.resolve().parents[1]
    assert str(cfg.datasets_root).startswith(str(ws / "var"))  # generic root unchanged


def test_dataset_namespace_roots_expand_like_datasets_root(tmp_path, monkeypatch):
    """${APOLLO_HOME}, ~ and bare-relative roots resolve exactly like the four dirs;
    subdir must be one directory name; namespace keys follow the REST <ns> grammar."""
    monkeypatch.setenv("APOLLO_HOME", str(tmp_path))
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "datasets:\n"
        "  default_namespace: demo\n"
        "  namespaces:\n"
        '    demo: {root: "${APOLLO_HOME}/var/demo"}\n'
        "    rel: {root: relative/rel, subdir: data}\n"
        "    home: {root: ~/h}\n"
    )
    cfg = load_runtime_config(cfg_file)
    ns = cfg.datasets.namespaces
    assert cfg.datasets.default_namespace == "demo"
    assert ns["demo"].root == tmp_path / "var" / "demo"          # ${APOLLO_HOME} expanded
    assert ns["rel"].root == tmp_path / "relative" / "rel"       # bare relative anchored
    assert ns["home"].root == Path("~/h").expanduser()           # ~ still works
    assert ns["rel"].dataset_dir("x") == tmp_path / "relative" / "rel" / "x" / "data"
    with pytest.raises(ValueError, match="single directory name"):
        DatasetNamespaceConfig(root=tmp_path, subdir="a/b")
    with pytest.raises(ValueError, match="single directory name"):
        DatasetNamespaceConfig(root=tmp_path, subdir="..")
    with pytest.raises(ValueError, match="grammar"):
        DatasetsConfig(namespaces={"bad ns": DatasetNamespaceConfig(root=tmp_path)})
    with pytest.raises(ValueError):
        DatasetsConfig(default_namespace="-nope")
    # an empty map + another default is the tests' pre-D4 layout (conftest pins it)
    plain = DatasetsConfig(default_namespace="apollo", namespaces={})
    assert plain.namespaces == {} and plain.default_namespace == "apollo"


@pytest.mark.parametrize("name", ["mavis_v2.yaml", "sim.yaml"])
def test_shipped_gello_block_pins_the_defaults(name):
    """phase-15 (16-gello §9.1): both shipped configs state the GELLO block with backend
    ``none`` (the lab render switches to ``dynamixel``), the hidden kitchen twin as the scene
    the GELLO card launches, the Perception Arm's GELLO hold posture, and the engagement
    numbers; the block equals the model defaults. ``twin_overlay.scene`` stays null (the
    overlays follow the hardware workcell's digital_twin_scene until the lab render says
    otherwise)."""
    cfg = load_runtime_config(CONFIGS / name)
    g = cfg.gello
    assert g.backend == "none" and g.scene_id == "mavis_v2_kitchen"
    assert g.view_posture_rad == [2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029]
    assert g.view_rail_m == 0.0
    assert (g.port, g.usb_serial, g.baud) == ("/dev/ttyUSB0", None, None)
    assert g.joint_ids == [1, 2, 3, 4, 5, 6, 7] and g.gripper_id == 8
    assert g.joint_signs == [1] * 7 and g.joint_offsets_rad is None
    assert (g.poll_hz, g.stale_s, g.max_jump_rad) == (100.0, 0.2, 0.5)
    assert (g.engage_tol_rad, g.leash_rad, g.gripper_quantum, g.max_joint_vel_rad_s) == (
        0.10, 0.80, 0.01, 0.6,
    )
    ws = CONFIGS.resolve().parents[1]
    assert g.calibration_path == ws / "var" / "gello_calibration.json"
    assert g == GelloConfig() == RuntimeConfig().gello  # the block IS the defaults
    assert cfg.twin_overlay.scene is None
