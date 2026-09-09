"""The Online DAgger skill ships as runtime package data (15-online-dagger §9, D8: served by
REST, mirrored byte-for-byte into the policy-node repo) and the ``online_dagger:`` runtime
config block (§7) loads from both shipped YAMLs with the model defaults."""

from __future__ import annotations

import io
import tarfile
from importlib.resources import files
from pathlib import Path

import pytest
from pydantic import ValidationError

from apollo_mavis_v2_runtime.config import OnlineDaggerRuntimeConfig, load_runtime_config
from apollo_mavis_v2_runtime.online_dagger import (
    PACKAGED_SKILL_DIR,
    SKILL_FILE,
    SKILL_NAME,
    skill_dir,
    skill_markdown,
    skill_tarball,
)

REPO = Path(__file__).resolve().parents[1]
MIRROR = (
    REPO.parents[1]
    / "apollo-mavis-v2-ws-p12"
    / "apollo-mavis-v2-policy-node"
    / "skills"
    / SKILL_NAME
)
MEMBERS = ("SKILL.md", "references/contract.md", "references/pro-dagger-example.md")


def test_skill_is_importlib_resources_package_data():
    root = files("apollo_mavis_v2_runtime.online_dagger") / "skill"
    assert (root / SKILL_FILE).is_file()
    assert {p.name for p in (root / "references").iterdir()} == {
        "contract.md",
        "pro-dagger-example.md",
    }
    assert Path(str(root)) == PACKAGED_SKILL_DIR == skill_dir()
    assert SKILL_NAME == "mavis-online-dagger-trainer"
    text = skill_markdown()
    assert text.startswith("---\nname: mavis-online-dagger-trainer\ndescription: ")
    assert text == (root / SKILL_FILE).read_text(encoding="utf-8")
    for needle in (
        "trainer_status",
        "episode_saved",
        "episode_discarded",
        "train_now",
        "mavis-policy-node",
        "--online-dagger",
        "swap_weights",
        "train_if_due",
    ):
        assert needle in text, needle
    contract = (PACKAGED_SKILL_DIR / "references" / "contract.md").read_text(encoding="utf-8")
    for needle in (
        '"online_dagger"',
        "waiting for the trainer to report ready",
        "training in progress",
        "no Online DAgger trainer attached",
        "episode_discarded",
        "session_id",
    ):
        assert needle in contract, needle
    # the shell is algorithm-agnostic: the contract lists exactly the ten wire event kinds
    from apollo_mavis_v2_core.protocol.external import EVENT_KINDS

    assert ", ".join(EVENT_KINDS) in contract.replace("\n", " ")


def test_skill_tarball_is_rooted_at_the_skill_name(tmp_path):
    data = skill_tarball()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
        assert names == [f"{SKILL_NAME}/{m}" for m in MEMBERS]  # deterministic order
        for m in MEMBERS:
            assert tar.extractfile(f"{SKILL_NAME}/{m}").read() == (
                PACKAGED_SKILL_DIR / m
            ).read_bytes()
    # `tar xz -C ~/.claude/skills/` lands the directory in place
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(tmp_path, filter="data")
    assert (tmp_path / SKILL_NAME / SKILL_FILE).read_text(encoding="utf-8") == skill_markdown()
    # a skill_dir override that does not exist is an OSError (REST maps it to 404)
    with pytest.raises(FileNotFoundError):
        skill_markdown(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        skill_tarball(tmp_path / "missing")
    assert skill_dir(tmp_path / "alt") == tmp_path / "alt"


@pytest.mark.skipif(not MIRROR.is_dir(), reason="policy-node mirror not checked out")
def test_skill_mirror_in_the_policy_node_repo_is_byte_identical():
    for m in MEMBERS:
        assert (PACKAGED_SKILL_DIR / m).read_bytes() == (MIRROR / m).read_bytes(), m


def test_online_dagger_runtime_config_block():
    for name in ("mavis_v2.yaml", "sim.yaml"):
        cfg = load_runtime_config(REPO / "configs" / name)
        assert cfg.online_dagger == OnlineDaggerRuntimeConfig(), name  # shipped == defaults
        assert cfg.online_dagger.skill_dir is None and cfg.online_dagger.session_file_hz == 1.0
    c = OnlineDaggerRuntimeConfig(skill_dir="~/skills/x", session_file_hz=0.5)
    assert c.skill_dir == Path("~/skills/x").expanduser() and c.session_file_hz == 0.5
    with pytest.raises(ValidationError):
        OnlineDaggerRuntimeConfig(session_file_hz=0)
