"""Tests for stereo backend detection and Fast-FS wrapper integration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

from volrecon.stereo.foundation_stereo_wrapper import FoundationStereoConfig, FoundationStereoWrapper
from volrecon.stereo.stereo_backends import (
    detect_stereo_backend,
    is_fast_foundation_stereo_repo,
    prepare_stereo_repo,
    require_cfg_yaml,
    resolve_checkpoint_path,
    resolve_repo_path,
)


def test_detect_fast_fs_from_serialize_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "model_best_bp2_serialize.pth"
    ckpt.touch()
    assert detect_stereo_backend(tmp_path, ckpt, "auto") == "fast_foundation_stereo"


def test_detect_classic_from_ckpt_name(tmp_path: Path):
    ckpt = tmp_path / "model_best_bp2.pth"
    ckpt.touch()
    repo = tmp_path / "FoundationStereo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_demo.py").write_text(
        "parser.add_argument('--ckpt_dir')\nclass FoundationStereo: pass\n",
        encoding="utf-8",
    )
    assert detect_stereo_backend(repo, ckpt, "auto") == "foundation_stereo"


def test_is_fast_fs_repo_from_run_demo(tmp_path: Path):
    repo = tmp_path / "Fast-FoundationStereo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run_demo.py").write_text(
        "parser.add_argument('--model_dir', default='weights/23-36-37/model_best_bp2_serialize.pth')\n",
        encoding="utf-8",
    )
    assert is_fast_foundation_stereo_repo(repo)


def test_resolve_checkpoint_directory(tmp_path: Path):
    d = tmp_path / "20-30-48"
    d.mkdir()
    (d / "cfg.yaml").write_text("vit_size: vitb\n", encoding="utf-8")
    pth = d / "model_best_bp2_serialize.pth"
    pth.write_text("", encoding="utf-8")
    resolved = resolve_checkpoint_path(d, "fast_foundation_stereo")
    assert resolved == pth.resolve()


def test_require_cfg_yaml_raises(tmp_path: Path):
    ckpt = tmp_path / "model_best_bp2_serialize.pth"
    ckpt.touch()
    with pytest.raises(FileNotFoundError, match="cfg.yaml"):
        require_cfg_yaml(ckpt)


def test_resolve_repo_path_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    nested = tmp_path / "Fast-FoundationStereo"
    nested.mkdir()
    assert resolve_repo_path(Path("~/Fast-FoundationStereo")) == nested.resolve()


def test_prepare_stereo_repo_requires_core_package(tmp_path: Path):
    repo = tmp_path / "Fast-FoundationStereo"
    repo.mkdir()
    (repo / "Utils.py").write_text("AMP_DTYPE = None\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="core/utils/utils.py"):
        prepare_stereo_repo(repo, backend="fast_foundation_stereo")


def test_prepare_stereo_repo_adds_repo_to_sys_path(tmp_path: Path):
    repo = tmp_path / "Fast-FoundationStereo"
    (repo / "core" / "utils").mkdir(parents=True)
    (repo / "core" / "utils" / "utils.py").write_text("class InputPadder: pass\n", encoding="utf-8")
    (repo / "Utils.py").write_text("AMP_DTYPE = None\n", encoding="utf-8")
    resolved = prepare_stereo_repo(repo, backend="fast_foundation_stereo")
    assert resolved == repo.resolve()
    assert str(resolved) in sys.path


def test_disable_torch_compile_disables_dynamo(monkeypatch: pytest.MonkeyPatch):
    from volrecon.stereo import fast_fs_inference

    fast_fs_inference._disable_torch_compile(force=True)
    import torch._dynamo as dynamo

    assert dynamo.config.disable is True
    fn = lambda x: x
    assert fast_fs_inference._unwrap_torch_compiled(torch.compile(fn)) is fn


def test_wrapper_builds_fast_fs_subprocess_command(tmp_path: Path):
    repo = tmp_path / "Fast-FoundationStereo"
    weights = repo / "weights" / "20-30-48"
    weights.mkdir(parents=True)
    ckpt = weights / "model_best_bp2_serialize.pth"
    ckpt.write_text("", encoding="utf-8")
    (weights / "cfg.yaml").write_text("valid_iters: 8\nmax_disp: 192\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_demo.py").write_text(
        "parser.add_argument('--model_dir', default='model_best_bp2_serialize.pth')\n",
        encoding="utf-8",
    )

    cfg = FoundationStereoConfig(
        foundationstereo_repo=repo,
        ckpt=weights,
        backend="fast_foundation_stereo",
        valid_iters=4,
        max_disp=192,
        use_subprocess=True,
    )
    wrapper = FoundationStereoWrapper(cfg)
    assert wrapper.backend == "fast_foundation_stereo"
    assert wrapper.ckpt_file.name == "model_best_bp2_serialize.pth"

    with mock.patch.object(FoundationStereoWrapper, "_load_disparity", return_value=__import__("numpy").zeros((2, 2))):
        with mock.patch("volrecon.stereo.foundation_stereo_wrapper.subprocess.run") as run_mock:
            disp = wrapper._run_fast_fs_subprocess(
                tmp_path / "l.png",
                tmp_path / "r.png",
                tmp_path / "sub_out",
            )
            assert disp.shape == (2, 2)
            cmd = run_mock.call_args[0][0]
            assert "-m" in cmd
            assert "volrecon.stereo.fast_fs_inference" in cmd
            assert "--model_dir" in cmd
            assert str(ckpt) in cmd
