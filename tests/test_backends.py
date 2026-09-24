import pytest

from laya_universal.backends import (
    _BACKENDS,
    choose_backend,
    get_backend,
    onnx_weight_file,
    select_backend,
)


class Fake:
    def __init__(self, name, avail=True):
        self._name, self._avail = name, avail

    @property
    def name(self):
        return self._name

    def available(self):
        return self._avail


@pytest.fixture
def registry():
    saved = dict(_BACKENDS)
    _BACKENDS.clear()
    yield _BACKENDS
    _BACKENDS.clear()
    _BACKENDS.update(saved)


def _dir_with(tmp_path, *names):
    d = tmp_path / "ckpt"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")
    return d


def test_onnx_checkpoint_prefers_onnx_backend(tmp_path, registry):
    registry["onnx-cpu"] = Fake("onnx-cpu")
    registry["onnx-directml"] = Fake("onnx-directml")
    assert choose_backend(_dir_with(tmp_path, "model.onnx")).name.startswith("onnx-")


def test_safetensors_without_mlx_is_a_clear_error(tmp_path, registry):
    registry["onnx-cpu"] = Fake("onnx-cpu")
    with pytest.raises(RuntimeError, match="convert"):
        choose_backend(_dir_with(tmp_path, "model.safetensors"))


def test_safetensors_with_mlx(tmp_path, registry):
    registry["mlx"] = Fake("mlx")
    assert choose_backend(_dir_with(tmp_path, "model.safetensors")).name == "mlx"


def test_onnx_checkpoint_on_apple_falls_back_to_onnx(tmp_path, registry):
    # MLX is the machine's best backend, but it cannot read .onnx.
    registry["mlx"] = Fake("mlx")
    registry["onnx-coreml"] = Fake("onnx-coreml")
    assert choose_backend(_dir_with(tmp_path, "model.onnx")).name == "onnx-coreml"


def test_checkpoint_without_weights_is_rejected(tmp_path, registry):
    registry["onnx-cpu"] = Fake("onnx-cpu")
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        choose_backend(d)


def test_preferred_alias_bypasses_format_check(tmp_path, registry):
    registry["onnx-directml"] = Fake("onnx-directml")
    d = _dir_with(tmp_path, "model.safetensors")
    assert choose_backend(d, "dml").name == "onnx-directml"
    assert select_backend("directml").name == "onnx-directml"


def test_select_backend_with_no_backends(registry):
    with pytest.raises(RuntimeError, match="onnxruntime"):
        select_backend()


def test_get_backend_unknown_name():
    with pytest.raises(ValueError, match="laya-universal info"):
        get_backend("definitely-not-a-backend")


def test_onnx_weight_file_lookup(tmp_path):
    assert onnx_weight_file(tmp_path) is None
    (tmp_path / "laya.onnx").write_bytes(b"x")
    assert onnx_weight_file(tmp_path) == "laya.onnx"
