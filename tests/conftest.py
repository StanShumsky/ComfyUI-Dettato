"""Stubs for the ComfyUI modules the nodes import, so tests run without a ComfyUI install."""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
STATE = {
    "output_dir": tempfile.gettempdir(),
    "temp_dir": str(Path(tempfile.gettempdir()) / "dettato-temp"),
    "models_dir": os.environ.get("DET_MODELS_DIR", str(Path(tempfile.gettempdir()) / "dettato-models")),
}


def _install_comfy_api_stub() -> None:
    """The real comfy_api.latest drags in the whole ComfyUI runtime; the tests only need the io
    surface the nodes use, so install a minimal stand-in when the real one is not importable."""
    try:
        import comfy_api.latest  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass

    class _Base:
        def __init__(self, id=None, display_name=None, tooltip=None, optional=False, default=None, **kwargs):
            self.id, self.display_name, self.tooltip, self.optional = id, display_name, tooltip, optional
            self.default, self.kwargs = default, kwargs

    def _custom(io_type):
        t = io_type  # class bodies do not close over function locals; bind under a plain name

        class _CustomInput(_Base):
            io_type = t

        class _CustomOutput(_Base):
            io_type = t

        outer = type("Custom", (), {"io_type": t})
        outer.Input = _CustomInput
        outer.Output = _CustomOutput
        return outer

    class _Audio:
        Input = _Base

    class _Boolean:
        Input = _Base

    class _Float:
        Input = _Base

    class _Combo:
        class Input(_Base):
            def __init__(self, id=None, options=None, display_name=None, tooltip=None,
                         optional=False, default=None, **kwargs):
                super().__init__(id, display_name, tooltip, optional, **kwargs)
                self.options = options or []
                self.default = default

    class _Int:
        Input = _Base

    class _String:
        Input = _Base
        Output = _Base

    class _Image:
        Output = _Base

    class _TemplatePrefix:
        def __init__(self, input, prefix, min=1, max=10):
            self.input, self.prefix, self.min, self.max = input, prefix, min, max
            self.names = [f"{prefix}{i}" for i in range(max)]

    class _Autogrow:
        TemplatePrefix = _TemplatePrefix

        class Input(_Base):
            def __init__(self, id=None, template=None, display_name=None, tooltip=None, optional=False, **kwargs):
                super().__init__(id, display_name, tooltip, optional, **kwargs)
                self.template = template

    class _Schema:
        def __init__(self, node_id=None, display_name=None, category=None, description=None,
                     inputs=None, outputs=None, **kwargs):
            self.node_id, self.display_name, self.category = node_id, display_name, category
            self.description = description
            self.inputs, self.outputs = inputs or [], outputs or []
            # mirror the real Schema finalization: outputs without ids get default ids
            for i, output in enumerate(self.outputs):
                if getattr(output, "id", None) is None:
                    output.id = f"_{i}_{getattr(output, 'io_type', 'X')}_"

    class _ComfyNode:
        pass

    class _NodeOutput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.ui = kwargs.get("ui")

        @property
        def result(self):
            return self.args if len(self.args) > 0 else None

    io = types.SimpleNamespace(
        Custom=_custom, Audio=_Audio, Boolean=_Boolean, Combo=_Combo, Float=_Float,
        Image=_Image, Int=_Int, String=_String, Autogrow=_Autogrow,
        Schema=_Schema, ComfyNode=_ComfyNode, NodeOutput=_NodeOutput,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ComfyExtension = type("ComfyExtension", (), {})
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest


def _install_stubs() -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = STATE["models_dir"]
    folder_paths.add_model_folder_path = lambda *a, **k: None
    folder_paths.get_folder_paths = lambda name: [STATE["models_dir"]]
    folder_paths.get_output_directory = lambda: STATE["output_dir"]

    def get_temp_directory():
        Path(STATE["temp_dir"]).mkdir(parents=True, exist_ok=True)
        return STATE["temp_dir"]

    folder_paths.get_temp_directory = get_temp_directory

    def get_save_image_path(prefix, output_dir, width, height):
        sub = Path(prefix).parent
        subfolder = "" if str(sub) == "." else str(sub)
        filename = Path(prefix).name
        folder = Path(output_dir) / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        counter = len(list(folder.glob(filename + "_*.png")))
        return str(folder), filename, counter, subfolder, prefix

    folder_paths.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mm.soft_empty_cache = lambda: None
    cu = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.done = 0

        def update(self, n):
            self.done += n

    cu.ProgressBar = ProgressBar
    comfy.model_management = mm
    comfy.utils = cu
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = cu


_install_comfy_api_stub()
_install_stubs()
_spec = importlib.util.spec_from_file_location(
    "dettanode", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["dettanode"] = _pkg
_spec.loader.exec_module(_pkg)
# pytest's Package.setup imports the repo-root __init__.py as top-level "__init__"
# (the hyphenated directory name is not a valid package identifier); reuse our
# correctly-loaded package instead of letting it re-import without a parent package.
sys.modules.setdefault("__init__", _pkg)


@pytest.fixture(scope="session")
def nodes():
    return sys.modules["dettanode.nodes"]


@pytest.fixture
def output_dir(tmp_path):
    STATE["output_dir"] = str(tmp_path)
    STATE["temp_dir"] = str(tmp_path / "temp")
    return tmp_path


@pytest.fixture
def models_dir(tmp_path):
    """A throwaway models dir, also pointed at by the folder_paths stub."""
    STATE["models_dir"] = str(tmp_path / "models")
    Path(STATE["models_dir"]).mkdir(parents=True, exist_ok=True)
    return Path(STATE["models_dir"])


def _make_audio(t=2, rate=48000, freq=440.0):
    """A (1, 1, N) stereo-free ComfyUI AUDIO payload: one sine at `freq` Hz."""
    n = int(t * rate)
    x = torch.sin(2 * 3.141592653589793 * freq * torch.arange(n) / rate).float()
    return {"waveform": x[None, None], "sample_rate": rate}


@pytest.fixture
def make_audio():
    return _make_audio


@pytest.fixture
def make_track():
    import numpy as np

    from dettanode.dettato.pipeline import Track

    def build(name="piano", program=0, notes=((0.0, 0.25, 60, 90.0), (0.5, 0.9, 64, 127.0)), is_drum=False):
        return Track(name, program, np.array(notes, np.float64).reshape(-1, 4), is_drum=is_drum)

    return build


@pytest.fixture
def make_tracks(make_track):
    from dettanode.track_io import make_tracks_payload

    def build(tracks=None, seconds=8.0, info=None):
        tracks = tracks if tracks is not None else [
            make_track("Piano", 0, ((0.0, 0.25, 60, 90.0), (0.5, 0.9, 64, 127.0))),
            make_track("Drums", 0, ((0.0, 0.1, 36, 127.0), (0.5, 0.6, 38, 99.0)), is_drum=True),
        ]
        return make_tracks_payload(tracks, seconds, "audio", info if info is not None else {"drums": "skipped: unit test"})

    return build