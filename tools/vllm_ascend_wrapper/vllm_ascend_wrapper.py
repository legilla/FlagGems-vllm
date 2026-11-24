"""PEP 517 backend that builds vllm-ascend from the vendor sources.

The published ``vllm-ascend`` wheel is a CPython extension compiled against one
exact torch build: its metadata pins ``torch==2.10.0`` and its ``_C_ascend``
module references ``torch::Library::_def`` signatures that torch 2.13 no longer
exports, so the wheel cannot even be loaded on the torch every CANN 9.2.1 image
ships. This backend clones the vendor repository at the pinned revision, relaxes
the single torch version gate in its CMakeLists, builds both the C++ extension
and the AscendC kernels against the interpreter that runs the build, and
republishes the result under the upstream distribution name.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

NAME = "vllm-ascend"
VERSION = "0.32.0rc1"
REPOSITORY = "https://github.com/vllm-project/vllm-ascend"
REVISION = "24487f4f48ee99cf45d86d1c504e645549b8c1ef"
TAG = f"v{VERSION}"
CATLASS_REPOSITORY = "https://gitcode.com/cann/catlass"
CATLASS_REVISION = "41bf90da655bba3c66d0acd7e00abe33960ecfd6"
# CANN installs the kernel objects under their absolute source path and its
# op-compiler helper globs them with ``**``, which never enters a dot-directory:
# a checkout under ``~/.cache`` installs them to ``.../root/.cache/...`` and the
# lookup then finds nothing.
CHECKOUT = pathlib.Path("~/flaggems/src/vllm-ascend").expanduser()
BUILD_PREREQS = {
    "attr": "attrs",
    "decorator": "decorator",
    "pybind11": "pybind11",
    "setuptools_scm": "setuptools-scm>=8",
    "wheel": "wheel",
}
PYPI_INDEX_URL = "https://resource.flagos.net/repository/pypi-proxy/simple"
TORCH_GATE = """# check torch version is 2.10.0
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.10.0")
  message(FATAL_ERROR "Expected PyTorch version 2.10.0, but found ${TORCH_VERSION}")
endif()
"""
REQUIRES = (
    "compressed-tensors>=0.11.0",
    "decorator",
    "einops",
    "fastapi>=0.133.0,<0.137.0",
    "fastokens==0.2.0",
    "memcache-hybrid==1.3.0",
    "memfabric-hybrid==1.3.0",
    "msgpack",
    "numba",
    "packaging",
    "pandas<3",
    "pandas-stubs",
    "psutil",
    "pybind11",
    "pyyaml",
    "quart",
    "regex",
    "scipy",
    "setuptools-scm>=8",
    "wheel",
)


def _log(message: str) -> None:
    print(f"vllm-ascend: {message}", file=sys.stderr, flush=True)


def _run(command: list[str], cwd: pathlib.Path | None = None) -> None:
    _log(" ".join(command))
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=CHECKOUT, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_tree() -> pathlib.Path:
    """Reuse the checkout only while it is the pinned revision and untouched."""
    if CHECKOUT.exists():
        head = _git_output("rev-parse", "HEAD")
        # Undo the one patch this backend applies so its own edit does not count.
        _git_output("checkout", "--", "CMakeLists.txt")
        status = _git_output(
            "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"
        )
        if head == REVISION and not status:
            _log(f"reusing {CHECKOUT}")
            return CHECKOUT
        _log(
            f"removing {CHECKOUT}: head={head or '<none>'} status={status.splitlines()[:3]}"
        )
        shutil.rmtree(CHECKOUT, ignore_errors=True)
    CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 9):
        _log(f"cloning {REPOSITORY} at {TAG} (attempt {attempt}/8)")
        shutil.rmtree(CHECKOUT, ignore_errors=True)
        clone = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                TAG,
                REPOSITORY,
                str(CHECKOUT),
            ],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if clone.returncode == 0:
            return CHECKOUT
        if attempt == 8:
            break
        time.sleep(10 * attempt)
    shutil.rmtree(CHECKOUT, ignore_errors=True)
    raise RuntimeError(f"could not clone {REPOSITORY} at {TAG}")


def _catlass(source: pathlib.Path) -> None:
    """The AscendC kernel build needs the catlass submodule headers."""
    target = source / "csrc" / "third_party" / "catlass"
    if (target / "include").is_dir():
        return
    _log("fetching catlass")
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--quiet", CATLASS_REPOSITORY, str(target)])
    _run(["git", "checkout", "--quiet", CATLASS_REVISION], cwd=target)


def _relax_torch_gate(source: pathlib.Path) -> None:
    """The gate pins 2.10.0; the image decides which torch is installed."""
    path = source / "CMakeLists.txt"
    text = path.read_text(encoding="utf-8")
    if TORCH_GATE not in text:
        return
    path.write_text(
        text.replace(
            TORCH_GATE,
            "# the vendor gate pins 2.10.0; build against the installed torch\n",
        ),
        encoding="utf-8",
    )
    _log("relaxed the CMakeLists torch version gate")


def _ensure_build_prereqs() -> None:
    """The CANN op compiler imports ``attr`` and ``decorator`` while it runs."""
    for module, package in BUILD_PREREQS.items():
        if importlib.util.find_spec(module) is not None:
            continue
        index = next(
            (
                os.environ[name]
                for name in ("PIP_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX_URL")
                if os.environ.get(name)
            ),
            PYPI_INDEX_URL,
        )
        _log(f"installing {package} for the CANN op compiler")
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--index-url",
                index,
                package,
            ]
        )


def _vendor_wheel(source: pathlib.Path) -> pathlib.Path:
    output = pathlib.Path(tempfile.mkdtemp(prefix="vllm-ascend-wheel-"))
    env = dict(os.environ)
    env["COMPILE_CUSTOM_KERNELS"] = "1"
    env.setdefault("CMAKE_BUILD_TYPE", "Release")
    env["SETUPTOOLS_SCM_PRETEND_VERSION"] = VERSION
    # AscendC links the kernels before the host objects it collects exist when
    # the job count matches the core count; one build per core is too many.
    env.setdefault("MAX_JOBS", "16")
    _log("building the C++ extension and the AscendC kernels")
    _run(
        [sys.executable, "-u", "setup.py", "bdist_wheel", "--dist-dir", str(output)],
        cwd=source,
    )
    wheels = sorted(output.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"no wheel produced in {output}")
    return wheels[0]


def _metadata() -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {NAME}",
        f"Version: {VERSION}",
        "Summary: vllm-ascend built from the vendor sources",
    ]
    lines.extend(f"Requires-Dist: {requirement}" for requirement in REQUIRES)
    return "\n".join(lines) + "\n"


def _wheel(tags: list[str]) -> str:
    lines = [
        "Wheel-Version: 1.0",
        "Generator: vllm_ascend_wrapper",
        "Root-Is-Purelib: false",
    ]
    lines.extend(f"Tag: {tag}" for tag in tags)
    return "\n".join(lines) + "\n"


def _dist_info() -> str:
    return f"{NAME.replace('-', '_')}-{VERSION}.dist-info"


def _record_row(path: str, data: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return [path, f"sha256={digest.decode()}", str(len(data))]


def _vendor_tags(vendor: pathlib.Path) -> list[str]:
    with zipfile.ZipFile(vendor) as archive:
        wheel_file = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        text = archive.read(wheel_file[0]).decode("utf-8")
    return [
        line.split(": ", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("Tag: ")
    ]


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _log("hook: prepare_metadata_for_build_wheel")
    directory = pathlib.Path(metadata_directory) / _dist_info()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "METADATA").write_text(_metadata(), encoding="utf-8")
    (directory / "WHEEL").write_text(_wheel(["py3-none-any"]), encoding="utf-8")
    return _dist_info()


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _log("hook: build_wheel")
    _ensure_build_prereqs()
    source = _source_tree()
    _catlass(source)
    _relax_torch_gate(source)
    vendor = _vendor_wheel(source)
    tags = _vendor_tags(vendor)
    target = (
        pathlib.Path(wheel_directory)
        / f"{NAME.replace('-', '_')}-{VERSION}-{tags[0]}.whl"
    )
    dist_info = _dist_info()
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    with (
        zipfile.ZipFile(vendor) as source_archive,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out,
    ):
        for member in source_archive.infolist():
            if ".dist-info/" in member.filename:
                continue
            data = source_archive.read(member)
            out.writestr(member, data)
            writer.writerow(_record_row(member.filename, data))
        for name, text in (
            ("METADATA", _metadata()),
            ("WHEEL", _wheel(tags)),
            ("INSTALLER", "uv\n"),
        ):
            data = text.encode("utf-8")
            out.writestr(f"{dist_info}/{name}", data)
            writer.writerow(_record_row(f"{dist_info}/{name}", data))
        writer.writerow([f"{dist_info}/RECORD", "", ""])
        out.writestr(f"{dist_info}/RECORD", record.getvalue())
    return target.name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    raise NotImplementedError("vllm_ascend_wrapper only distributes a wheel")


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError("vllm_ascend_wrapper only distributes a wheel")
