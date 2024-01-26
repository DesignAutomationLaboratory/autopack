from PyInstaller.building.build_main import COLLECT, EXE, PYZ, Analysis
from PyInstaller.utils.hooks import collect_data_files, conda_support


def collect_nvidia_libs(*args, **kwargs):
    """
    Nasty hack to collect nvidia-packaged libs from the conda
    environment, that are stored in a non-standard location
    (<conda_env>/bin).
    """
    old_lib_dir = conda_support.lib_dir
    conda_support.lib_dir = conda_support.PackagePath("bin")
    libs = conda_support.collect_dynamic_libs(*args, **kwargs)
    conda_support.lib_dir = old_lib_dir
    return libs


block_cipher = None

PROJECT_NAME = "autopack"
# VERSION =
# BUNDLE_NAME = f"{PROJECT_NAME}-v{VERSION}"
BUNDLE_NAME = PROJECT_NAME

a = Analysis(
    ["src/autopack/gui_entrypoint.py"],
    pathex=[],
    binaries=[
        # Needed for CUDA support in PyTorch
        *collect_nvidia_libs("cuda-nvrtc-dev"),
    ],
    datas=[
        *collect_data_files("autopack"),
        # The PyViz packages determines version on runtime and needs a
        # data file to do so. Ugh.
        *collect_data_files("holoviews", excludes=["examples", "tests"]),
        *collect_data_files("hvplot", excludes=["examples"]),
        *collect_data_files("param"),
        # torch.jit fails if we don't have the sources for linear_operator
        *collect_data_files("linear_operator", include_py_files=True),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "ipykernel",
        "IPython",
        "nbconvert",
        "nbformat",
        "notebook",
        "PyQt5",
        "pytest",
        "win32com",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=PROJECT_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    # Tree("docs/_build/html", "docs"),
    strip=False,
    upx=False,
    upx_exclude=[],
    name=BUNDLE_NAME,
)
