from PyInstaller.building.build_main import COLLECT, EXE, PYZ, Analysis

block_cipher = None

PROJECT_NAME = "autopack"
# VERSION =
# BUNDLE_NAME = f"{PROJECT_NAME}-v{VERSION}"
BUNDLE_NAME = PROJECT_NAME

a = Analysis(
    ["src/autopack/gui_entrypoint.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "ipykernel",
        "IPython",
        "matplotlib.backends",
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
