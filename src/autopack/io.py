import ctypes
import pathlib
import pickle

import xarray as xr

from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import load_scene, save_scene


def save_dataset(ds, path):
    with open(path, "wb") as file:
        pickle.dump(ds, file)
    return path


def load_dataset(path):
    with open(path, "rb") as file:
        ds = pickle.load(file)
    return ds


def save_session(dataset: xr.Dataset, ips: IPSInstance, session_dir: pathlib.Path):
    session_dir.mkdir(exist_ok=True)
    save_scene(ips, session_dir / "scene.ips")
    save_dataset(dataset, session_dir / "dataset.pkl")
    session_marker = session_dir / ".autopack-session"
    session_marker.touch()

    # Hide the session marker file
    ctypes.windll.kernel32.SetFileAttributesW(
        str(pathlib.PureWindowsPath(session_marker)), 0x02
    )


def load_session(ips: IPSInstance, session_dir: pathlib.Path):
    session_marker = session_dir / ".autopack-session"
    if not session_marker.exists():
        raise ValueError(f"{session_dir} is not a complete Autopack session directory")

    dataset = load_dataset(session_dir / "dataset.pkl")
    load_scene(ips, session_dir / "scene.ips", clear=True)

    return dataset
