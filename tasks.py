import pathlib
import re

from invoke import Context, task
from invoke.runners import Result
from packaging.version import VERSION_PATTERN, Version

DEFAULT_ENV_NAME = "autopack"


@task
def version(c: Context):
    """
    Uses `hatch` to determine the version of the package from source.
    """

    # Why not use importlib.metadata.version? Because it does not
    # reflect the version of the source but rather of the installed
    # package.

    result = conda_run(c, "hatch version", hide=True)
    match = re.search(VERSION_PATTERN, result.stdout + result.stderr, re.VERBOSE)
    assert match is not None, "Could not determine version"
    ver = Version(match[0])
    print(ver)
    return ver


@task
def conda_run(c: Context, cmd, env_name=DEFAULT_ENV_NAME, **kwargs) -> Result:
    with c.prefix(f"conda activate {env_name}"):
        return c.run(cmd, **kwargs)


@task
def env(c: Context, env_name=DEFAULT_ENV_NAME):
    """
    Installs the Python development environment.
    """
    c.run(f"conda lock install -n {env_name} conda-lock.yml")
    # For some reason, the conda-forge msgpack does not include compiled extensions
    conda_run(c, "pip install --force-reinstall -U msgpack==1.0.7", env_name=env_name)
    conda_run(c, "pip install --no-deps -e .", env_name=env_name)


@task
def lock(c: Context):
    c.run("conda lock -f pyproject.toml --lockfile conda-lock.yml")


@task
def build(c: Context, env_name=DEFAULT_ENV_NAME):
    ver = version(c)

    # TODO: build docs so they can be included in the bundle by
    # PyInstaller
    pass

    conda_run(c, "pyinstaller --noconfirm autopack.spec")
    build_path = pathlib.Path("build").resolve()
    dist_path = pathlib.Path("dist").resolve()
    bundle_path = dist_path / "autopack"
    unversioned_archive_path = build_path / "autopack.7z"
    versioned_archive_path = dist_path / f"autopack-v{ver}.7z"

    # The reason we use 7z is because the user might try to unzip the
    # file using the built-in Windows unzipper, which doesn't support
    # long file names.

    # Make sure that 7z does not append to an existing archive
    unversioned_archive_path.unlink(missing_ok=True)

    with c.cd(bundle_path):
        # 7z handles dots in filenames really badly, so we're archiving
        # into a safe filename and renaming afterwards
        conda_run(c, f"7z a {unversioned_archive_path} .")

    print(f"Moving archive to {versioned_archive_path}")
    # Moving to an existing file will fail
    versioned_archive_path.unlink(missing_ok=True)
    unversioned_archive_path.rename(versioned_archive_path)


@task
def tests(c: Context, env_name=DEFAULT_ENV_NAME):
    conda_run(c, "pytest -s -vvv --color=yes", env_name=env_name)


@task
def app(c: Context, env_name=DEFAULT_ENV_NAME):
    conda_run(
        c, "panel serve --autoreload src/autopack/gui_entrypoint.py", env_name=env_name
    )
