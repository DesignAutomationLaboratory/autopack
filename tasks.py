import pathlib

from invoke import task

DEFAULT_ENV_NAME = "autopack"


@task
def conda_run(c, cmd, env_name=DEFAULT_ENV_NAME):
    return c.run(f"conda run -n {env_name} {cmd}")


@task
def env(c, env_name=DEFAULT_ENV_NAME):
    """
    Installs the Python development environment.
    """
    c.run(f"conda lock install -n {env_name} conda-lock.yml")
    conda_run(c, f"pip install --no-deps -e .", env_name=env_name)


@task
def lock(c):
    c.run(f"conda lock -f pyproject.toml --lockfile conda-lock.yml")


@task
def build(c, env_name=DEFAULT_ENV_NAME):
    version_response = conda_run(c, f"hatch version")
    version = version_response.stdout.strip()

    # TODO: build docs so they can be included in the bundle by
    # PyInstaller
    pass

    conda_run(c, f"pyinstaller --noconfirm autopack.spec")
    bundle_path = pathlib.Path("dist/autopack")

    # The reason we use 7z is because the user might try to unzip the
    # file using the built-in Windows unzipper, which doesn't support
    # long file names.

    # TODO: fix the paths in the bundle so that they are relative to the bundle root
    archive_path = pathlib.Path(f"dist/archive/autopack-{version}.7z")
    archive_path.unlink(missing_ok=True)
    conda_run(c, f"7z a {archive_path} {bundle_path}")
