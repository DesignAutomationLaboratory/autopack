import pathlib

from invoke import task

DEFAULT_ENV_NAME = "autopack"


@task
def version(c):
    version_response = conda_run(c, "hatch version")
    return version_response.stdout.strip()


@task
def conda_run(c, cmd, env_name=DEFAULT_ENV_NAME):
    with c.prefix(f"conda activate {env_name}"):
        return c.run(cmd)


@task
def env(c, env_name=DEFAULT_ENV_NAME):
    """
    Installs the Python development environment.
    """
    c.run(f"conda lock install -n {env_name} conda-lock.yml")
    conda_run(c, "pip install --no-deps -e .", env_name=env_name)


@task
def lock(c):
    c.run("conda lock -f pyproject.toml --lockfile conda-lock.yml")


@task
def build(c, env_name=DEFAULT_ENV_NAME):
    ver = version(c)

    # TODO: build docs so they can be included in the bundle by
    # PyInstaller
    pass

    conda_run(c, "pyinstaller --noconfirm autopack.spec")
    bundle_path = pathlib.Path("dist/autopack")

    # The reason we use 7z is because the user might try to unzip the
    # file using the built-in Windows unzipper, which doesn't support
    # long file names.

    archive_path = pathlib.Path(f"dist/archive/autopack-{ver}.7z").resolve()
    archive_path.unlink(missing_ok=True)

    with c.cd(bundle_path):
        conda_run(c, f"7z a {archive_path} .")
