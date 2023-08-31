from invoke import task

DEFAULT_ENV_NAME = "autopack"


@task
def conda_run(c, cmd, env_name=DEFAULT_ENV_NAME):
    c.run(f"conda run -n {env_name} {cmd}")


@task
def dev(c, env_name=DEFAULT_ENV_NAME):
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
    conda_run(c, f"pyinstaller --noconfirm src/autopack.spec")
