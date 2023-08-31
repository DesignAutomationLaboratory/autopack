# Autopack

## Directory structure

* `pyproject.toml`: project metadata and abstract dependencies used to build the project and install development environment.
* `conda-lock.yml`: locked dependencies for conda, generated from `pyproject.toml`. Enables reproducible builds and development environments.
* `tasks.py`: automation tasks for building, testing, and releasing the project. See [invoke](https://www.pyinvoke.org/) for more information.
