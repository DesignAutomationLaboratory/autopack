# Autopack

## Directory structure

* `pyproject.toml`: project metadata and abstract dependencies used to build the project and install development environment.
* `conda-lock.yml`: locked dependencies for conda, generated from `pyproject.toml`. Enables reproducible builds and development environments.
* `tasks.py`: automation tasks for building, testing, and releasing the project. See [invoke](https://www.pyinvoke.org/) for more information.

## Development activities

Want to make a change? Terrific!

All steps involving terminal commands below assume that you are in the root directory of the project, i.e. that you have run `cd <somewhere-on-your-computer>/autopack`.

### Getting started (setting up the development environment)

The project uses Conda to manage the development environment and its dependencies. Additionally, the `conda-lock` and `invoke` packages need to be installed in the base environment. Luckily, there's a script that will do all of this for you, and you should only need to do it once. Just follow the steps below:

1. Run `bin/setup-global-environment.bat` to setup your global Conda/Python environment.
2. In a new terminal (Powershell or whatever you prefer), `cd` to the root directory for this project and run `invoke env` to create the Conda development environment. This will create a new Conda environment called `autopack` and install all dependencies.
3. Code away, and see below for instructions on more specific tasks.

All instructions below assume you have an installed development environment.

### Running the tests

1. Run `invoke test`.

or...

1. Activate the `autopack` environment by running `conda activate autopack`.
2. Run `pytest` with your preferred options.

### Starting the application (for development)

1. Run `invoke app`.

### Building the application (for distribution)

This will be done automatically by the CI/CD pipeline, but if you want to do it locally, follow these steps:

1. Run `invoke build` and wait. The zipped build will be placed in the `dist` directory.

### Adding another package/dependency to the project

1. Activate the `autopack` environment by running `conda activate autopack`.
2. Doodle around with `conda install <package>` (or `pip install <package>` if needed) until you get what you want.
3. Add the package to the `pyproject.toml` file under the `[project.dependencies]` or `[project.optional-dependencies]` section. See the file for when to use which section.
4. Run `invoke lock` to update the `conda-lock.yml` file with the new dependency.
    * If you get an error from `conda-lock`, try removing the `conda-lock.yml` file and running `invoke lock` again.
5. Commit the changes to `pyproject.toml` **and** `conda-lock.yml` to source control. This will make sure that everyone on the project can install the exact same dependencies, and give us traceability for when things break.
