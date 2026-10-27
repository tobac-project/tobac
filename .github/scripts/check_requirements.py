"""Check that our requirement lists agree

Requirements are stated in several places, because conda-forge and PyPI
names differ:
  - pyproject.toml: [project] dependencies / optional-dependencies and
    [dependency-groups] (PyPI names, pip and uv)
  - pixi.toml (conda-forge names, pixi)
  - environment.yml / environment-examples.yml (conda-forge names, conda),
    mirroring the "default" / "examples" pixi environments

This script checks that they name the same packages with the same version
pins, and that the pixi min-versions environment tests the lower bounds of the
core dependencies.
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# PyPI name -> conda-forge name, when they differ
PYPI_TO_CONDA = {
    "scitools-iris": "iris",
}
# conda-only packages (not on PyPI/not needed on pip)
CONDA_ONLY = {"pip", "ffmpeg", "pandoc"}

# conda environment file -> pixi environment it mirrors
ENV_FILES = {
    "environment.yml": "default",
    "environment-examples.yml": "examples",
}


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def split_requirement(req):
    """Split a PEP 508 requirement into (name, version specifier), ignoring markers."""
    req = req.split(";")[0]
    match = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*)$", req)
    return match.group(1), match.group(3).replace(" ", "")


def conda_spec(spec):
    return "" if spec == "*" else spec.replace(" ", "")


def load_toml(filename):
    with open(ROOT / filename, "rb") as f:
        return tomllib.load(f)


def read_environment_file(filename):
    """Return {name: spec} of the dependencies listed in a conda environment file."""
    deps = {}
    in_dependencies = False
    for line in (ROOT / filename).read_text().splitlines():
        line = line.split("#")[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            in_dependencies = line == "dependencies:"
        elif in_dependencies and line.lstrip().startswith("- "):
            item = line.lstrip()[2:].strip()
            match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", item)
            deps[match.group(1)] = match.group(2).replace(" ", "")
    return deps


def requirement_lists(pyproject):
    """Return {name: [requirements]} of all extras and dependency groups."""
    lists = dict(pyproject["project"].get("optional-dependencies", {}))
    for group, reqs in pyproject.get("dependency-groups", {}).items():
        if group in lists:
            sys.exit(f"'{group}' is both an extra and a dependency group")
        lists[group] = reqs
    return lists


def own_requirements(project, reqs):
    """Return {name: spec}, skipping include-group entries and tobac[...] self references."""
    out = {}
    for req in reqs:
        if isinstance(req, dict):
            continue
        name, spec = split_requirement(req)
        if normalize(name) != normalize(project["name"]):
            out[normalize(name)] = spec
    return out


def conda_dependencies(pixi, environment):
    """Return {name: spec} of conda packages in a pixi environment."""
    deps = {
        name: conda_spec(spec) for name, spec in pixi.get("dependencies", {}).items()
    }
    for feature in pixi["environments"][environment]:
        for name, spec in pixi["feature"][feature].get("dependencies", {}).items():
            deps[name] = conda_spec(spec)
    return deps


def compare(label, pypi_reqs, conda_reqs, core=None):
    """Compare PyPI requirements to conda requirements, return a list of errors.

    Requirements also in ``core`` may be left out of conda_reqs, since pixi
    environments always include the core dependencies.
    """
    core = core or {}
    errors = []
    conda_reqs = {normalize(k): v for k, v in conda_reqs.items() if k not in CONDA_ONLY}
    mapped = {
        normalize(PYPI_TO_CONDA.get(name, name)): (name, spec)
        for name, spec in pypi_reqs.items()
    }
    for conda_name, (pypi_name, spec) in mapped.items():
        if conda_name not in conda_reqs:
            if pypi_name not in core:
                errors.append(f"{label}: '{pypi_name}' has no conda counterpart")
        elif conda_reqs[conda_name] != spec:
            errors.append(
                f"{label}: '{pypi_name}' pinned as '{spec}' (PyPI) "
                f"but '{conda_reqs[conda_name]}' (conda)"
            )
    for conda_name in conda_reqs.keys() - mapped.keys():
        errors.append(f"{label}: conda package '{conda_name}' has no PyPI counterpart")
    return errors


def compare_environment_file(filename, expected):
    """Compare a conda environment file to the {name: spec} it should list."""
    errors = []
    actual = {normalize(k): v for k, v in read_environment_file(filename).items()}
    expected = {normalize(k): v for k, v in expected.items()}
    for name in expected.keys() - actual.keys():
        errors.append(f"{filename}: '{name}' is missing")
    for name in actual.keys() - expected.keys():
        errors.append(f"{filename}: '{name}' is not in the pixi environment")
    for name in expected.keys() & actual.keys():
        if actual[name] != expected[name]:
            source = "requires-python" if name == "python" else "pixi.toml"
            errors.append(
                f"{filename}: '{name}' pinned as '{actual[name]}' "
                f"but '{expected[name]}' in {source}"
            )
    return errors


def check(pyproject, pixi):
    project = pyproject["project"]
    core = own_requirements(project, project["dependencies"])

    errors = compare(
        "core dependencies",
        core,
        {k: conda_spec(v) for k, v in pixi["dependencies"].items()},
    )
    for name, reqs in requirement_lists(pyproject).items():
        if name not in pixi["feature"]:
            continue
        conda_reqs = {
            k: conda_spec(v)
            for k, v in pixi["feature"][name].get("dependencies", {}).items()
        }
        errors += compare(
            f"'{name}'", own_requirements(project, reqs), conda_reqs, core
        )

    # the min-versions environment must test exactly the core lower bounds
    min_versions = {
        normalize(k): v
        for k, v in pixi["feature"]["min-versions"]["dependencies"].items()
    }
    for name, spec in core.items():
        conda_name = normalize(PYPI_TO_CONDA.get(name, name))
        floor = re.fullmatch(r">=([0-9.]+)", spec)
        expected = f"{floor.group(1)}.*" if floor else None
        if expected is None:
            errors.append(f"core dependency '{name}' needs a lower bound (>=X.Y)")
        elif min_versions.get(conda_name) != expected:
            errors.append(
                f"min-versions: '{conda_name}' is '{min_versions.get(conda_name)}' "
                f"but the lower bound in [project] dependencies is '{spec}'"
            )
    return errors


def main():
    pyproject = load_toml("pyproject.toml")
    pixi = load_toml("pixi.toml")
    errors = check(pyproject, pixi)

    requires_python = pyproject["project"]["requires-python"].replace(" ", "")
    for filename, environment in ENV_FILES.items():
        expected = {"python": requires_python, **conda_dependencies(pixi, environment)}
        errors += compare_environment_file(filename, expected)

    if errors:
        print("Requirements are out of sync:", *errors, sep="\n  ", file=sys.stderr)
        print(
            "Update pyproject.toml, pixi.toml, environment.yml and "
            "environment-examples.yml to match.",
            file=sys.stderr,
        )
        return 1
    print("Requirements are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
