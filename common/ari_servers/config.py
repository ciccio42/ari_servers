"""Configuration of the servers, from the lowest to the highest priority:

1. the defaults of the dataclass of the server (e.g. WhisperConfig)
2. its section of config/servers.yaml (ARI_SERVERS_CONFIG to use another file)
3. environment: <PREFIX>_<FIELD> (e.g. WHISPER_MODEL, WHISPER_PORT), then the aliases of the field
4. command line: --<field> (e.g. --model, --port)

So the same server runs with the model of the YAML file, or another one with one variable or option, without
changing the code. WORKSPACE is the root of ari_servers (models/, latency_logs/, run/).
"""
import argparse
import dataclasses
import os
import typing
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[2])
CONFIG_FILE = Path(os.environ.get("ARI_SERVERS_CONFIG") or WORKSPACE / "config" / "servers.yaml")


def read_config_file(path: Path = CONFIG_FILE) -> dict:
    if not path.is_file():
        return {}
    import yaml  # only when the file exists: pyyaml comes with transformers in every image
    with open(path) as f:
        return yaml.safe_load(f) or {}


def shared(key: str, default=None, env: str = None):
    """A top-level value of servers.yaml (e.g. execution_lock), overridden by the environment variable `env`."""
    if env and env in os.environ:
        return os.environ[env]
    return read_config_file().get(key, default)


def _convert(kind, value):
    if value is None:
        return None
    if typing.get_origin(kind) is typing.Union:  # Optional[X]
        kind = next(arg for arg in typing.get_args(kind) if arg is not type(None))
    if kind is bool and isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return kind(value)


def load(cls, section: str, env_prefix: str, argv=None, aliases: dict = None, description: str = None):
    """Builds the dataclass `cls` from defaults, servers.yaml[section], environment and command line."""
    aliases = aliases or {}
    hints = typing.get_type_hints(cls)
    fields = [f.name for f in dataclasses.fields(cls)]
    values = dict(read_config_file().get(section) or {})
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise SystemExit(f"{CONFIG_FILE}: unknown keys in section {section}: {unknown}. Known: {fields}")
    for name in fields:
        for variable in (f"{env_prefix}_{name.upper()}", *aliases.get(name, ())):
            if os.environ.get(variable, "") != "":
                values[name] = os.environ[variable]
                break

    parser = argparse.ArgumentParser(description=description)
    for name in fields:
        default = values.get(name, getattr(cls, name, None))
        parser.add_argument("--" + name.replace("_", "-"), dest=name, default=None,
                            help=f"default: {default!r} (env {env_prefix}_{name.upper()})")
    args = parser.parse_args(argv)
    for name in fields:
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    return cls(**{name: _convert(hints[name], value) for name, value in values.items()})


def workspace_path(path) -> Path:
    """A path of the configuration: absolute as it is, relative to WORKSPACE otherwise."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else WORKSPACE / path
