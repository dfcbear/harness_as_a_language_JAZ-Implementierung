"""Configuration system for JAZ."""

import os
from pathlib import Path
import tomllib
from typing import Any, Optional

from .llm import BaseLLM, OpenAICompatibleLLM
from .protocol import BaseProtocol, RawCodeProtocol
from .repl import BaseREPL, ContainerREPL, LocalPythonREPL, SubprocessREPL


class Config:
    """Configuration containing the LLM backend, REPL backend, and protocol."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        repl: Optional[BaseREPL] = None,
        protocol: Optional[BaseProtocol] = None,
    ):
        self.llm = llm or OpenAICompatibleLLM()
        self.repl = repl or LocalPythonREPL()
        self.protocol = protocol or RawCodeProtocol()

    def clone(self) -> "Config":
        """Create a shallow clone of the configuration."""
        return Config(llm=self.llm, repl=self.repl, protocol=self.protocol)

    def __repr__(self) -> str:
        return (
            f"Config(llm={self.llm.model}, repl={type(self.repl).__name__}, "
            f"protocol={type(self.protocol).__name__})"
        )


class ConfigOverride:
    """Local or dynamically scoped configuration override."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        repl: Optional[BaseREPL] = None,
        protocol: Optional[BaseProtocol] = None,
    ):
        self.llm = llm
        self.repl = repl
        self.protocol = protocol
        self._token = None

    def apply_to(self, base_config: Config) -> Config:
        """Apply override fields onto a base Config."""
        return Config(
            llm=self.llm if self.llm is not None else base_config.llm,
            repl=self.repl if self.repl is not None else base_config.repl,
            protocol=self.protocol if self.protocol is not None else base_config.protocol,
        )

    # Scoped context manager support
    def __enter__(self):
        from .scope import _enter_scoped_config
        self._token = _enter_scoped_config(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from .scope import _exit_scoped_config
        if self._token is not None:
            _exit_scoped_config(self._token)
        return False


_global_default_config: Optional[Config] = None


def get_default_config() -> Config:
    """Return the global default configuration, loading it lazily if necessary."""
    global _global_default_config
    if _global_default_config is None:
        _global_default_config = load_config_from_env_or_file()
    return _global_default_config


def set_default_config(config: Config) -> None:
    """Set the global default configuration."""
    global _global_default_config
    _global_default_config = config


def _load_dotenv() -> None:
    """Load key-value pairs from .env file into os.environ if not already set."""
    paths_to_check = [
        Path(".env"),
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for env_path in paths_to_check:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.lstrip("\ufeff").strip()
                        val = val.strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        if key and key not in os.environ:
                            os.environ[key] = val
                break
            except Exception:
                pass


def _expand_env_value(val: Any) -> Any:
    """Expand ${VAR} and ${VAR:-default} patterns in string values using os.environ."""
    import re
    if not isinstance(val, str):
        return val

    def _replace(match):
        expr = match.group(1)
        if ":-" in expr:
            k, default = expr.split(":-", 1)
            return os.environ.get(k, default)
        return os.environ.get(expr, "")

    return re.sub(r"\$\{([^}]+)\}", _replace, val)


def load_config_from_env_or_file(config_path: Optional[str] = None) -> Config:
    """Load unified configuration from jaz.toml or environment variables.
    
    All LLM and REPL settings in one central place.
    """
    _load_dotenv()

    settings = {
        "base_url": os.environ.get("JAZ_BASE_URL", "http://localhost:8000/v1"),
        "model": os.environ.get("JAZ_MODEL", "default"),
        "api_key": os.environ.get("JAZ_API_KEY", "EMPTY"),
        "context_window": int(os.environ.get("JAZ_CONTEXT_WINDOW", "131072")),
        "max_tokens": int(os.environ["JAZ_MAX_TOKENS"]) if "JAZ_MAX_TOKENS" in os.environ else None,
        "temperature": float(os.environ.get("JAZ_TEMPERATURE", "0.0")),
        "timeout": float(os.environ.get("JAZ_TIMEOUT", "180.0")),
        "price_per_1k_input": float(os.environ.get("JAZ_PRICE_INPUT", "0.0")),
        "price_per_1k_output": float(os.environ.get("JAZ_PRICE_OUTPUT", "0.0")),
        "sandbox": os.environ.get("JAZ_SANDBOX", "direct").lower(),
        "container_runtime": os.environ.get("JAZ_CONTAINER_RUNTIME"),
        "container_image": os.environ.get("JAZ_CONTAINER_IMAGE", "python:3.11-slim"),
    }

    # Search for config file
    paths_to_check = []
    if config_path:
        paths_to_check.append(Path(config_path))
    else:
        paths_to_check.extend([
            Path("jaz.toml"),
            Path(__file__).resolve().parent.parent / "jaz.toml",
            Path.home() / ".jaz" / "config.toml",
        ])

    for p in paths_to_check:
        if p.is_file():
            try:
                with open(p, "rb") as f:
                    file_data = tomllib.load(f)
                llm_sec = file_data.get("llm", {})
                repl_sec = file_data.get("repl", {})
                settings.update({
                    "base_url": llm_sec.get("base_url", settings["base_url"]),
                    "model": llm_sec.get("model", settings["model"]),
                    "api_key": llm_sec.get("api_key", settings["api_key"]),
                    "context_window": int(llm_sec.get("context_window", settings["context_window"])),
                    "max_tokens": int(llm_sec["max_tokens"]) if "max_tokens" in llm_sec else settings["max_tokens"],
                    "temperature": float(llm_sec.get("temperature", settings["temperature"])),
                    "timeout": float(llm_sec.get("timeout", settings["timeout"])),
                    "price_per_1k_input": float(llm_sec.get("price_per_1k_input", settings["price_per_1k_input"])),
                    "price_per_1k_output": float(llm_sec.get("price_per_1k_output", settings["price_per_1k_output"])),
                    "sandbox": repl_sec.get("sandbox", settings["sandbox"]).lower(),
                    "container_runtime": repl_sec.get("container_runtime", settings["container_runtime"]),
                    "container_image": repl_sec.get("container_image", settings["container_image"]),
                })
                break
            except Exception:
                pass

    # Resolve environment variable references in string settings
    settings["base_url"] = _expand_env_value(settings["base_url"])
    settings["model"] = _expand_env_value(settings["model"])
    settings["api_key"] = _expand_env_value(settings["api_key"])
    if not settings["api_key"] or settings["api_key"] == "EMPTY":
        settings["api_key"] = os.environ.get("JAZ_API_KEY", settings["api_key"])

    # Build LLM
    llm = OpenAICompatibleLLM(
        base_url=settings["base_url"],
        model=settings["model"],
        api_key=settings["api_key"],
        context_window=settings["context_window"],
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        timeout=settings["timeout"],
        price_per_1k_input=settings["price_per_1k_input"],
        price_per_1k_output=settings["price_per_1k_output"],
    )

    # Build REPL
    sandbox_mode = settings["sandbox"]
    if sandbox_mode == "subprocess":
        repl = SubprocessREPL()
    elif sandbox_mode in ("container", "docker", "podman"):
        runtime = settings["container_runtime"] or ("podman" if sandbox_mode == "podman" else "docker")
        repl = ContainerREPL(runtime=runtime, image=settings["container_image"])
    else:
        repl = LocalPythonREPL()

    return Config(llm=llm, repl=repl, protocol=RawCodeProtocol())
