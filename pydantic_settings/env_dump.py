"""Dump a settings instance to environment variables.

This module implements `BaseSettings.model_dump_env`, which converts a live settings
instance into a `dict[str, str]` of environment variables that `EnvSettingsSource`
can read back, e.g. to hand the configuration to another process.
"""

from __future__ import annotations as _annotations

import json
import pickle
import subprocess
import sys
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_origin

from pydantic import AliasChoices, AliasPath, BaseModel, TypeAdapter
from pydantic._internal._utils import is_model_class
from pydantic.dataclasses import is_pydantic_dataclass
from pydantic.fields import FieldInfo
from pydantic.types import _SecretBase

from .exceptions import SettingsError
from .sources import EnvSettingsSource
from .sources.types import ForceDecode, NoDecode
from .sources.utils import (
    _annotation_enum_val_to_name,
    _get_field_metadata,
    _get_model_fields,
    _resolve_type_alias,
    _strip_annotated,
)
from .utils import _lenient_issubclass

if TYPE_CHECKING:
    from .main import BaseSettings


def _is_secret(annotation: Any, value: Any) -> bool:
    """Whether a field holds a secret value (`SecretStr`, `SecretBytes`, `Secret[...]`, ...)."""
    if isinstance(value, _SecretBase):
        return True
    if annotation is None:
        return False
    annotation = _strip_annotated(_resolve_type_alias(annotation))
    return _lenient_issubclass(annotation, _SecretBase) or _lenient_issubclass(get_origin(annotation), _SecretBase)


def _is_model_value(value: Any) -> bool:
    return is_model_class(type(value)) or is_pydantic_dataclass(type(value))


def _alias_path_tail(field: FieldInfo) -> list[str | int] | None:
    """Return the remaining segments when the field's preferred validation alias is a multi-part `AliasPath`."""
    v_alias = field.validation_alias
    path: list[str | int] | None = None
    if isinstance(v_alias, AliasPath):
        path = list(v_alias.path)
    elif isinstance(v_alias, AliasChoices) and v_alias.choices:
        first_choice = v_alias.choices[0]
        if isinstance(first_choice, AliasPath):
            path = list(first_choice.path)
    if path and len(path) > 1:
        return path[1:]
    return None


def _wrap_alias_path(path: list[str | int], value: Any) -> Any:
    """Wrap `value` in the minimal container that an `AliasPath` navigates into."""
    for segment in reversed(path):
        if isinstance(segment, int):
            items: list[Any] = [None] * (segment + 1)
            items[segment] = value
            value = items
        else:
            value = {segment: value}
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _SecretBase):
        return value.get_secret_value()
    if isinstance(value, bytes):
        try:
            return value.decode()
        except UnicodeDecodeError:
            return str(value)
    if isinstance(value, (set, frozenset, tuple, deque)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    return str(value)


class _EnvDump:
    """Builds the environment variable mapping for one settings instance."""

    def __init__(self, instance: BaseSettings) -> None:
        self.instance = instance
        self.settings_cls = type(instance)
        self.config = self.settings_cls.model_config
        self.source = EnvSettingsSource(self.settings_cls)
        # Nested keys never carry the env prefix, so nested fields are resolved with an empty one.
        self.nested_source = EnvSettingsSource(self.settings_cls, env_prefix='')
        self.delimiter = self.source.env_nested_delimiter
        self.parse_none_str = self.source.env_parse_none_str
        self.parse_enums = self.source.env_parse_enums
        self.env: dict[str, str] = {}
        self._owners: dict[str, str] = {}
        self.compared_fields: list[str] = []

    def dump(self, include_secrets: bool) -> dict[str, str]:
        for field_name, field in self.settings_cls.model_fields.items():
            value = getattr(self.instance, field_name)
            if _is_secret(field.annotation, value) and not include_secrets:
                continue
            self.compared_fields.append(field_name)
            _, env_name, value_is_complex = self.source._extract_field_info(field, field_name)[0]
            self._emit(env_name, field, value, value_is_complex, field_name)
        return self.env

    def _emit(self, env_name: str, field: FieldInfo, value: Any, value_is_complex: bool, path: str) -> None:
        if isinstance(value, _SecretBase):
            value = value.get_secret_value()
        if value is None:
            # `None` only survives the round trip when `env_parse_none_str` is configured;
            # otherwise the variable is omitted and the field default applies.
            if self.parse_none_str is not None:
                self._set(env_name, self.parse_none_str, path)
            return
        alias_path_tail = _alias_path_tail(field)
        if alias_path_tail:
            self._set(env_name, self._json_dumps(_wrap_alias_path(alias_path_tail, value), field), path)
            return
        if self._no_decode(field):
            self._set(env_name, value if isinstance(value, str) else str(value), path)
            return
        if self.delimiter and _is_model_value(value):
            self._emit_model(env_name, value, path)
            return
        if value_is_complex or self.source._field_is_complex(field)[0]:
            self._set(env_name, self._json_dumps(value, field), path)
            return
        self._set(env_name, self._encode_scalar(field, value), path)

    def _emit_model(self, env_name: str, model_value: Any, path: str) -> None:
        for sub_name, sub_field in _get_model_fields(type(model_value)).items():
            sub_value = getattr(model_value, sub_name)
            _, sub_env_name, sub_is_complex = self.nested_source._extract_field_info(sub_field, sub_name)[0]
            full_name = f'{env_name}{self.delimiter}{sub_env_name}'
            self._emit(full_name, sub_field, sub_value, sub_is_complex, f'{path}.{sub_name}')

    def _encode_scalar(self, field: FieldInfo, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode()
            except UnicodeDecodeError:
                return str(value)
        if self.parse_enums:
            enum_name = _annotation_enum_val_to_name(field.annotation, value)
            if enum_name is not None:
                return enum_name
        if isinstance(value, Enum):
            enum_value = value.value
            return enum_value if isinstance(enum_value, str) else json.dumps(enum_value)
        if isinstance(value, bool):
            return 'true' if value else 'false'
        try:
            dumped = TypeAdapter(field.annotation).dump_python(value, mode='json')
        except Exception:
            dumped = value
        if isinstance(dumped, str):
            return dumped
        if isinstance(dumped, bytes):
            try:
                return dumped.decode()
            except UnicodeDecodeError:
                return str(dumped)
        return json.dumps(dumped, default=_json_default)

    def _json_dumps(self, value: Any, field: FieldInfo) -> str:
        try:
            return TypeAdapter(field.annotation).dump_json(value).decode()
        except Exception:
            pass
        try:
            return json.dumps(value, default=_json_default, separators=(',', ':'))
        except (TypeError, ValueError):
            return str(value)

    def _no_decode(self, field: FieldInfo) -> bool:
        metadata = _get_field_metadata(field)
        return NoDecode in metadata or (self.config.get('enable_decoding') is False and ForceDecode not in metadata)

    def _set(self, env_name: str, value: str, path: str) -> None:
        if env_name in self.env:
            raise SettingsError(
                'conflicting environment variable names while dumping settings: '
                f'{env_name!r} would be set by both {self._owners[env_name]!r} and {path!r}'
            )
        self._owners[env_name] = path
        self.env[env_name] = value


def model_dump_env(instance: BaseSettings, include_secrets: bool) -> tuple[dict[str, str], list[str]]:
    """Dump `instance` as environment variables, returning the mapping and the dumped field names."""
    dumper = _EnvDump(instance)
    env = dumper.dump(include_secrets)
    return env, dumper.compared_fields


_VERIFY_SCRIPT = """
import importlib
import importlib.util
import json
import os
import pickle
import sys


def _load_module(module_name, module_file):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def _safe_equal(left, right):
    try:
        return bool(left == right)
    except Exception:
        return False


def main():
    module_name, module_file, qualname = sys.argv[1], sys.argv[2], sys.argv[3]
    module = _load_module(module_name, module_file)
    settings_cls = module
    for part in qualname.split('.'):
        settings_cls = getattr(settings_cls, part)
    payload = pickle.load(sys.stdin.buffer)
    os.environ.clear()
    os.environ.update(payload['env_dump'])
    from pydantic_settings import EnvSettingsSource

    source = EnvSettingsSource(settings_cls)
    try:
        rebuilt = settings_cls.model_validate(source())
    except Exception as exc:
        json.dump({'error': f'{type(exc).__name__}: {exc}'}, sys.stdout)
        return
    original = payload['instance']
    non_reversible = []
    for field_name in payload['field_names']:
        original_value = getattr(original, field_name)
        rebuilt_value = getattr(rebuilt, field_name)
        if type(original_value) is not type(rebuilt_value) or not _safe_equal(original_value, rebuilt_value):
            non_reversible.append(field_name)
    original_extra = getattr(original, '__pydantic_extra__', None) or {}
    rebuilt_extra = getattr(rebuilt, '__pydantic_extra__', None) or {}
    for key, original_value in original_extra.items():
        rebuilt_value = rebuilt_extra.get(key)
        if (
            key not in rebuilt_extra
            or type(rebuilt_value) is not type(original_value)
            or not _safe_equal(rebuilt_value, original_value)
        ):
            non_reversible.append(key)
    json.dump({'non_reversible': non_reversible}, sys.stdout)


main()
"""


def verify_env_dump(instance: BaseSettings, env_dump: dict[str, str], field_names: list[str]) -> list[str]:
    """Rebuild `type(instance)` from `env_dump` alone in an isolated subprocess.

    The subprocess clears the whole environment, applies `env_dump`, resolves the settings
    with only an `EnvSettingsSource` (init kwargs, dotenv, secrets and custom sources are all
    disabled) and compares the type and value of every dumped field with this instance.

    Returns the names of the fields that could not be reproduced.
    """
    settings_cls = type(instance)
    module_name = settings_cls.__module__
    qualname = settings_cls.__qualname__
    module_file = getattr(sys.modules.get(module_name), '__file__', None)
    if '<locals>' in qualname or module_name == '__main__' or not module_file:
        raise SettingsError(
            f'cannot verify the environment dump of {settings_cls.__name__!r}: isolated verification '
            'requires the settings class to be defined at the top level of an importable module'
        )
    try:
        payload = pickle.dumps({'env_dump': env_dump, 'field_names': field_names, 'instance': instance})
    except Exception as e:
        raise SettingsError(
            f'cannot verify the environment dump of {settings_cls.__name__!r}: the instance is not picklable: {e}'
        ) from e
    process = subprocess.run(
        [sys.executable, '-c', _VERIFY_SCRIPT, module_name, module_file, qualname],
        input=payload,
        capture_output=True,
        timeout=120,
    )
    if process.returncode != 0:
        raise SettingsError(
            f'could not verify the environment dump of {settings_cls.__name__!r} in an isolated process: '
            f'{process.stderr.decode(errors="replace").strip()}'
        )
    result = json.loads(process.stdout)
    if result.get('error'):
        raise SettingsError(
            f'could not rebuild {settings_cls.__name__!r} from the dumped environment variables: {result["error"]}'
        )
    return result['non_reversible']


__all__ = ['model_dump_env', 'verify_env_dump']
