from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any
from uuid import UUID

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    Field,
    Json,
    RootModel,
    SecretStr,
    field_validator,
)
from pydantic.dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError


class Color(Enum):
    RED = 'red'
    BLUE = 'blue'


class Size(IntEnum):
    SMALL = 1
    LARGE = 2


def test_model_dump_env_basic_scalars():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        name: str = 'myapp'
        count: int = 3
        ratio: float = 1.5
        debug: bool = False
        color: Color = Color.BLUE
        size: Size = Size.LARGE

    assert Settings().model_dump_env() == {
        'app_name': 'myapp',
        'app_count': '3',
        'app_ratio': '1.5',
        'app_debug': 'false',
        'app_color': 'blue',
        'app_size': '2',
    }
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_case_sensitive():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_', case_sensitive=True)

        MyField: str = 'value'

    assert Settings().model_dump_env() == {'APP_MyField': 'value'}
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_aliases():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        field_one: str = Field(default='a', validation_alias='ONE')
        field_two: str = Field(default='b', validation_alias=AliasChoices('TWO', 'two_alt'))
        field_three: str = Field(default='c', alias='THREE')

    # With the default env_prefix_target='variable', aliases do not carry the prefix.
    assert Settings().model_dump_env() == {'one': 'a', 'two': 'b', 'three': 'c'}
    assert Settings().model_dump_env_check() == []

    class TargetAllSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_', env_prefix_target='all')

        field_one: str = Field(default='a', validation_alias='ONE')

    assert TargetAllSettings().model_dump_env() == {'app_one': 'a'}
    assert TargetAllSettings().model_dump_env_check() == []


def test_model_dump_env_alias_path():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        first: int = Field(validation_alias=AliasPath('deep', 'deeper', 1))
        second: str = Field(default='x', validation_alias=AliasChoices(AliasPath('nested', 'value'), 'fallback'))

    settings = Settings(deep={'deeper': [0, 42]}, nested={'value': 'y'})
    assert settings.model_dump_env() == {
        'deep': '{"deeper": [null, 42]}',
        'nested': '{"value": "y"}',
    }
    assert settings.model_dump_env_check() == []


def test_model_dump_env_nested_delimiter_explodes_models_and_mappings():
    class SubSub(BaseModel):
        val: str = 'v3'

    class Sub(BaseModel):
        num: int = 1
        sub_sub: SubSub = SubSub()
        items: list[int] = [1, 2]

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_', env_nested_delimiter='__')

        sub: Sub = Sub()
        mapping: dict[str, int] = {'a': 1, 'b': 2}
        empty_mapping: dict[str, int] = {}

    assert Settings().model_dump_env() == {
        'app_sub__num': '1',
        'app_sub__sub_sub__val': 'v3',
        'app_sub__items': '[1, 2]',
        'app_mapping__a': '1',
        'app_mapping__b': '2',
        'app_empty_mapping': '{}',
    }
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_nested_delimiter_max_split_falls_back_to_json():
    class SubSub(BaseModel):
        val: str = 'v3'

    class Sub(BaseModel):
        sub_sub: SubSub = SubSub()

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_', env_nested_delimiter='__', env_nested_max_split=1)

        sub: Sub = Sub()

    assert Settings().model_dump_env() == {'app_sub__sub_sub': '{"val": "v3"}'}
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_complex_values_json_encoded_without_delimiter():
    class Sub(BaseModel):
        num: int = 1
        aliased: str = Field(default='x', validation_alias='ALIASED')

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        sub: Sub = Sub()
        tags: list[str] = ['a', 'b']
        mapping: dict[str, list[int]] = {'k': [1]}

    assert Settings().model_dump_env() == {
        'app_sub': '{"num": 1, "ALIASED": "x"}',
        'app_tags': '["a", "b"]',
        'app_mapping': '{"k": [1]}',
    }
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_none_empty_string_and_empty_containers():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_', env_parse_none_str='null')

        maybe: str | None = None
        empty_str: str = ''
        empty_list: list[int] = []
        empty_dict: dict[str, int] = {}

    assert Settings().model_dump_env() == {
        'app_maybe': 'null',
        'app_empty_str': '',
        'app_empty_list': '[]',
        'app_empty_dict': '{}',
    }
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_none_without_parse_none_str_is_omitted():
    class Settings(BaseSettings):
        maybe: str | None = None
        other: str = 'x'

    assert Settings().model_dump_env() == {'other': 'x'}
    # The default is also None, so the omission is lossless.
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_check_lists_none_that_cannot_be_represented():
    class Settings(BaseSettings):
        maybe: str | None = 'default'
        other: str = 'x'

    settings = Settings(maybe=None)
    assert settings.model_dump_env() == {'other': 'x'}
    assert settings.model_dump_env_check() == ['maybe']


def test_model_dump_env_json_annotation():
    class Settings(BaseSettings):
        payload: Json[dict[str, int]]

    settings = Settings(payload='{"a": 1}')
    assert settings.model_dump_env() == {'payload': '{"a": 1}'}
    assert settings.model_dump_env_check() == []


def test_model_dump_env_key_conflict_raises():
    class Settings(BaseSettings):
        one: str = 'a'
        ONE: str = 'b'

    with pytest.raises(SettingsError, match="'one' is required by both 'one' and 'ONE'"):
        Settings().model_dump_env()


def test_model_dump_env_key_conflict_with_alias_raises():
    class Settings(BaseSettings):
        one: str = 'a'
        two: str = Field(default='b', validation_alias='one')

    with pytest.raises(SettingsError, match='one'):
        Settings().model_dump_env()


def test_model_dump_env_secrets_omitted_by_default():
    class Sub(BaseModel):
        token: SecretStr = SecretStr('nested-secret')
        plain: str = 'p'

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        password: SecretStr = SecretStr('top-secret')
        sub: Sub = Sub()

    assert Settings().model_dump_env() == {'app_sub': '{"plain": "p"}'}
    assert Settings().model_dump_env(include_secrets=True) == {
        'app_password': 'top-secret',
        'app_sub': '{"token": "nested-secret", "plain": "p"}',
    }
    assert Settings().model_dump_env_check(include_secrets=True) == []


def test_model_dump_env_check_reports_non_idempotent_validator():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        shouted: str = 'a'
        normal: str = 'b'

        @field_validator('shouted')
        @classmethod
        def append_bang(cls, v: str) -> str:
            return v + '!'

    settings = Settings()
    assert settings.shouted == 'a!'
    assert settings.model_dump_env() == {'app_shouted': 'a!', 'app_normal': 'b'}
    # The validator appends '!' on every validation, so the field is not env-reversible.
    assert settings.model_dump_env_check() == ['shouted']


def test_model_dump_env_check_reports_validation_failures():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        count: int = 1

        @field_validator('count', mode='before')
        @classmethod
        def reject_string_values(cls, v: Any) -> Any:
            if isinstance(v, str):
                raise ValueError('strings are not accepted')
            return v

    settings = Settings()
    assert settings.model_dump_env() == {'app_count': '1'}
    assert settings.model_dump_env_check() == ['count']


def test_model_dump_env_check_reports_missing_required_secret():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        password: SecretStr
        name: str = 'x'

    settings = Settings(password='s3cret')
    assert settings.model_dump_env() == {'app_name': 'x'}
    # The required secret is omitted from the dump, so the env-only rebuild fails on it.
    assert settings.model_dump_env_check() == ['password']


def test_model_dump_env_scalar_types():
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        data: bytes = b'hello'
        when: datetime = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        uid: UUID = UUID('12345678-1234-5678-1234-567812345678')

    dumped = Settings().model_dump_env()
    assert dumped == {
        'app_data': 'hello',
        'app_when': '2026-01-02 03:04:05+00:00',
        'app_uid': '12345678-1234-5678-1234-567812345678',
    }
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_root_model_and_excluded_fields():
    class Tags(RootModel[list[int]]):
        pass

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        tags: Tags = Tags([1, 2])
        internal: str = Field(default='hidden', exclude=True)
        name: str = 'x'

    assert Settings().model_dump_env() == {'app_tags': '[1, 2]', 'app_name': 'x'}
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_nested_dataclass():
    @dataclass
    class Sub:
        num: int = 1
        word: str = 'w'

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_', env_nested_delimiter='__')

        sub: Sub = Sub()

    assert Settings().model_dump_env() == {'app_sub__num': '1', 'app_sub__word': 'w'}
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_empty_model_falls_back_to_json():
    class Empty(BaseModel):
        pass

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_', env_nested_delimiter='__')

        empty: Empty = Empty()

    assert Settings().model_dump_env() == {'app_empty': '{}'}
    assert Settings().model_dump_env_check() == []


def test_model_dump_env_arbitrary_type_inside_complex_value():
    class Custom:
        def __init__(self, token: str):
            self.token = token

        def __str__(self):
            return f'Custom({self.token})'

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='APP_')

        customs: list[Custom] = [Custom('a')]

    assert Settings().model_dump_env() == {'app_customs': '["Custom(a)"]'}


ROUNDTRIP_MODELS = """
from enum import Enum

from pydantic import BaseModel, Field, SecretStr

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Color(Enum):
    RED = 'red'
    BLUE = 'blue'


class SubSub(BaseModel):
    val: str
    flag: bool


class Sub(BaseModel):
    num: int
    sub_sub: SubSub
    aliased: str = Field(validation_alias='ALIASED')


class RoundTripSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='APP_', env_nested_delimiter='__', env_parse_none_str='null')

    name: str
    debug: bool
    color: Color
    maybe: str | None
    empty_str: str
    sub: Sub
    tags: list[int]
    mapping: dict[str, int]
    password: SecretStr


class EnvOnlySettings(RoundTripSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        # The child process rebuilds from environment variables only.
        return (env_settings,)
"""

ROUNDTRIP_CHILD = """
import json
import sys

sys.path.insert(0, {models_dir!r})

from rt_models import EnvOnlySettings

settings = EnvOnlySettings()
types = {{name: type(getattr(settings, name)).__name__ for name in settings.model_fields}}
types['sub'] = type(settings.sub).__name__
types['sub.sub_sub'] = type(settings.sub.sub_sub).__name__
print(json.dumps({{'values': json.loads(settings.model_dump_json()), 'types': types}}))
"""


def test_model_dump_env_subprocess_roundtrip(tmp_path):
    (tmp_path / 'rt_models.py').write_text(ROUNDTRIP_MODELS)
    sys.path.insert(0, str(tmp_path))
    try:
        import rt_models
    finally:
        sys.path.remove(str(tmp_path))

    settings = rt_models.RoundTripSettings(
        name='myapp',
        debug=True,
        color=rt_models.Color.BLUE,
        maybe=None,
        empty_str='',
        sub=rt_models.Sub(num=7, sub_sub=rt_models.SubSub(val='deep', flag=False), ALIASED='a'),
        tags=[1, 2, 3],
        mapping={'x': 1},
        password='s3cret',
    )
    dumped = settings.model_dump_env(include_secrets=True)

    # Rebuild in an isolated subprocess whose only environment variables are the dumped ones.
    result = subprocess.run(
        [sys.executable, '-c', ROUNDTRIP_CHILD.format(models_dir=str(tmp_path))],
        env=dict(dumped),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    rebuilt = json.loads(result.stdout)

    assert rebuilt['values'] == json.loads(settings.model_dump_json())
    assert rebuilt['types'] == {
        'name': 'str',
        'debug': 'bool',
        'color': 'Color',
        'maybe': 'NoneType',
        'empty_str': 'str',
        'sub': 'Sub',
        'tags': 'list',
        'mapping': 'dict',
        'password': 'SecretStr',
        'sub.sub_sub': 'SubSub',
    }
