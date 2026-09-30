from __future__ import annotations

import enum
import json
from typing import Annotated, Any

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, Field, SecretStr, field_validator

from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
)


class Sub(BaseModel):
    host: str = 'localhost'
    port: int = 5432


class Fruit(enum.Enum):
    apple = 'a'
    pear = 'p'


def test_dump_basic_scalars():
    class BasicSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='app_')

        name: str = 'x'
        count: int = 3
        ratio: float = 1.5
        flag: bool = True

    settings = BasicSettings(name='hello', count=5, flag=False)
    assert settings.model_dump_env() == {
        'app_name': 'hello',
        'app_count': '5',
        'app_ratio': '1.5',
        'app_flag': 'false',
    }


def test_dump_case_sensitive():
    class CaseSettings(BaseSettings):
        model_config = SettingsConfigDict(case_sensitive=True, env_prefix='APP_')

        FooBar: str = 'x'

    assert CaseSettings().model_dump_env() == {'APP_FooBar': 'x'}


def test_dump_aliases():
    class AliasSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='app_')

        auth: str = Field('x', validation_alias='my_auth')
        svc: str = Field('y', validation_alias=AliasChoices('svc_url', 'svc_uri'))

    # aliases are preferred over field names and, with the default
    # env_prefix_target='variable', do not carry the prefix
    assert AliasSettings().model_dump_env() == {'my_auth': 'x', 'svc_url': 'y'}


def test_dump_alias_prefix_target():
    class AliasPrefixSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='app_', env_prefix_target='alias')

        auth: str = Field('x', validation_alias='my_auth')

    assert AliasPrefixSettings().model_dump_env() == {'app_my_auth': 'x'}


def test_dump_alias_path(env):
    class AliasPathSettings(BaseSettings):
        first: str = Field('x', validation_alias=AliasPath('my_path', 0))
        named: str = Field('y', validation_alias=AliasPath('my_nested', 'key'))

    settings = AliasPathSettings(my_path=['v1'], my_nested={'key': 'v2'})
    dump = settings.model_dump_env()
    assert dump == {'my_path': '["v1"]', 'my_nested': '{"key":"v2"}'}
    for key, value in dump.items():
        env.set(key, value)
    assert AliasPathSettings() == settings


def test_dump_complex_values_are_json():
    class ComplexSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='c_')

        sub: Sub = Sub(host='example.com')
        mapping: dict[str, list[int]] = {'a': [1, 2]}
        tags: list[str] = []

    dump = ComplexSettings().model_dump_env()
    assert json.loads(dump['c_sub']) == {'host': 'example.com', 'port': 5432}
    assert json.loads(dump['c_mapping']) == {'a': [1, 2]}
    assert dump['c_tags'] == '[]'


def test_dump_nested_delimiter(env):
    class NestedSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='app_', env_nested_delimiter='__')

        sub: Sub = Sub()
        name: str = 'n'

    settings = NestedSettings(sub=Sub(host='h', port=1), name='svc')
    dump = settings.model_dump_env()
    assert dump == {'app_sub__host': 'h', 'app_sub__port': '1', 'app_name': 'svc'}
    for key, value in dump.items():
        env.set(key, value)
    assert NestedSettings() == settings


def test_dump_nested_delimiter_deep(env):
    class Inner(BaseModel):
        val: str = 'i'

    class Outer(BaseModel):
        inner: Inner = Inner()
        leaf: list[int] = []

    class DeepSettings(BaseSettings):
        model_config = SettingsConfigDict(env_nested_delimiter='__')

        outer: Outer = Outer()

    settings = Outer and DeepSettings(outer=Outer(inner=Inner(val='v'), leaf=[1, 2]))
    dump = settings.model_dump_env()
    assert dump == {'outer__inner__val': 'v', 'outer__leaf': '[1,2]'}
    for key, value in dump.items():
        env.set(key, value)
    assert DeepSettings() == settings


def test_dump_none_with_parse_none_str(env):
    class NoneSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='n_', env_parse_none_str='null')

        maybe: int | None = None
        other: str = 'x'

    settings = NoneSettings()
    assert settings.model_dump_env() == {'n_maybe': 'null', 'n_other': 'x'}
    for key, value in settings.model_dump_env().items():
        env.set(key, value)
    assert NoneSettings() == settings


def test_dump_none_without_parse_none_str_is_omitted():
    class NoneOmitSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='n_')

        maybe: int | None = None
        other: str = 'x'

    assert NoneOmitSettings().model_dump_env() == {'n_other': 'x'}


def test_dump_empty_values(env):
    class EmptySettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='e_')

        text: str = ''
        mapping: dict[str, int] = {}
        items: list[int] = []

    settings = EmptySettings()
    dump = settings.model_dump_env()
    assert dump == {'e_text': '', 'e_mapping': '{}', 'e_items': '[]'}
    for key, value in dump.items():
        env.set(key, value)
    assert EmptySettings() == settings


def test_dump_enum_value_and_name(env):
    class EnumSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='v_')

        fruit: Fruit = Fruit.apple

    class EnumNameSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='n_', env_parse_enums=True)

        fruit: Fruit = Fruit.apple

    value_settings = EnumSettings(fruit=Fruit.pear)
    name_settings = EnumNameSettings(fruit=Fruit.pear)
    assert value_settings.model_dump_env() == {'v_fruit': 'p'}
    assert name_settings.model_dump_env() == {'n_fruit': 'pear'}
    for key, value in value_settings.model_dump_env().items():
        env.set(key, value)
    for key, value in name_settings.model_dump_env().items():
        env.set(key, value)
    assert EnumSettings() == value_settings
    assert EnumNameSettings() == name_settings


def test_dump_secrets_omitted_by_default():
    class SecretSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='s_')

        token: SecretStr = SecretStr('hidden')
        name: str = 'n'

    settings = SecretSettings(token=SecretStr('s3cr3t'))
    assert settings.model_dump_env() == {'s_name': 'n'}
    assert settings.model_dump_env(include_secrets=True) == {'s_token': 's3cr3t', 's_name': 'n'}


def test_dump_conflicting_names_raise():
    class ConflictSettings(BaseSettings):
        one: str = Field('a', validation_alias='same_name')
        two: str = Field('b', validation_alias='same_name')

    with pytest.raises(SettingsError, match="same_name.*'one'.*'two'"):
        ConflictSettings().model_dump_env()


def test_dump_conflicting_names_case_folded():
    class CaseConflictSettings(BaseSettings):
        foo: str = 'a'
        FOO: str = 'b'

    with pytest.raises(SettingsError, match='conflicting environment variable names'):
        CaseConflictSettings().model_dump_env()


def test_dump_bytes(env):
    class BytesSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='b_')

        data: bytes = b'hello'

    settings = BytesSettings(data=b'world')
    assert settings.model_dump_env() == {'b_data': 'world'}
    env.set('b_data', 'world')
    assert BytesSettings() == settings


def test_dump_no_decode_field():
    class NoDecodeSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='nd_')

        raw: Annotated[list[int], NoDecode] = [1, 2]

    assert NoDecodeSettings().model_dump_env() == {'nd_raw': '[1, 2]'}


class VerifySub(BaseModel):
    host: str = 'localhost'
    ports: list[int] = []


class VerifySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='verify_', env_nested_delimiter='__', env_parse_none_str='null')

    name: str = 'default'
    count: int = 0
    ratio: float = 1.5
    flag: bool = False
    sub: VerifySub = VerifySub()
    mapping: dict[str, int] = {}
    maybe: str | None = None


def test_verify_lossless(env):
    settings = VerifySettings(
        name='abc',
        count=3,
        ratio=2.5,
        flag=True,
        sub=VerifySub(host='h', ports=[1, 2]),
        mapping={'a': 1},
        maybe=None,
    )
    dump = settings.model_dump_env(verify=True)
    assert dump == {
        'verify_name': 'abc',
        'verify_count': '3',
        'verify_ratio': '2.5',
        'verify_flag': 'true',
        'verify_sub__host': 'h',
        'verify_sub__ports': '[1,2]',
        'verify_mapping': '{"a":1}',
        'verify_maybe': 'null',
    }


def test_verify_isolated_from_parent_process_env(env):
    # the parent process environment must not leak into the verification rebuild
    env.set('VERIFY_NAME', 'junk-from-parent')
    settings = VerifySettings(name='real')
    dump = settings.model_dump_env(verify=True)
    assert dump['verify_name'] == 'real'


class VerifySecretSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='vs_')

    token: SecretStr = SecretStr('default-token')
    name: str = 'n'


def test_verify_secrets():
    settings = VerifySecretSettings(token=SecretStr('s3cr3t'), name='x')
    # secrets are omitted by default and excluded from the verification
    assert settings.model_dump_env(verify=True) == {'vs_name': 'x'}
    assert settings.model_dump_env(include_secrets=True, verify=True) == {'vs_token': 's3cr3t', 'vs_name': 'x'}


class AppendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='append_')

    name: str = 'x'

    @field_validator('name')
    @classmethod
    def _append(cls, v: str) -> str:
        return v + '!'


def test_verify_lists_non_reversible_validator_field():
    settings = AppendSettings()
    with pytest.raises(SettingsError, match='non-reversible fields: name'):
        settings.model_dump_env(verify=True)


MARKER = object()


class CustomSourceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='cs_')

    marker: Any = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        class StaticSource(PydanticBaseSettingsSource):
            def get_field_value(self, field, field_name):
                return None, '', False

            def __call__(self):
                # a bare object cannot survive the trip through env vars
                return {'marker': MARKER}

        return (StaticSource(settings_cls),)


def test_verify_lists_non_reversible_custom_source_field():
    settings = CustomSourceSettings()
    assert settings.marker is MARKER
    with pytest.raises(SettingsError, match='non-reversible fields: marker'):
        settings.model_dump_env(verify=True)


class VerifyNoneDefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='nd_')

    maybe: int | None = 5


def test_verify_lists_unrepresentable_none():
    settings = VerifyNoneDefaultSettings(maybe=None)
    assert settings.model_dump_env() == {}
    with pytest.raises(SettingsError, match='non-reversible fields: maybe'):
        settings.model_dump_env(verify=True)


def test_verify_requires_importable_class():
    class LocalSettings(BaseSettings):
        x: int = 1

    with pytest.raises(SettingsError, match='importable module'):
        LocalSettings().model_dump_env(verify=True)


def test_roundtrip_through_env_source(env):
    class RoundtripSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='rt_', env_nested_delimiter='__')

        name: str = 'n'
        sub: Sub = Sub()
        tags: list[str] = []

    settings = RoundtripSettings(name='svc', sub=Sub(host='h', port=9), tags=['a', 'b'])
    for key, value in settings.model_dump_env().items():
        env.set(key, value)
    assert RoundtripSettings() == settings
    assert RoundtripSettings().model_dump_env() == settings.model_dump_env()
