# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class JsonErrorStatus(enum.IntEnum):
    UNKNOWN = 0
    INVALID_JSON_STRING = 1
    INVALID_JSON_NUMBER = 2
    JSON_VALUE_NOT_FOUND = 3
    IMPLEMENTATION_LIMIT = 4

class JsonValueType(enum.IntEnum):
    NULL = 0
    BOOLEAN = 1
    NUMBER = 2
    STRING = 3
    ARRAY = 4
    OBJECT = 5

class JsonArray(IJsonValue, winrt.windows.foundation.collections.IVector[IJsonValue], winrt.windows.foundation.collections.IIterable[IJsonValue], winrt.windows.foundation.IStringable, _winrt.winrt_base):
    ...
    value_type: JsonValueType
    size: int
    def append(value: IJsonValue) -> None:
        ...
    def clear() -> None:
        ...
    def first() -> winrt.windows.foundation.collections.IIterator[IJsonValue]:
        ...
    def get_array() -> JsonArray:
        ...
    def get_array_at(index: int) -> JsonArray:
        ...
    def get_at(index: int) -> IJsonValue:
        ...
    def get_boolean() -> bool:
        ...
    def get_boolean_at(index: int) -> bool:
        ...
    def get_many(start_index: int, items_size: int) -> typing.Tuple[int, typing.List[IJsonValue]]:
        ...
    def get_number() -> float:
        ...
    def get_number_at(index: int) -> float:
        ...
    def get_object() -> JsonObject:
        ...
    def get_object_at(index: int) -> JsonObject:
        ...
    def get_string() -> str:
        ...
    def get_string_at(index: int) -> str:
        ...
    def get_view() -> winrt.windows.foundation.collections.IVectorView[IJsonValue]:
        ...
    def index_of(value: IJsonValue) -> typing.Tuple[bool, int]:
        ...
    def insert_at(index: int, value: IJsonValue) -> None:
        ...
    def parse(input: str) -> JsonArray:
        ...
    def remove_at(index: int) -> None:
        ...
    def remove_at_end() -> None:
        ...
    def replace_all(items: typing.Sequence[IJsonValue]) -> None:
        ...
    def set_at(index: int, value: IJsonValue) -> None:
        ...
    def stringify() -> str:
        ...
    def to_string() -> str:
        ...
    def try_parse(input: str) -> typing.Tuple[bool, JsonArray]:
        ...

class JsonError(_winrt.winrt_base):
    ...
    def get_json_status(hresult: int) -> JsonErrorStatus:
        ...

class JsonObject(IJsonValue, winrt.windows.foundation.collections.IMap[str, IJsonValue], winrt.windows.foundation.collections.IIterable[winrt.windows.foundation.collections.IKeyValuePair[str, IJsonValue]], winrt.windows.foundation.IStringable, _winrt.winrt_base):
    ...
    value_type: JsonValueType
    size: int
    def clear() -> None:
        ...
    def first() -> winrt.windows.foundation.collections.IIterator[winrt.windows.foundation.collections.IKeyValuePair[str, IJsonValue]]:
        ...
    def get_array() -> JsonArray:
        ...
    def get_boolean() -> bool:
        ...
    def get_named_array(name: str) -> JsonArray:
        ...
    def get_named_array(name: str, default_value: JsonArray) -> JsonArray:
        ...
    def get_named_boolean(name: str) -> bool:
        ...
    def get_named_boolean(name: str, default_value: bool) -> bool:
        ...
    def get_named_number(name: str) -> float:
        ...
    def get_named_number(name: str, default_value: float) -> float:
        ...
    def get_named_object(name: str) -> JsonObject:
        ...
    def get_named_object(name: str, default_value: JsonObject) -> JsonObject:
        ...
    def get_named_string(name: str) -> str:
        ...
    def get_named_string(name: str, default_value: str) -> str:
        ...
    def get_named_value(name: str) -> JsonValue:
        ...
    def get_named_value(name: str, default_value: JsonValue) -> JsonValue:
        ...
    def get_number() -> float:
        ...
    def get_object() -> JsonObject:
        ...
    def get_string() -> str:
        ...
    def get_view() -> winrt.windows.foundation.collections.IMapView[str, IJsonValue]:
        ...
    def has_key(key: str) -> bool:
        ...
    def insert(key: str, value: IJsonValue) -> bool:
        ...
    def lookup(key: str) -> IJsonValue:
        ...
    def parse(input: str) -> JsonObject:
        ...
    def remove(key: str) -> None:
        ...
    def set_named_value(name: str, value: IJsonValue) -> None:
        ...
    def stringify() -> str:
        ...
    def to_string() -> str:
        ...
    def try_parse(input: str) -> typing.Tuple[bool, JsonObject]:
        ...

class JsonValue(IJsonValue, winrt.windows.foundation.IStringable, _winrt.winrt_base):
    ...
    value_type: JsonValueType
    def create_boolean_value(input: bool) -> JsonValue:
        ...
    def create_null_value() -> JsonValue:
        ...
    def create_number_value(input: float) -> JsonValue:
        ...
    def create_string_value(input: str) -> JsonValue:
        ...
    def get_array() -> JsonArray:
        ...
    def get_boolean() -> bool:
        ...
    def get_number() -> float:
        ...
    def get_object() -> JsonObject:
        ...
    def get_string() -> str:
        ...
    def parse(input: str) -> JsonValue:
        ...
    def stringify() -> str:
        ...
    def to_string() -> str:
        ...
    def try_parse(input: str) -> typing.Tuple[bool, JsonValue]:
        ...

class IJsonValue(_winrt.winrt_base):
    ...
    value_type: JsonValueType
    def get_array() -> JsonArray:
        ...
    def get_boolean() -> bool:
        ...
    def get_number() -> float:
        ...
    def get_object() -> JsonObject:
        ...
    def get_string() -> str:
        ...
    def stringify() -> str:
        ...

