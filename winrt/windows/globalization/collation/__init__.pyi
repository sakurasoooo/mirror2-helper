# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class CharacterGrouping(_winrt.winrt_base):
    ...
    first: str
    label: str

class CharacterGroupings(winrt.windows.foundation.collections.IVectorView[CharacterGrouping], winrt.windows.foundation.collections.IIterable[CharacterGrouping], _winrt.winrt_base):
    ...
    size: int
    def first() -> winrt.windows.foundation.collections.IIterator[CharacterGrouping]:
        ...
    def get_at(index: int) -> CharacterGrouping:
        ...
    def get_many(start_index: int, items_size: int) -> typing.Tuple[int, typing.List[CharacterGrouping]]:
        ...
    def index_of(value: CharacterGrouping) -> typing.Tuple[bool, int]:
        ...
    def lookup(text: str) -> str:
        ...
