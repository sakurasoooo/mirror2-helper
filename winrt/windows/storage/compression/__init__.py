# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Storage.Compression")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class CompressAlgorithm(enum.IntEnum):
    INVALID_ALGORITHM = 0
    NULL_ALGORITHM = 1
    MSZIP = 2
    XPRESS = 3
    XPRESS_HUFF = 4
    LZMS = 5

Compressor = _ns_module.Compressor
Decompressor = _ns_module.Decompressor