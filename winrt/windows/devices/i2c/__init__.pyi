# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.devices.i2c.provider
except Exception:
    pass

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class I2cBusSpeed(enum.IntEnum):
    STANDARD_MODE = 0
    FAST_MODE = 1

class I2cSharingMode(enum.IntEnum):
    EXCLUSIVE = 0
    SHARED = 1

class I2cTransferStatus(enum.IntEnum):
    FULL_TRANSFER = 0
    PARTIAL_TRANSFER = 1
    SLAVE_ADDRESS_NOT_ACKNOWLEDGED = 2
    CLOCK_STRETCH_TIMEOUT = 3
    UNKNOWN_ERROR = 4

class I2cTransferResult(_winrt.winrt_base):
    ...

class I2cConnectionSettings(_winrt.winrt_base):
    ...
    slave_address: int
    sharing_mode: I2cSharingMode
    bus_speed: I2cBusSpeed

class I2cController(_winrt.winrt_base):
    ...
    def get_controllers_async(provider: winrt.windows.devices.i2c.provider.II2cProvider) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[I2cController]]:
        ...
    def get_default_async() -> winrt.windows.foundation.IAsyncOperation[I2cController]:
        ...
    def get_device(settings: I2cConnectionSettings) -> I2cDevice:
        ...

class I2cDevice(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    connection_settings: I2cConnectionSettings
    device_id: str
    def close() -> None:
        ...
    def from_id_async(device_id: str, settings: I2cConnectionSettings) -> winrt.windows.foundation.IAsyncOperation[I2cDevice]:
        ...
    def get_device_selector() -> str:
        ...
    def get_device_selector(friendly_name: str) -> str:
        ...
    def read(buffer_size: int) -> typing.List[int]:
        ...
    def read_partial(buffer_size: int) -> typing.Tuple[I2cTransferResult, typing.List[int]]:
        ...
    def write(buffer: typing.Sequence[int]) -> None:
        ...
    def write_partial(buffer: typing.Sequence[int]) -> I2cTransferResult:
        ...
    def write_read(write_buffer: typing.Sequence[int], read_buffer_size: int) -> typing.List[int]:
        ...
    def write_read_partial(write_buffer: typing.Sequence[int], read_buffer_size: int) -> typing.Tuple[I2cTransferResult, typing.List[int]]:
        ...

class II2cDeviceStatics(_winrt.winrt_base):
    ...
    def from_id_async(device_id: str, settings: I2cConnectionSettings) -> winrt.windows.foundation.IAsyncOperation[I2cDevice]:
        ...
    def get_device_selector() -> str:
        ...
    def get_device_selector(friendly_name: str) -> str:
        ...

