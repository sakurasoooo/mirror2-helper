# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class CustomSensor(_winrt.winrt_base):
    ...
    report_interval: int
    device_id: str
    minimum_report_interval: int
    report_latency: int
    max_batch_size: int
    def from_id_async(sensor_id: str) -> winrt.windows.foundation.IAsyncOperation[CustomSensor]:
        ...
    def get_current_reading() -> CustomSensorReading:
        ...
    def get_device_selector(interface_id: uuid.UUID) -> str:
        ...
    def add_reading_changed(handler: winrt.windows.foundation.TypedEventHandler[CustomSensor, CustomSensorReadingChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_reading_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class CustomSensorReading(_winrt.winrt_base):
    ...
    properties: winrt.windows.foundation.collections.IMapView[str, _winrt.winrt_base]
    timestamp: winrt.windows.foundation.DateTime
    performance_count: typing.Optional[winrt.windows.foundation.TimeSpan]

class CustomSensorReadingChangedEventArgs(_winrt.winrt_base):
    ...
    reading: CustomSensorReading
