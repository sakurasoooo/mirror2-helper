# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.foundation
except Exception:
    pass

class OemSupportInfo(_winrt.winrt_base):
    ...
    support_app_link: winrt.windows.foundation.Uri
    support_link: winrt.windows.foundation.Uri
    support_provider: str

class SmbiosInformation(_winrt.winrt_base):
    ...
    serial_number: str

class SystemSupportDeviceInfo(_winrt.winrt_base):
    ...
    friendly_name: str
    operating_system: str
    system_firmware_version: str
    system_hardware_version: str
    system_manufacturer: str
    system_product_name: str
    system_sku: str

class SystemSupportInfo(_winrt.winrt_base):
    ...
    local_system_edition: str
    oem_support_info: OemSupportInfo
    local_device_info: SystemSupportDeviceInfo

