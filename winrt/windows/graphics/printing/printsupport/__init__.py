# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Graphics.Printing.PrintSupport")

try:
    import winrt.windows.applicationmodel
except Exception:
    pass

try:
    import winrt.windows.applicationmodel.activation
except Exception:
    pass

try:
    import winrt.windows.data.xml.dom
except Exception:
    pass

try:
    import winrt.windows.devices.printers
except Exception:
    pass

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.graphics.printing.printticket
except Exception:
    pass

try:
    import winrt.windows.system
except Exception:
    pass

class SettingsLaunchKind(enum.IntEnum):
    JOB_PRINT_TICKET = 0
    USER_DEFAULT_PRINT_TICKET = 1

class WorkflowPrintTicketValidationStatus(enum.IntEnum):
    RESOLVED = 0
    CONFLICTING = 1
    INVALID = 2

PrintSupportExtensionSession = _ns_module.PrintSupportExtensionSession
PrintSupportExtensionTriggerDetails = _ns_module.PrintSupportExtensionTriggerDetails
PrintSupportPrintDeviceCapabilitiesChangedEventArgs = _ns_module.PrintSupportPrintDeviceCapabilitiesChangedEventArgs
PrintSupportPrintTicketValidationRequestedEventArgs = _ns_module.PrintSupportPrintTicketValidationRequestedEventArgs
PrintSupportSessionInfo = _ns_module.PrintSupportSessionInfo
PrintSupportSettingsActivatedEventArgs = _ns_module.PrintSupportSettingsActivatedEventArgs
PrintSupportSettingsUISession = _ns_module.PrintSupportSettingsUISession
