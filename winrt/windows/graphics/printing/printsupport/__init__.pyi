# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
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

class PrintSupportExtensionSession(_winrt.winrt_base):
    ...
    printer: winrt.windows.devices.printers.IppPrintDevice
    def start() -> None:
        ...
    def add_print_device_capabilities_changed(handler: winrt.windows.foundation.TypedEventHandler[PrintSupportExtensionSession, PrintSupportPrintDeviceCapabilitiesChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_print_device_capabilities_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_print_ticket_validation_requested(handler: winrt.windows.foundation.TypedEventHandler[PrintSupportExtensionSession, PrintSupportPrintTicketValidationRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_print_ticket_validation_requested(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class PrintSupportExtensionTriggerDetails(_winrt.winrt_base):
    ...
    session: PrintSupportExtensionSession

class PrintSupportPrintDeviceCapabilitiesChangedEventArgs(_winrt.winrt_base):
    ...
    def get_current_print_device_capabilities() -> winrt.windows.data.xml.dom.XmlDocument:
        ...
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...
    def update_print_device_capabilities(updated_pdc: winrt.windows.data.xml.dom.XmlDocument) -> None:
        ...

class PrintSupportPrintTicketValidationRequestedEventArgs(_winrt.winrt_base):
    ...
    print_ticket: winrt.windows.graphics.printing.printticket.WorkflowPrintTicket
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...
    def set_print_ticket_validation_status(status: WorkflowPrintTicketValidationStatus) -> None:
        ...

class PrintSupportSessionInfo(_winrt.winrt_base):
    ...
    printer: winrt.windows.devices.printers.IppPrintDevice
    source_app_info: winrt.windows.applicationmodel.AppInfo

class PrintSupportSettingsActivatedEventArgs(winrt.windows.applicationmodel.activation.IActivatedEventArgs, winrt.windows.applicationmodel.activation.IActivatedEventArgsWithUser, _winrt.winrt_base):
    ...
    kind: winrt.windows.applicationmodel.activation.ActivationKind
    previous_execution_state: winrt.windows.applicationmodel.activation.ApplicationExecutionState
    splash_screen: winrt.windows.applicationmodel.activation.SplashScreen
    user: winrt.windows.system.User
    session: PrintSupportSettingsUISession
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...

class PrintSupportSettingsUISession(_winrt.winrt_base):
    ...
    document_title: str
    launch_kind: SettingsLaunchKind
    session_info: PrintSupportSessionInfo
    session_print_ticket: winrt.windows.graphics.printing.printticket.WorkflowPrintTicket
    def update_print_ticket(print_ticket: winrt.windows.graphics.printing.printticket.WorkflowPrintTicket) -> None:
        ...

