# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Graphics.Printing.PrintTicket")

try:
    import winrt.windows.data.xml.dom
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

class PrintTicketFeatureSelectionType(enum.IntEnum):
    PICK_ONE = 0
    PICK_MANY = 1

class PrintTicketParameterDataType(enum.IntEnum):
    INTEGER = 0
    NUMERIC_STRING = 1
    STRING = 2

class PrintTicketValueType(enum.IntEnum):
    INTEGER = 0
    STRING = 1
    UNKNOWN = 2

PrintTicketCapabilities = _ns_module.PrintTicketCapabilities
PrintTicketFeature = _ns_module.PrintTicketFeature
PrintTicketOption = _ns_module.PrintTicketOption
PrintTicketParameterDefinition = _ns_module.PrintTicketParameterDefinition
PrintTicketParameterInitializer = _ns_module.PrintTicketParameterInitializer
PrintTicketValue = _ns_module.PrintTicketValue
WorkflowPrintTicket = _ns_module.WorkflowPrintTicket
WorkflowPrintTicketValidationResult = _ns_module.WorkflowPrintTicketValidationResult