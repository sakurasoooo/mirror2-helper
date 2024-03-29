# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
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

class MessageDialogOptions(enum.IntFlag):
    NONE = 0
    ACCEPT_USER_INPUT_AFTER_DELAY = 0x1

class Placement(enum.IntEnum):
    DEFAULT = 0
    ABOVE = 1
    BELOW = 2
    LEFT = 3
    RIGHT = 4

class MessageDialog(_winrt.winrt_base):
    ...
    title: str
    options: MessageDialogOptions
    default_command_index: int
    content: str
    cancel_command_index: int
    commands: winrt.windows.foundation.collections.IVector[IUICommand]
    def show_async() -> winrt.windows.foundation.IAsyncOperation[IUICommand]:
        ...

class PopupMenu(_winrt.winrt_base):
    ...
    commands: winrt.windows.foundation.collections.IVector[IUICommand]
    def show_async(invocation_point: winrt.windows.foundation.Point) -> winrt.windows.foundation.IAsyncOperation[IUICommand]:
        ...
    def show_for_selection_async(selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[IUICommand]:
        ...
    def show_for_selection_async(selection: winrt.windows.foundation.Rect, preferred_placement: Placement) -> winrt.windows.foundation.IAsyncOperation[IUICommand]:
        ...

class UICommand(IUICommand, _winrt.winrt_base):
    ...
    label: str
    invoked: UICommandInvokedHandler
    id: _winrt.winrt_base

class UICommandSeparator(IUICommand, _winrt.winrt_base):
    ...
    label: str
    invoked: UICommandInvokedHandler
    id: _winrt.winrt_base

class IUICommand(_winrt.winrt_base):
    ...
    id: _winrt.winrt_base
    invoked: UICommandInvokedHandler
    label: str

UICommandInvokedHandler = typing.Callable[[IUICommand], None]

