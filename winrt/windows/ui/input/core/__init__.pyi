# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.applicationmodel.core
except Exception:
    pass

try:
    import winrt.windows.system
except Exception:
    pass

try:
    import winrt.windows.ui.core
except Exception:
    pass

try:
    import winrt.windows.ui.input
except Exception:
    pass

class RadialControllerIndependentInputSource(_winrt.winrt_base):
    ...
    controller: winrt.windows.ui.input.RadialController
    dispatcher: winrt.windows.ui.core.CoreDispatcher
    dispatcher_queue: winrt.windows.system.DispatcherQueue
    def create_for_view(view: winrt.windows.applicationmodel.core.CoreApplicationView) -> RadialControllerIndependentInputSource:
        ...
