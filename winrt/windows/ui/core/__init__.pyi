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

try:
    import winrt.windows.system
except Exception:
    pass

try:
    import winrt.windows.ui
except Exception:
    pass

try:
    import winrt.windows.ui.composition
except Exception:
    pass

try:
    import winrt.windows.ui.input
except Exception:
    pass

class AppViewBackButtonVisibility(enum.IntEnum):
    VISIBLE = 0
    COLLAPSED = 1
    DISABLED = 2

class CoreAcceleratorKeyEventType(enum.IntEnum):
    CHARACTER = 2
    DEAD_CHARACTER = 3
    KEY_DOWN = 0
    KEY_UP = 1
    SYSTEM_CHARACTER = 6
    SYSTEM_DEAD_CHARACTER = 7
    SYSTEM_KEY_DOWN = 4
    SYSTEM_KEY_UP = 5
    UNICODE_CHARACTER = 8

class CoreCursorType(enum.IntEnum):
    ARROW = 0
    CROSS = 1
    CUSTOM = 2
    HAND = 3
    HELP = 4
    I_BEAM = 5
    SIZE_ALL = 6
    SIZE_NORTHEAST_SOUTHWEST = 7
    SIZE_NORTH_SOUTH = 8
    SIZE_NORTHWEST_SOUTHEAST = 9
    SIZE_WEST_EAST = 10
    UNIVERSAL_NO = 11
    UP_ARROW = 12
    WAIT = 13
    PIN = 14
    PERSON = 15

class CoreDispatcherPriority(enum.IntEnum):
    IDLE = -2
    LOW = -1
    NORMAL = 0
    HIGH = 1

class CoreIndependentInputFilters(enum.IntFlag):
    NONE = 0
    MOUSE_BUTTON = 0x1
    MOUSE_WHEEL = 0x2
    MOUSE_HOVER = 0x4
    PEN_WITH_BARREL = 0x8
    PEN_INVERTED = 0x10

class CoreInputDeviceTypes(enum.IntFlag):
    NONE = 0
    TOUCH = 0x1
    PEN = 0x2
    MOUSE = 0x4

class CoreProcessEventsOption(enum.IntEnum):
    PROCESS_ONE_AND_ALL_PENDING = 0
    PROCESS_ONE_IF_PRESENT = 1
    PROCESS_UNTIL_QUIT = 2
    PROCESS_ALL_IF_PRESENT = 3

class CoreProximityEvaluationScore(enum.IntEnum):
    CLOSEST = 0
    FARTHEST = 2147483647

class CoreVirtualKeyStates(enum.IntFlag):
    NONE = 0
    DOWN = 0x1
    LOCKED = 0x2

class CoreWindowActivationMode(enum.IntEnum):
    NONE = 0
    DEACTIVATED = 1
    ACTIVATED_NOT_FOREGROUND = 2
    ACTIVATED_IN_FOREGROUND = 3

class CoreWindowActivationState(enum.IntEnum):
    CODE_ACTIVATED = 0
    DEACTIVATED = 1
    POINTER_ACTIVATED = 2

class CoreWindowFlowDirection(enum.IntEnum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1

class CorePhysicalKeyStatus(_winrt.winrt_base):
    ...

class CoreProximityEvaluation(_winrt.winrt_base):
    ...

class AcceleratorKeyEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    event_type: CoreAcceleratorKeyEventType
    key_status: CorePhysicalKeyStatus
    virtual_key: winrt.windows.system.VirtualKey
    device_id: str
    handled: bool

class AutomationProviderRequestedEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    automation_provider: _winrt.winrt_base
    handled: bool

class BackRequestedEventArgs(_winrt.winrt_base):
    ...
    handled: bool

class CharacterReceivedEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    key_code: int
    key_status: CorePhysicalKeyStatus
    handled: bool

class ClosestInteractiveBoundsRequestedEventArgs(_winrt.winrt_base):
    ...
    closest_interactive_bounds: winrt.windows.foundation.Rect
    pointer_position: winrt.windows.foundation.Point
    search_bounds: winrt.windows.foundation.Rect

class CoreAcceleratorKeys(ICoreAcceleratorKeys, _winrt.winrt_base):
    ...
    def add_accelerator_key_activated(handler: winrt.windows.foundation.TypedEventHandler[CoreDispatcher, AcceleratorKeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_accelerator_key_activated(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class CoreComponentInputSource(ICoreInputSourceBase, ICorePointerInputSource, ICorePointerInputSource2, _winrt.winrt_base):
    ...
    has_focus: bool
    is_input_enabled: bool
    dispatcher: CoreDispatcher
    pointer_cursor: CoreCursor
    has_capture: bool
    pointer_position: winrt.windows.foundation.Point
    dispatcher_queue: winrt.windows.system.DispatcherQueue
    def get_current_key_event_device_id() -> str:
        ...
    def get_current_key_state(virtual_key: winrt.windows.system.VirtualKey) -> CoreVirtualKeyStates:
        ...
    def release_pointer_capture() -> None:
        ...
    def set_pointer_capture() -> None:
        ...
    def add_input_enabled(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, InputEnabledEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_input_enabled(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_capture_lost(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_capture_lost(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_entered(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_entered(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_exited(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_exited(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_moved(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_moved(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_pressed(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_pressed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_released(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_wheel_changed(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_wheel_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_character_received(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, CharacterReceivedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_character_received(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_key_down(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, KeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_key_down(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_key_up(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, KeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_key_up(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_got_focus(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, CoreWindowEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_got_focus(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_lost_focus(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, CoreWindowEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_lost_focus(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_touch_hit_testing(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, TouchHitTestingEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_touch_hit_testing(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_closest_interactive_bounds_requested(handler: winrt.windows.foundation.TypedEventHandler[CoreComponentInputSource, ClosestInteractiveBoundsRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_closest_interactive_bounds_requested(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class CoreCursor(_winrt.winrt_base):
    ...
    id: int
    type: CoreCursorType

class CoreDispatcher(ICoreAcceleratorKeys, _winrt.winrt_base):
    ...
    has_thread_access: bool
    current_priority: CoreDispatcherPriority
    def process_events(options: CoreProcessEventsOption) -> None:
        ...
    def run_async(priority: CoreDispatcherPriority, agile_callback: DispatchedHandler) -> winrt.windows.foundation.IAsyncAction:
        ...
    def run_idle_async(agile_callback: IdleDispatchedHandler) -> winrt.windows.foundation.IAsyncAction:
        ...
    def should_yield() -> bool:
        ...
    def should_yield(priority: CoreDispatcherPriority) -> bool:
        ...
    def stop_process_events() -> None:
        ...
    def try_run_async(priority: CoreDispatcherPriority, agile_callback: DispatchedHandler) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_run_idle_async(agile_callback: IdleDispatchedHandler) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def add_accelerator_key_activated(handler: winrt.windows.foundation.TypedEventHandler[CoreDispatcher, AcceleratorKeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_accelerator_key_activated(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class CoreIndependentInputSource(ICoreInputSourceBase, ICorePointerInputSource, ICorePointerInputSource2, ICorePointerRedirector, _winrt.winrt_base):
    ...
    is_input_enabled: bool
    dispatcher: CoreDispatcher
    pointer_cursor: CoreCursor
    has_capture: bool
    pointer_position: winrt.windows.foundation.Point
    dispatcher_queue: winrt.windows.system.DispatcherQueue
    def release_pointer_capture() -> None:
        ...
    def set_pointer_capture() -> None:
        ...
    def add_input_enabled(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, InputEnabledEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_input_enabled(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_capture_lost(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_capture_lost(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_entered(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_entered(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_exited(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_exited(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_moved(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_moved(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_pressed(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_pressed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_released(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_wheel_changed(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_wheel_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_routed_away(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_away(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_routed_released(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_routed_to(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_to(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class CoreIndependentInputSourceController(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    is_transparent_for_uncontrolled_input: bool
    is_palm_rejection_enabled: bool
    source: CoreIndependentInputSource
    def close() -> None:
        ...
    def create_for_i_visual_element(visual_element: winrt.windows.ui.composition.IVisualElement) -> CoreIndependentInputSourceController:
        ...
    def create_for_visual(visual: winrt.windows.ui.composition.Visual) -> CoreIndependentInputSourceController:
        ...
    def set_controlled_input(input_types: CoreInputDeviceTypes) -> None:
        ...
    def set_controlled_input(input_types: CoreInputDeviceTypes, required: CoreIndependentInputFilters, excluded: CoreIndependentInputFilters) -> None:
        ...

class CoreWindow(ICoreWindow, ICorePointerRedirector, _winrt.winrt_base):
    ...
    pointer_position: winrt.windows.foundation.Point
    pointer_cursor: CoreCursor
    flow_direction: CoreWindowFlowDirection
    is_input_enabled: bool
    dispatcher: CoreDispatcher
    automation_host_provider: _winrt.winrt_base
    bounds: winrt.windows.foundation.Rect
    custom_properties: winrt.windows.foundation.collections.IPropertySet
    visible: bool
    activation_mode: CoreWindowActivationMode
    dispatcher_queue: winrt.windows.system.DispatcherQueue
    u_i_context: winrt.windows.ui.UIContext
    def activate() -> None:
        ...
    def close() -> None:
        ...
    def get_async_key_state(virtual_key: winrt.windows.system.VirtualKey) -> CoreVirtualKeyStates:
        ...
    def get_current_key_event_device_id() -> str:
        ...
    def get_for_current_thread() -> CoreWindow:
        ...
    def get_key_state(virtual_key: winrt.windows.system.VirtualKey) -> CoreVirtualKeyStates:
        ...
    def release_pointer_capture() -> None:
        ...
    def set_pointer_capture() -> None:
        ...
    def add_activated(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, WindowActivatedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_activated(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_automation_provider_requested(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, AutomationProviderRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_automation_provider_requested(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_character_received(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, CharacterReceivedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_character_received(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_closed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, CoreWindowEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_closed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_input_enabled(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, InputEnabledEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_input_enabled(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_key_down(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, KeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_key_down(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_key_up(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, KeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_key_up(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_capture_lost(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_capture_lost(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_entered(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_entered(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_exited(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_exited(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_moved(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_moved(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_pressed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_pressed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_released(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_wheel_changed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_wheel_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_size_changed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, WindowSizeChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_size_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_touch_hit_testing(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, TouchHitTestingEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_touch_hit_testing(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_visibility_changed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, VisibilityChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_visibility_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_routed_away(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_away(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_routed_released(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_routed_to(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_to(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_closest_interactive_bounds_requested(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, ClosestInteractiveBoundsRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_closest_interactive_bounds_requested(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_resize_completed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_resize_completed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_resize_started(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_resize_started(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class CoreWindowEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    handled: bool

class CoreWindowResizeManager(_winrt.winrt_base):
    ...
    should_wait_for_layout_completion: bool
    def get_for_current_view() -> CoreWindowResizeManager:
        ...
    def notify_layout_completed() -> None:
        ...

class IdleDispatchedHandlerArgs(_winrt.winrt_base):
    ...
    is_dispatcher_idle: bool

class InputEnabledEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    handled: bool
    input_enabled: bool

class KeyEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    handled: bool
    key_status: CorePhysicalKeyStatus
    virtual_key: winrt.windows.system.VirtualKey
    device_id: str

class PointerEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    handled: bool
    current_point: winrt.windows.ui.input.PointerPoint
    key_modifiers: winrt.windows.system.VirtualKeyModifiers
    def get_intermediate_points() -> winrt.windows.foundation.collections.IVector[winrt.windows.ui.input.PointerPoint]:
        ...

class SystemNavigationManager(_winrt.winrt_base):
    ...
    app_view_back_button_visibility: AppViewBackButtonVisibility
    def get_for_current_view() -> SystemNavigationManager:
        ...
    def add_back_requested(handler: winrt.windows.foundation.EventHandler[BackRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_back_requested(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class TouchHitTestingEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    handled: bool
    proximity_evaluation: CoreProximityEvaluation
    bounding_box: winrt.windows.foundation.Rect
    point: winrt.windows.foundation.Point
    def evaluate_proximity(control_bounding_box: winrt.windows.foundation.Rect) -> CoreProximityEvaluation:
        ...

class VisibilityChangedEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    handled: bool
    visible: bool

class WindowActivatedEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    handled: bool
    window_activation_state: CoreWindowActivationState

class WindowSizeChangedEventArgs(ICoreWindowEventArgs, _winrt.winrt_base):
    ...
    handled: bool
    size: winrt.windows.foundation.Size

class ICoreAcceleratorKeys(_winrt.winrt_base):
    ...
    def add_accelerator_key_activated(handler: winrt.windows.foundation.TypedEventHandler[CoreDispatcher, AcceleratorKeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_accelerator_key_activated(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ICoreInputSourceBase(_winrt.winrt_base):
    ...
    dispatcher: CoreDispatcher
    is_input_enabled: bool
    def add_input_enabled(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, InputEnabledEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_input_enabled(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ICorePointerInputSource(_winrt.winrt_base):
    ...
    has_capture: bool
    pointer_cursor: CoreCursor
    pointer_position: winrt.windows.foundation.Point
    def release_pointer_capture() -> None:
        ...
    def set_pointer_capture() -> None:
        ...
    def add_pointer_capture_lost(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_capture_lost(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_entered(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_entered(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_exited(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_exited(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_moved(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_moved(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_pressed(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_pressed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_released(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_wheel_changed(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_wheel_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ICorePointerInputSource2(ICorePointerInputSource, _winrt.winrt_base):
    ...
    dispatcher_queue: winrt.windows.system.DispatcherQueue
    has_capture: bool
    pointer_cursor: CoreCursor
    pointer_position: winrt.windows.foundation.Point
    def release_pointer_capture() -> None:
        ...
    def set_pointer_capture() -> None:
        ...
    def add_pointer_capture_lost(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_capture_lost(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_entered(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_entered(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_exited(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_exited(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_moved(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_moved(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_pressed(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_pressed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_released(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_wheel_changed(handler: winrt.windows.foundation.TypedEventHandler[_winrt.winrt_base, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_wheel_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ICorePointerRedirector(_winrt.winrt_base):
    ...
    def add_pointer_routed_away(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_away(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_routed_released(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_routed_to(handler: winrt.windows.foundation.TypedEventHandler[ICorePointerRedirector, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_routed_to(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ICoreWindow(_winrt.winrt_base):
    ...
    automation_host_provider: _winrt.winrt_base
    bounds: winrt.windows.foundation.Rect
    custom_properties: winrt.windows.foundation.collections.IPropertySet
    dispatcher: CoreDispatcher
    flow_direction: CoreWindowFlowDirection
    is_input_enabled: bool
    pointer_cursor: CoreCursor
    pointer_position: winrt.windows.foundation.Point
    visible: bool
    def activate() -> None:
        ...
    def close() -> None:
        ...
    def get_async_key_state(virtual_key: winrt.windows.system.VirtualKey) -> CoreVirtualKeyStates:
        ...
    def get_key_state(virtual_key: winrt.windows.system.VirtualKey) -> CoreVirtualKeyStates:
        ...
    def release_pointer_capture() -> None:
        ...
    def set_pointer_capture() -> None:
        ...
    def add_activated(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, WindowActivatedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_activated(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_automation_provider_requested(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, AutomationProviderRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_automation_provider_requested(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_character_received(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, CharacterReceivedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_character_received(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_closed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, CoreWindowEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_closed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_input_enabled(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, InputEnabledEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_input_enabled(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_key_down(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, KeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_key_down(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_key_up(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, KeyEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_key_up(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_capture_lost(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_capture_lost(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_entered(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_entered(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_exited(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_exited(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_moved(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_moved(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_pressed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_pressed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_released(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_released(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_pointer_wheel_changed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, PointerEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_pointer_wheel_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_size_changed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, WindowSizeChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_size_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_touch_hit_testing(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, TouchHitTestingEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_touch_hit_testing(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_visibility_changed(handler: winrt.windows.foundation.TypedEventHandler[CoreWindow, VisibilityChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_visibility_changed(cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ICoreWindowEventArgs(_winrt.winrt_base):
    ...
    handled: bool

class IInitializeWithCoreWindow(_winrt.winrt_base):
    ...
    def initialize(window: CoreWindow) -> None:
        ...

DispatchedHandler = typing.Callable[[], None]

IdleDispatchedHandler = typing.Callable[[IdleDispatchedHandlerArgs], None]
