# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.applicationmodel.appservice
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

try:
    import winrt.windows.networking.sockets
except Exception:
    pass

try:
    import winrt.windows.web.http
except Exception:
    pass

class DevicePortalConnectionClosedReason(enum.IntEnum):
    UNKNOWN = 0
    RESOURCE_LIMITS_EXCEEDED = 1
    PROTOCOL_ERROR = 2
    NOT_AUTHORIZED = 3
    USER_NOT_PRESENT = 4
    SERVICE_TERMINATED = 5

class DevicePortalConnection(_winrt.winrt_base):
    ...
    def get_for_app_service_connection(app_service_connection: winrt.windows.applicationmodel.appservice.AppServiceConnection) -> DevicePortalConnection:
        ...
    def get_server_message_web_socket_for_request(request: winrt.windows.web.http.HttpRequestMessage) -> winrt.windows.networking.sockets.ServerMessageWebSocket:
        ...
    def get_server_message_web_socket_for_request(request: winrt.windows.web.http.HttpRequestMessage, message_type: winrt.windows.networking.sockets.SocketMessageType, protocol: str) -> winrt.windows.networking.sockets.ServerMessageWebSocket:
        ...
    def get_server_message_web_socket_for_request(request: winrt.windows.web.http.HttpRequestMessage, message_type: winrt.windows.networking.sockets.SocketMessageType, protocol: str, outbound_buffer_size_in_bytes: int, max_message_size: int, receive_mode: winrt.windows.networking.sockets.MessageWebSocketReceiveMode) -> winrt.windows.networking.sockets.ServerMessageWebSocket:
        ...
    def get_server_stream_web_socket_for_request(request: winrt.windows.web.http.HttpRequestMessage) -> winrt.windows.networking.sockets.ServerStreamWebSocket:
        ...
    def get_server_stream_web_socket_for_request(request: winrt.windows.web.http.HttpRequestMessage, protocol: str, outbound_buffer_size_in_bytes: int, no_delay: bool) -> winrt.windows.networking.sockets.ServerStreamWebSocket:
        ...
    def add_closed(handler: winrt.windows.foundation.TypedEventHandler[DevicePortalConnection, DevicePortalConnectionClosedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_closed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_request_received(handler: winrt.windows.foundation.TypedEventHandler[DevicePortalConnection, DevicePortalConnectionRequestReceivedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_request_received(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class DevicePortalConnectionClosedEventArgs(_winrt.winrt_base):
    ...
    reason: DevicePortalConnectionClosedReason

class DevicePortalConnectionRequestReceivedEventArgs(_winrt.winrt_base):
    ...
    request_message: winrt.windows.web.http.HttpRequestMessage
    response_message: winrt.windows.web.http.HttpResponseMessage
    is_web_socket_upgrade_request: bool
    web_socket_protocols_requested: winrt.windows.foundation.collections.IVectorView[str]
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...
