# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.applicationmodel.activation
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
    import winrt.windows.networking
except Exception:
    pass

try:
    import winrt.windows.networking.sockets
except Exception:
    pass

try:
    import winrt.windows.security.credentials
except Exception:
    pass

try:
    import winrt.windows.security.cryptography.certificates
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

try:
    import winrt.windows.system
except Exception:
    pass

class VpnAppIdType(enum.IntEnum):
    PACKAGE_FAMILY_NAME = 0
    FULLY_QUALIFIED_BINARY_NAME = 1
    FILE_PATH = 2

class VpnAuthenticationMethod(enum.IntEnum):
    MSCHAPV2 = 0
    EAP = 1
    CERTIFICATE = 2
    PRESHARED_KEY = 3

class VpnChannelActivityEventType(enum.IntEnum):
    IDLE = 0
    ACTIVE = 1

class VpnChannelRequestCredentialsOptions(enum.IntFlag):
    NONE = 0
    RETRYING = 0x1
    USE_FOR_SINGLE_SIGN_IN = 0x2

class VpnCredentialType(enum.IntEnum):
    USERNAME_PASSWORD = 0
    USERNAME_OTP_PIN = 1
    USERNAME_PASSWORD_AND_PIN = 2
    USERNAME_PASSWORD_CHANGE = 3
    SMART_CARD = 4
    PROTECTED_CERTIFICATE = 5
    UN_PROTECTED_CERTIFICATE = 6

class VpnDataPathType(enum.IntEnum):
    SEND = 0
    RECEIVE = 1

class VpnDomainNameType(enum.IntEnum):
    SUFFIX = 0
    FULLY_QUALIFIED = 1
    RESERVED = 65535

class VpnIPProtocol(enum.IntEnum):
    NONE = 0
    TCP = 6
    UDP = 17
    ICMP = 1
    IPV6_ICMP = 58
    IGMP = 2
    PGM = 113

class VpnManagementConnectionStatus(enum.IntEnum):
    DISCONNECTED = 0
    DISCONNECTING = 1
    CONNECTED = 2
    CONNECTING = 3

class VpnManagementErrorStatus(enum.IntEnum):
    OK = 0
    OTHER = 1
    INVALID_XML_SYNTAX = 2
    PROFILE_NAME_TOO_LONG = 3
    PROFILE_INVALID_APP_ID = 4
    ACCESS_DENIED = 5
    CANNOT_FIND_PROFILE = 6
    ALREADY_DISCONNECTING = 7
    ALREADY_CONNECTED = 8
    GENERAL_AUTHENTICATION_FAILURE = 9
    EAP_FAILURE = 10
    SMART_CARD_FAILURE = 11
    CERTIFICATE_FAILURE = 12
    SERVER_CONFIGURATION = 13
    NO_CONNECTION = 14
    SERVER_CONNECTION = 15
    USER_NAME_PASSWORD = 16
    DNS_NOT_RESOLVABLE = 17
    INVALID_I_P = 18

class VpnNativeProtocolType(enum.IntEnum):
    PPTP = 0
    L2TP = 1
    IPSEC_IKEV2 = 2

class VpnPacketBufferStatus(enum.IntEnum):
    OK = 0
    INVALID_BUFFER_SIZE = 1

class VpnRoutingPolicyType(enum.IntEnum):
    SPLIT_ROUTING = 0
    FORCE_ALL_TRAFFIC_OVER_VPN = 1

class VpnAppId(_winrt.winrt_base):
    ...
    value: str
    type: VpnAppIdType

class VpnChannel(_winrt.winrt_base):
    ...
    plug_in_context: _winrt.winrt_base
    configuration: VpnChannelConfiguration
    id: int
    system_health: VpnSystemHealth
    current_request_transport_context: _winrt.winrt_base
    def activate_foreground(package_relative_app_id: str, shared_context: winrt.windows.foundation.collections.ValueSet) -> winrt.windows.foundation.collections.ValueSet:
        ...
    def add_and_associate_transport(transport: _winrt.winrt_base, context: _winrt.winrt_base) -> None:
        ...
    def append_vpn_receive_packet_buffer(decapsulated_packet_buffer: VpnPacketBuffer) -> None:
        ...
    def append_vpn_send_packet_buffer(encapsulated_packet_buffer: VpnPacketBuffer) -> None:
        ...
    def associate_transport(main_outer_tunnel_transport: _winrt.winrt_base, optional_outer_tunnel_transport: _winrt.winrt_base) -> None:
        ...
    def flush_vpn_receive_packet_buffers() -> None:
        ...
    def flush_vpn_send_packet_buffers() -> None:
        ...
    def get_slot_type_for_transport_context(context: _winrt.winrt_base) -> winrt.windows.networking.sockets.ControlChannelTriggerStatus:
        ...
    def get_vpn_receive_packet_buffer() -> VpnPacketBuffer:
        ...
    def get_vpn_send_packet_buffer() -> VpnPacketBuffer:
        ...
    def log_diagnostic_message(message: str) -> None:
        ...
    def process_event_async(third_party_plug_in: _winrt.winrt_base, event: _winrt.winrt_base) -> None:
        ...
    def replace_and_associate_transport(transport: _winrt.winrt_base, context: _winrt.winrt_base) -> None:
        ...
    def request_credentials(cred_type: VpnCredentialType, is_retry: bool, is_single_sign_on_credential: bool, certificate: winrt.windows.security.cryptography.certificates.Certificate) -> VpnPickedCredential:
        ...
    def request_credentials_async(cred_type: VpnCredentialType) -> winrt.windows.foundation.IAsyncOperation[VpnCredential]:
        ...
    def request_credentials_async(cred_type: VpnCredentialType, cred_options: int) -> winrt.windows.foundation.IAsyncOperation[VpnCredential]:
        ...
    def request_credentials_async(cred_type: VpnCredentialType, cred_options: int, certificate: winrt.windows.security.cryptography.certificates.Certificate) -> winrt.windows.foundation.IAsyncOperation[VpnCredential]:
        ...
    def request_custom_prompt(custom_prompt: winrt.windows.foundation.collections.IVectorView[IVpnCustomPrompt]) -> None:
        ...
    def request_custom_prompt_async(custom_prompt_element: winrt.windows.foundation.collections.IVectorView[IVpnCustomPromptElement]) -> winrt.windows.foundation.IAsyncAction:
        ...
    def request_vpn_packet_buffer(type: VpnDataPathType) -> VpnPacketBuffer:
        ...
    def set_allowed_ssl_tls_versions(tunnel_transport: _winrt.winrt_base, use_tls12: bool) -> None:
        ...
    def set_error_message(message: str) -> None:
        ...
    def start(assigned_client_i_pv4list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName], assigned_client_i_pv6list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName], vpn_interface_id: VpnInterfaceId, route_scope: VpnRouteAssignment, namespace_scope: VpnNamespaceAssignment, mtu_size: int, max_frame_size: int, optimize_for_low_cost_network: bool, main_outer_tunnel_transport: _winrt.winrt_base, optional_outer_tunnel_transport: _winrt.winrt_base) -> None:
        ...
    def start_existing_transports(assigned_client_i_pv4list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName], assigned_client_i_pv6list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName], vpn_interface_id: VpnInterfaceId, assigned_routes: VpnRouteAssignment, assigned_domain_name: VpnDomainNameAssignment, mtu_size: int, max_frame_size: int, reserved: bool) -> None:
        ...
    def start_reconnecting_transport(transport: _winrt.winrt_base, context: _winrt.winrt_base) -> None:
        ...
    def start_with_main_transport(assigned_client_i_pv4list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName], assigned_client_i_pv6list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName], vpn_interface_id: VpnInterfaceId, assigned_routes: VpnRouteAssignment, assigned_domain_name: VpnDomainNameAssignment, mtu_size: int, max_frame_size: int, reserved: bool, main_outer_tunnel_transport: _winrt.winrt_base) -> None:
        ...
    def start_with_traffic_filter(assigned_client_ipv4_addresses: typing.Iterable[winrt.windows.networking.HostName], assigned_client_ipv6_addresses: typing.Iterable[winrt.windows.networking.HostName], vpninterface_id: VpnInterfaceId, assigned_routes: VpnRouteAssignment, assigned_namespace: VpnDomainNameAssignment, mtu_size: int, max_frame_size: int, reserved: bool, transports: typing.Iterable[_winrt.winrt_base], assigned_traffic_filters: VpnTrafficFilterAssignment) -> None:
        ...
    def start_with_traffic_filter(assigned_client_ipv4_list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName], assigned_client_ipv6_list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName], vpn_interface_id: VpnInterfaceId, assigned_routes: VpnRouteAssignment, assigned_namespace: VpnDomainNameAssignment, mtu_size: int, max_frame_size: int, reserved: bool, main_outer_tunnel_transport: _winrt.winrt_base, optional_outer_tunnel_transport: _winrt.winrt_base, assigned_traffic_filters: VpnTrafficFilterAssignment) -> None:
        ...
    def stop() -> None:
        ...
    def terminate_connection(message: str) -> None:
        ...
    def add_activity_change(handler: winrt.windows.foundation.TypedEventHandler[VpnChannel, VpnChannelActivityEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_activity_change(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_activity_state_change(handler: winrt.windows.foundation.TypedEventHandler[VpnChannel, VpnChannelActivityStateChangedArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_activity_state_change(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class VpnChannelActivityEventArgs(_winrt.winrt_base):
    ...
    type: VpnChannelActivityEventType

class VpnChannelActivityStateChangedArgs(_winrt.winrt_base):
    ...
    activity_state: VpnChannelActivityEventType

class VpnChannelConfiguration(_winrt.winrt_base):
    ...
    custom_field: str
    server_host_name_list: winrt.windows.foundation.collections.IVectorView[winrt.windows.networking.HostName]
    server_service_name: str
    server_uris: winrt.windows.foundation.collections.IVectorView[winrt.windows.foundation.Uri]

class VpnCredential(IVpnCredential, _winrt.winrt_base):
    ...
    additional_pin: str
    certificate_credential: winrt.windows.security.cryptography.certificates.Certificate
    old_password_credential: winrt.windows.security.credentials.PasswordCredential
    passkey_credential: winrt.windows.security.credentials.PasswordCredential

class VpnCustomCheckBox(IVpnCustomPrompt, _winrt.winrt_base):
    ...
    initial_check_state: bool
    checked: bool
    label: str
    compulsory: bool
    bordered: bool

class VpnCustomComboBox(IVpnCustomPrompt, _winrt.winrt_base):
    ...
    options_text: winrt.windows.foundation.collections.IVectorView[str]
    selected: int
    label: str
    compulsory: bool
    bordered: bool

class VpnCustomEditBox(IVpnCustomPrompt, _winrt.winrt_base):
    ...
    no_echo: bool
    default_text: str
    text: str
    label: str
    compulsory: bool
    bordered: bool

class VpnCustomErrorBox(IVpnCustomPrompt, _winrt.winrt_base):
    ...
    label: str
    compulsory: bool
    bordered: bool

class VpnCustomPromptBooleanInput(IVpnCustomPromptElement, _winrt.winrt_base):
    ...
    initial_value: bool
    value: bool
    emphasized: bool
    display_name: str
    compulsory: bool

class VpnCustomPromptOptionSelector(IVpnCustomPromptElement, _winrt.winrt_base):
    ...
    emphasized: bool
    display_name: str
    compulsory: bool
    options: winrt.windows.foundation.collections.IVector[str]
    selected_index: int

class VpnCustomPromptText(IVpnCustomPromptElement, _winrt.winrt_base):
    ...
    emphasized: bool
    display_name: str
    compulsory: bool
    text: str

class VpnCustomPromptTextInput(IVpnCustomPromptElement, _winrt.winrt_base):
    ...
    emphasized: bool
    display_name: str
    compulsory: bool
    placeholder_text: str
    is_text_hidden: bool
    text: str

class VpnCustomTextBox(IVpnCustomPrompt, _winrt.winrt_base):
    ...
    label: str
    compulsory: bool
    bordered: bool
    display_text: str

class VpnDomainNameAssignment(_winrt.winrt_base):
    ...
    proxy_auto_configuration_uri: winrt.windows.foundation.Uri
    domain_name_list: winrt.windows.foundation.collections.IVector[VpnDomainNameInfo]

class VpnDomainNameInfo(_winrt.winrt_base):
    ...
    domain_name_type: VpnDomainNameType
    domain_name: winrt.windows.networking.HostName
    dns_servers: winrt.windows.foundation.collections.IVector[winrt.windows.networking.HostName]
    web_proxy_servers: winrt.windows.foundation.collections.IVector[winrt.windows.networking.HostName]
    web_proxy_uris: winrt.windows.foundation.collections.IVector[winrt.windows.foundation.Uri]

class VpnForegroundActivatedEventArgs(winrt.windows.applicationmodel.activation.IActivatedEventArgs, winrt.windows.applicationmodel.activation.IActivatedEventArgsWithUser, _winrt.winrt_base):
    ...
    kind: winrt.windows.applicationmodel.activation.ActivationKind
    previous_execution_state: winrt.windows.applicationmodel.activation.ApplicationExecutionState
    splash_screen: winrt.windows.applicationmodel.activation.SplashScreen
    user: winrt.windows.system.User
    activation_operation: VpnForegroundActivationOperation
    profile_name: str
    shared_context: winrt.windows.foundation.collections.ValueSet

class VpnForegroundActivationOperation(_winrt.winrt_base):
    ...
    def complete(result: winrt.windows.foundation.collections.ValueSet) -> None:
        ...

class VpnInterfaceId(_winrt.winrt_base):
    ...
    def get_address_info() -> typing.List[int]:
        ...

class VpnManagementAgent(_winrt.winrt_base):
    ...
    def add_profile_from_object_async(profile: IVpnProfile) -> winrt.windows.foundation.IAsyncOperation[VpnManagementErrorStatus]:
        ...
    def add_profile_from_xml_async(xml: str) -> winrt.windows.foundation.IAsyncOperation[VpnManagementErrorStatus]:
        ...
    def connect_profile_async(profile: IVpnProfile) -> winrt.windows.foundation.IAsyncOperation[VpnManagementErrorStatus]:
        ...
    def connect_profile_with_password_credential_async(profile: IVpnProfile, password_credential: winrt.windows.security.credentials.PasswordCredential) -> winrt.windows.foundation.IAsyncOperation[VpnManagementErrorStatus]:
        ...
    def delete_profile_async(profile: IVpnProfile) -> winrt.windows.foundation.IAsyncOperation[VpnManagementErrorStatus]:
        ...
    def disconnect_profile_async(profile: IVpnProfile) -> winrt.windows.foundation.IAsyncOperation[VpnManagementErrorStatus]:
        ...
    def get_profiles_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[IVpnProfile]]:
        ...
    def update_profile_from_object_async(profile: IVpnProfile) -> winrt.windows.foundation.IAsyncOperation[VpnManagementErrorStatus]:
        ...
    def update_profile_from_xml_async(xml: str) -> winrt.windows.foundation.IAsyncOperation[VpnManagementErrorStatus]:
        ...

class VpnNamespaceAssignment(_winrt.winrt_base):
    ...
    proxy_auto_config_uri: winrt.windows.foundation.Uri
    namespace_list: winrt.windows.foundation.collections.IVector[VpnNamespaceInfo]

class VpnNamespaceInfo(_winrt.winrt_base):
    ...
    web_proxy_servers: winrt.windows.foundation.collections.IVector[winrt.windows.networking.HostName]
    namespace: str
    dns_servers: winrt.windows.foundation.collections.IVector[winrt.windows.networking.HostName]

class VpnNativeProfile(IVpnProfile, _winrt.winrt_base):
    ...
    user_authentication_method: VpnAuthenticationMethod
    tunnel_authentication_method: VpnAuthenticationMethod
    routing_policy_type: VpnRoutingPolicyType
    eap_configuration: str
    native_protocol_type: VpnNativeProtocolType
    servers: winrt.windows.foundation.collections.IVector[str]
    require_vpn_client_app_u_i: bool
    connection_status: VpnManagementConnectionStatus
    profile_name: str
    remember_credentials: bool
    always_on: bool
    routes: winrt.windows.foundation.collections.IVector[VpnRoute]
    app_triggers: winrt.windows.foundation.collections.IVector[VpnAppId]
    traffic_filters: winrt.windows.foundation.collections.IVector[VpnTrafficFilter]
    domain_name_info_list: winrt.windows.foundation.collections.IVector[VpnDomainNameInfo]

class VpnPacketBuffer(_winrt.winrt_base):
    ...
    transport_affinity: int
    status: VpnPacketBufferStatus
    buffer: winrt.windows.storage.streams.Buffer
    app_id: VpnAppId
    transport_context: _winrt.winrt_base

class VpnPacketBufferList(winrt.windows.foundation.collections.IIterable[VpnPacketBuffer], _winrt.winrt_base):
    ...
    status: VpnPacketBufferStatus
    size: int
    def add_at_begin(next_vpn_packet_buffer: VpnPacketBuffer) -> None:
        ...
    def append(next_vpn_packet_buffer: VpnPacketBuffer) -> None:
        ...
    def clear() -> None:
        ...
    def first() -> winrt.windows.foundation.collections.IIterator[VpnPacketBuffer]:
        ...
    def remove_at_begin() -> VpnPacketBuffer:
        ...
    def remove_at_end() -> VpnPacketBuffer:
        ...

class VpnPickedCredential(_winrt.winrt_base):
    ...
    additional_pin: str
    old_password_credential: winrt.windows.security.credentials.PasswordCredential
    passkey_credential: winrt.windows.security.credentials.PasswordCredential

class VpnPlugInProfile(IVpnProfile, _winrt.winrt_base):
    ...
    vpn_plugin_package_family_name: str
    custom_configuration: str
    server_uris: winrt.windows.foundation.collections.IVector[winrt.windows.foundation.Uri]
    require_vpn_client_app_u_i: bool
    connection_status: VpnManagementConnectionStatus
    profile_name: str
    remember_credentials: bool
    always_on: bool
    app_triggers: winrt.windows.foundation.collections.IVector[VpnAppId]
    domain_name_info_list: winrt.windows.foundation.collections.IVector[VpnDomainNameInfo]
    routes: winrt.windows.foundation.collections.IVector[VpnRoute]
    traffic_filters: winrt.windows.foundation.collections.IVector[VpnTrafficFilter]

class VpnRoute(_winrt.winrt_base):
    ...
    prefix_size: int
    address: winrt.windows.networking.HostName

class VpnRouteAssignment(_winrt.winrt_base):
    ...
    ipv6_inclusion_routes: winrt.windows.foundation.collections.IVector[VpnRoute]
    ipv6_exclusion_routes: winrt.windows.foundation.collections.IVector[VpnRoute]
    ipv4_inclusion_routes: winrt.windows.foundation.collections.IVector[VpnRoute]
    ipv4_exclusion_routes: winrt.windows.foundation.collections.IVector[VpnRoute]
    exclude_local_subnets: bool

class VpnSystemHealth(_winrt.winrt_base):
    ...
    statement_of_health: winrt.windows.storage.streams.Buffer

class VpnTrafficFilter(_winrt.winrt_base):
    ...
    routing_policy_type: VpnRoutingPolicyType
    protocol: VpnIPProtocol
    app_id: VpnAppId
    app_claims: winrt.windows.foundation.collections.IVector[str]
    local_address_ranges: winrt.windows.foundation.collections.IVector[str]
    local_port_ranges: winrt.windows.foundation.collections.IVector[str]
    remote_address_ranges: winrt.windows.foundation.collections.IVector[str]
    remote_port_ranges: winrt.windows.foundation.collections.IVector[str]

class VpnTrafficFilterAssignment(_winrt.winrt_base):
    ...
    allow_outbound: bool
    allow_inbound: bool
    traffic_filter_list: winrt.windows.foundation.collections.IVector[VpnTrafficFilter]

class IVpnChannelStatics(_winrt.winrt_base):
    ...
    def process_event_async(third_party_plug_in: _winrt.winrt_base, event: _winrt.winrt_base) -> None:
        ...

class IVpnCredential(_winrt.winrt_base):
    ...
    additional_pin: str
    certificate_credential: winrt.windows.security.cryptography.certificates.Certificate
    old_password_credential: winrt.windows.security.credentials.PasswordCredential
    passkey_credential: winrt.windows.security.credentials.PasswordCredential

class IVpnCustomPrompt(_winrt.winrt_base):
    ...
    bordered: bool
    compulsory: bool
    label: str

class IVpnCustomPromptElement(_winrt.winrt_base):
    ...
    compulsory: bool
    display_name: str
    emphasized: bool

class IVpnDomainNameInfoFactory(_winrt.winrt_base):
    ...
    def create_vpn_domain_name_info(name: str, name_type: VpnDomainNameType, dns_server_list: typing.Iterable[winrt.windows.networking.HostName], proxy_server_list: typing.Iterable[winrt.windows.networking.HostName]) -> VpnDomainNameInfo:
        ...

class IVpnInterfaceIdFactory(_winrt.winrt_base):
    ...
    def create_vpn_interface_id(address: typing.Sequence[int]) -> VpnInterfaceId:
        ...

class IVpnNamespaceInfoFactory(_winrt.winrt_base):
    ...
    def create_vpn_namespace_info(name: str, dns_server_list: winrt.windows.foundation.collections.IVector[winrt.windows.networking.HostName], proxy_server_list: winrt.windows.foundation.collections.IVector[winrt.windows.networking.HostName]) -> VpnNamespaceInfo:
        ...

class IVpnPacketBufferFactory(_winrt.winrt_base):
    ...
    def create_vpn_packet_buffer(parent_buffer: VpnPacketBuffer, offset: int, length: int) -> VpnPacketBuffer:
        ...

class IVpnPlugIn(_winrt.winrt_base):
    ...
    def connect(channel: VpnChannel) -> None:
        ...
    def decapsulate(channel: VpnChannel, encap_buffer: VpnPacketBuffer, decapsulated_packets: VpnPacketBufferList, control_packets_to_send: VpnPacketBufferList) -> None:
        ...
    def disconnect(channel: VpnChannel) -> None:
        ...
    def encapsulate(channel: VpnChannel, packets: VpnPacketBufferList, encapulated_packets: VpnPacketBufferList) -> None:
        ...
    def get_keep_alive_payload(channel: VpnChannel) -> VpnPacketBuffer:
        ...

class IVpnProfile(_winrt.winrt_base):
    ...
    always_on: bool
    app_triggers: winrt.windows.foundation.collections.IVector[VpnAppId]
    domain_name_info_list: winrt.windows.foundation.collections.IVector[VpnDomainNameInfo]
    profile_name: str
    remember_credentials: bool
    routes: winrt.windows.foundation.collections.IVector[VpnRoute]
    traffic_filters: winrt.windows.foundation.collections.IVector[VpnTrafficFilter]

class IVpnRouteFactory(_winrt.winrt_base):
    ...
    def create_vpn_route(address: winrt.windows.networking.HostName, prefix_size: int) -> VpnRoute:
        ...

