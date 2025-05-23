# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.devices.bluetooth
except Exception:
    pass

try:
    import winrt.windows.devices.enumeration
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
    import winrt.windows.storage.streams
except Exception:
    pass

class GattCharacteristicProperties(enum.IntFlag):
    NONE = 0
    BROADCAST = 0x1
    READ = 0x2
    WRITE_WITHOUT_RESPONSE = 0x4
    WRITE = 0x8
    NOTIFY = 0x10
    INDICATE = 0x20
    AUTHENTICATED_SIGNED_WRITES = 0x40
    EXTENDED_PROPERTIES = 0x80
    RELIABLE_WRITES = 0x100
    WRITABLE_AUXILIARIES = 0x200

class GattClientCharacteristicConfigurationDescriptorValue(enum.IntEnum):
    NONE = 0
    NOTIFY = 1
    INDICATE = 2

class GattCommunicationStatus(enum.IntEnum):
    SUCCESS = 0
    UNREACHABLE = 1
    PROTOCOL_ERROR = 2
    ACCESS_DENIED = 3

class GattOpenStatus(enum.IntEnum):
    UNSPECIFIED = 0
    SUCCESS = 1
    ALREADY_OPENED = 2
    NOT_FOUND = 3
    SHARING_VIOLATION = 4
    ACCESS_DENIED = 5

class GattProtectionLevel(enum.IntEnum):
    PLAIN = 0
    AUTHENTICATION_REQUIRED = 1
    ENCRYPTION_REQUIRED = 2
    ENCRYPTION_AND_AUTHENTICATION_REQUIRED = 3

class GattRequestState(enum.IntEnum):
    PENDING = 0
    COMPLETED = 1
    CANCELED = 2

class GattServiceProviderAdvertisementStatus(enum.IntEnum):
    CREATED = 0
    STOPPED = 1
    STARTED = 2
    ABORTED = 3
    STARTED_WITHOUT_ALL_ADVERTISEMENT_DATA = 4

class GattSessionStatus(enum.IntEnum):
    CLOSED = 0
    ACTIVE = 1

class GattSharingMode(enum.IntEnum):
    UNSPECIFIED = 0
    EXCLUSIVE = 1
    SHARED_READ_ONLY = 2
    SHARED_READ_AND_WRITE = 3

class GattWriteOption(enum.IntEnum):
    WRITE_WITH_RESPONSE = 0
    WRITE_WITHOUT_RESPONSE = 1

class GattCharacteristic(_winrt.winrt_base):
    ...
    protection_level: GattProtectionLevel
    attribute_handle: int
    characteristic_properties: GattCharacteristicProperties
    presentation_formats: winrt.windows.foundation.collections.IVectorView[GattPresentationFormat]
    user_description: str
    uuid: uuid.UUID
    service: GattDeviceService
    def convert_short_id_to_uuid(short_id: int) -> uuid.UUID:
        ...
    def get_all_descriptors() -> winrt.windows.foundation.collections.IVectorView[GattDescriptor]:
        ...
    def get_descriptors(descriptor_uuid: uuid.UUID) -> winrt.windows.foundation.collections.IVectorView[GattDescriptor]:
        ...
    def get_descriptors_async() -> winrt.windows.foundation.IAsyncOperation[GattDescriptorsResult]:
        ...
    def get_descriptors_async(cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[GattDescriptorsResult]:
        ...
    def get_descriptors_for_uuid_async(descriptor_uuid: uuid.UUID) -> winrt.windows.foundation.IAsyncOperation[GattDescriptorsResult]:
        ...
    def get_descriptors_for_uuid_async(descriptor_uuid: uuid.UUID, cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[GattDescriptorsResult]:
        ...
    def read_client_characteristic_configuration_descriptor_async() -> winrt.windows.foundation.IAsyncOperation[GattReadClientCharacteristicConfigurationDescriptorResult]:
        ...
    def read_value_async() -> winrt.windows.foundation.IAsyncOperation[GattReadResult]:
        ...
    def read_value_async(cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[GattReadResult]:
        ...
    def write_client_characteristic_configuration_descriptor_async(client_characteristic_configuration_descriptor_value: GattClientCharacteristicConfigurationDescriptorValue) -> winrt.windows.foundation.IAsyncOperation[GattCommunicationStatus]:
        ...
    def write_client_characteristic_configuration_descriptor_with_result_async(client_characteristic_configuration_descriptor_value: GattClientCharacteristicConfigurationDescriptorValue) -> winrt.windows.foundation.IAsyncOperation[GattWriteResult]:
        ...
    def write_value_async(value: winrt.windows.storage.streams.IBuffer) -> winrt.windows.foundation.IAsyncOperation[GattCommunicationStatus]:
        ...
    def write_value_async(value: winrt.windows.storage.streams.IBuffer, write_option: GattWriteOption) -> winrt.windows.foundation.IAsyncOperation[GattCommunicationStatus]:
        ...
    def write_value_with_result_async(value: winrt.windows.storage.streams.IBuffer) -> winrt.windows.foundation.IAsyncOperation[GattWriteResult]:
        ...
    def write_value_with_result_async(value: winrt.windows.storage.streams.IBuffer, write_option: GattWriteOption) -> winrt.windows.foundation.IAsyncOperation[GattWriteResult]:
        ...
    def add_value_changed(value_changed_handler: winrt.windows.foundation.TypedEventHandler[GattCharacteristic, GattValueChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_value_changed(value_changed_event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GattCharacteristicUuids(_winrt.winrt_base):
    ...
    heart_rate_measurement: uuid.UUID
    battery_level: uuid.UUID
    blood_pressure_feature: uuid.UUID
    blood_pressure_measurement: uuid.UUID
    body_sensor_location: uuid.UUID
    csc_feature: uuid.UUID
    csc_measurement: uuid.UUID
    glucose_feature: uuid.UUID
    glucose_measurement: uuid.UUID
    glucose_measurement_context: uuid.UUID
    heart_rate_control_point: uuid.UUID
    intermediate_cuff_pressure: uuid.UUID
    intermediate_temperature: uuid.UUID
    measurement_interval: uuid.UUID
    record_access_control_point: uuid.UUID
    rsc_feature: uuid.UUID
    rsc_measurement: uuid.UUID
    s_c_control_point: uuid.UUID
    sensor_location: uuid.UUID
    temperature_measurement: uuid.UUID
    temperature_type: uuid.UUID
    gap_peripheral_preferred_connection_parameters: uuid.UUID
    gap_peripheral_privacy_flag: uuid.UUID
    gap_reconnection_address: uuid.UUID
    gatt_service_changed: uuid.UUID
    hardware_revision_string: uuid.UUID
    hid_control_point: uuid.UUID
    hid_information: uuid.UUID
    ieee1107320601_regulatory_certification_data_list: uuid.UUID
    ln_control_point: uuid.UUID
    ln_feature: uuid.UUID
    local_time_information: uuid.UUID
    location_and_speed: uuid.UUID
    manufacturer_name_string: uuid.UUID
    model_number_string: uuid.UUID
    navigation: uuid.UUID
    new_alert: uuid.UUID
    pnp_id: uuid.UUID
    position_quality: uuid.UUID
    protocol_mode: uuid.UUID
    cycling_power_feature: uuid.UUID
    report: uuid.UUID
    report_map: uuid.UUID
    ringer_control_point: uuid.UUID
    ringer_setting: uuid.UUID
    scan_interval_window: uuid.UUID
    scan_refresh: uuid.UUID
    serial_number_string: uuid.UUID
    software_revision_string: uuid.UUID
    support_unread_alert_category: uuid.UUID
    supported_new_alert_category: uuid.UUID
    system_id: uuid.UUID
    time_accuracy: uuid.UUID
    time_source: uuid.UUID
    time_update_control_point: uuid.UUID
    time_update_state: uuid.UUID
    time_with_dst: uuid.UUID
    time_zone: uuid.UUID
    tx_power_level: uuid.UUID
    unread_alert_status: uuid.UUID
    alert_category_id: uuid.UUID
    alert_category_id_bit_mask: uuid.UUID
    alert_level: uuid.UUID
    alert_notification_control_point: uuid.UUID
    alert_status: uuid.UUID
    boot_keyboard_input_report: uuid.UUID
    boot_keyboard_output_report: uuid.UUID
    boot_mouse_input_report: uuid.UUID
    current_time: uuid.UUID
    cycling_power_control_point: uuid.UUID
    reference_time_information: uuid.UUID
    cycling_power_measurement: uuid.UUID
    cycling_power_vector: uuid.UUID
    date_time: uuid.UUID
    day_date_time: uuid.UUID
    day_of_week: uuid.UUID
    dst_offset: uuid.UUID
    exact_time256: uuid.UUID
    firmware_revision_string: uuid.UUID
    gap_appearance: uuid.UUID
    gap_device_name: uuid.UUID

class GattCharacteristicsResult(_winrt.winrt_base):
    ...
    characteristics: winrt.windows.foundation.collections.IVectorView[GattCharacteristic]
    protocol_error: typing.Optional[int]
    status: GattCommunicationStatus

class GattClientNotificationResult(_winrt.winrt_base):
    ...
    protocol_error: typing.Optional[int]
    status: GattCommunicationStatus
    subscribed_client: GattSubscribedClient
    bytes_sent: int

class GattDescriptor(_winrt.winrt_base):
    ...
    protection_level: GattProtectionLevel
    attribute_handle: int
    uuid: uuid.UUID
    def convert_short_id_to_uuid(short_id: int) -> uuid.UUID:
        ...
    def read_value_async() -> winrt.windows.foundation.IAsyncOperation[GattReadResult]:
        ...
    def read_value_async(cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[GattReadResult]:
        ...
    def write_value_async(value: winrt.windows.storage.streams.IBuffer) -> winrt.windows.foundation.IAsyncOperation[GattCommunicationStatus]:
        ...
    def write_value_with_result_async(value: winrt.windows.storage.streams.IBuffer) -> winrt.windows.foundation.IAsyncOperation[GattWriteResult]:
        ...

class GattDescriptorUuids(_winrt.winrt_base):
    ...
    characteristic_aggregate_format: uuid.UUID
    characteristic_extended_properties: uuid.UUID
    characteristic_presentation_format: uuid.UUID
    characteristic_user_description: uuid.UUID
    client_characteristic_configuration: uuid.UUID
    server_characteristic_configuration: uuid.UUID

class GattDescriptorsResult(_winrt.winrt_base):
    ...
    descriptors: winrt.windows.foundation.collections.IVectorView[GattDescriptor]
    protocol_error: typing.Optional[int]
    status: GattCommunicationStatus

class GattDeviceService(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    attribute_handle: int
    device_id: str
    uuid: uuid.UUID
    device: winrt.windows.devices.bluetooth.BluetoothLEDevice
    parent_services: winrt.windows.foundation.collections.IVectorView[GattDeviceService]
    device_access_information: winrt.windows.devices.enumeration.DeviceAccessInformation
    session: GattSession
    sharing_mode: GattSharingMode
    def close() -> None:
        ...
    def convert_short_id_to_uuid(short_id: int) -> uuid.UUID:
        ...
    def from_id_async(device_id: str) -> winrt.windows.foundation.IAsyncOperation[GattDeviceService]:
        ...
    def from_id_async(device_id: str, sharing_mode: GattSharingMode) -> winrt.windows.foundation.IAsyncOperation[GattDeviceService]:
        ...
    def get_all_characteristics() -> winrt.windows.foundation.collections.IVectorView[GattCharacteristic]:
        ...
    def get_all_included_services() -> winrt.windows.foundation.collections.IVectorView[GattDeviceService]:
        ...
    def get_characteristics(characteristic_uuid: uuid.UUID) -> winrt.windows.foundation.collections.IVectorView[GattCharacteristic]:
        ...
    def get_characteristics_async() -> winrt.windows.foundation.IAsyncOperation[GattCharacteristicsResult]:
        ...
    def get_characteristics_async(cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[GattCharacteristicsResult]:
        ...
    def get_characteristics_for_uuid_async(characteristic_uuid: uuid.UUID) -> winrt.windows.foundation.IAsyncOperation[GattCharacteristicsResult]:
        ...
    def get_characteristics_for_uuid_async(characteristic_uuid: uuid.UUID, cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[GattCharacteristicsResult]:
        ...
    def get_device_selector_for_bluetooth_device_id(bluetooth_device_id: winrt.windows.devices.bluetooth.BluetoothDeviceId) -> str:
        ...
    def get_device_selector_for_bluetooth_device_id(bluetooth_device_id: winrt.windows.devices.bluetooth.BluetoothDeviceId, cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> str:
        ...
    def get_device_selector_for_bluetooth_device_id_and_uuid(bluetooth_device_id: winrt.windows.devices.bluetooth.BluetoothDeviceId, service_uuid: uuid.UUID) -> str:
        ...
    def get_device_selector_for_bluetooth_device_id_and_uuid(bluetooth_device_id: winrt.windows.devices.bluetooth.BluetoothDeviceId, service_uuid: uuid.UUID, cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> str:
        ...
    def get_device_selector_from_short_id(service_short_id: int) -> str:
        ...
    def get_device_selector_from_uuid(service_uuid: uuid.UUID) -> str:
        ...
    def get_included_services(service_uuid: uuid.UUID) -> winrt.windows.foundation.collections.IVectorView[GattDeviceService]:
        ...
    def get_included_services_async() -> winrt.windows.foundation.IAsyncOperation[GattDeviceServicesResult]:
        ...
    def get_included_services_async(cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[GattDeviceServicesResult]:
        ...
    def get_included_services_for_uuid_async(service_uuid: uuid.UUID) -> winrt.windows.foundation.IAsyncOperation[GattDeviceServicesResult]:
        ...
    def get_included_services_for_uuid_async(service_uuid: uuid.UUID, cache_mode: winrt.windows.devices.bluetooth.BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[GattDeviceServicesResult]:
        ...
    def open_async(sharing_mode: GattSharingMode) -> winrt.windows.foundation.IAsyncOperation[GattOpenStatus]:
        ...
    def request_access_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.enumeration.DeviceAccessStatus]:
        ...

class GattDeviceServicesResult(_winrt.winrt_base):
    ...
    protocol_error: typing.Optional[int]
    services: winrt.windows.foundation.collections.IVectorView[GattDeviceService]
    status: GattCommunicationStatus

class GattLocalCharacteristic(_winrt.winrt_base):
    ...
    characteristic_properties: GattCharacteristicProperties
    descriptors: winrt.windows.foundation.collections.IVectorView[GattLocalDescriptor]
    presentation_formats: winrt.windows.foundation.collections.IVectorView[GattPresentationFormat]
    read_protection_level: GattProtectionLevel
    static_value: winrt.windows.storage.streams.IBuffer
    subscribed_clients: winrt.windows.foundation.collections.IVectorView[GattSubscribedClient]
    user_description: str
    uuid: uuid.UUID
    write_protection_level: GattProtectionLevel
    def create_descriptor_async(descriptor_uuid: uuid.UUID, parameters: GattLocalDescriptorParameters) -> winrt.windows.foundation.IAsyncOperation[GattLocalDescriptorResult]:
        ...
    def notify_value_async(value: winrt.windows.storage.streams.IBuffer) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[GattClientNotificationResult]]:
        ...
    def notify_value_async(value: winrt.windows.storage.streams.IBuffer, subscribed_client: GattSubscribedClient) -> winrt.windows.foundation.IAsyncOperation[GattClientNotificationResult]:
        ...
    def add_read_requested(handler: winrt.windows.foundation.TypedEventHandler[GattLocalCharacteristic, GattReadRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_read_requested(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_subscribed_clients_changed(handler: winrt.windows.foundation.TypedEventHandler[GattLocalCharacteristic, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_subscribed_clients_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_write_requested(handler: winrt.windows.foundation.TypedEventHandler[GattLocalCharacteristic, GattWriteRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_write_requested(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GattLocalCharacteristicParameters(_winrt.winrt_base):
    ...
    write_protection_level: GattProtectionLevel
    user_description: str
    static_value: winrt.windows.storage.streams.IBuffer
    read_protection_level: GattProtectionLevel
    characteristic_properties: GattCharacteristicProperties
    presentation_formats: winrt.windows.foundation.collections.IVector[GattPresentationFormat]

class GattLocalCharacteristicResult(_winrt.winrt_base):
    ...
    characteristic: GattLocalCharacteristic
    error: winrt.windows.devices.bluetooth.BluetoothError

class GattLocalDescriptor(_winrt.winrt_base):
    ...
    read_protection_level: GattProtectionLevel
    static_value: winrt.windows.storage.streams.IBuffer
    uuid: uuid.UUID
    write_protection_level: GattProtectionLevel
    def add_read_requested(handler: winrt.windows.foundation.TypedEventHandler[GattLocalDescriptor, GattReadRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_read_requested(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_write_requested(handler: winrt.windows.foundation.TypedEventHandler[GattLocalDescriptor, GattWriteRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_write_requested(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GattLocalDescriptorParameters(_winrt.winrt_base):
    ...
    write_protection_level: GattProtectionLevel
    static_value: winrt.windows.storage.streams.IBuffer
    read_protection_level: GattProtectionLevel

class GattLocalDescriptorResult(_winrt.winrt_base):
    ...
    descriptor: GattLocalDescriptor
    error: winrt.windows.devices.bluetooth.BluetoothError

class GattLocalService(_winrt.winrt_base):
    ...
    characteristics: winrt.windows.foundation.collections.IVectorView[GattLocalCharacteristic]
    uuid: uuid.UUID
    def create_characteristic_async(characteristic_uuid: uuid.UUID, parameters: GattLocalCharacteristicParameters) -> winrt.windows.foundation.IAsyncOperation[GattLocalCharacteristicResult]:
        ...

class GattPresentationFormat(_winrt.winrt_base):
    ...
    description: int
    exponent: int
    format_type: int
    namespace: int
    unit: int
    bluetooth_sig_assigned_numbers: int
    def from_parts(format_type: int, exponent: int, unit: int, namespace_id: int, description: int) -> GattPresentationFormat:
        ...

class GattPresentationFormatTypes(_winrt.winrt_base):
    ...
    bit2: int
    boolean: int
    d_uint16: int
    float: int
    float32: int
    float64: int
    nibble: int
    s_float: int
    s_int12: int
    s_int128: int
    s_int16: int
    s_int24: int
    s_int32: int
    s_int48: int
    s_int64: int
    s_int8: int
    struct: int
    uint12: int
    uint128: int
    uint16: int
    uint24: int
    uint32: int
    uint48: int
    uint64: int
    uint8: int
    utf16: int
    utf8: int

class GattProtocolError(_winrt.winrt_base):
    ...
    attribute_not_found: int
    attribute_not_long: int
    insufficient_authentication: int
    insufficient_authorization: int
    insufficient_encryption: int
    insufficient_encryption_key_size: int
    insufficient_resources: int
    invalid_attribute_value_length: int
    invalid_handle: int
    invalid_offset: int
    invalid_pdu: int
    prepare_queue_full: int
    read_not_permitted: int
    request_not_supported: int
    unlikely_error: int
    unsupported_group_type: int
    write_not_permitted: int

class GattReadClientCharacteristicConfigurationDescriptorResult(_winrt.winrt_base):
    ...
    client_characteristic_configuration_descriptor: GattClientCharacteristicConfigurationDescriptorValue
    status: GattCommunicationStatus
    protocol_error: typing.Optional[int]

class GattReadRequest(_winrt.winrt_base):
    ...
    length: int
    offset: int
    state: GattRequestState
    def respond_with_protocol_error(protocol_error: int) -> None:
        ...
    def respond_with_value(value: winrt.windows.storage.streams.IBuffer) -> None:
        ...
    def add_state_changed(handler: winrt.windows.foundation.TypedEventHandler[GattReadRequest, GattRequestStateChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_state_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GattReadRequestedEventArgs(_winrt.winrt_base):
    ...
    session: GattSession
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...
    def get_request_async() -> winrt.windows.foundation.IAsyncOperation[GattReadRequest]:
        ...

class GattReadResult(_winrt.winrt_base):
    ...
    status: GattCommunicationStatus
    value: winrt.windows.storage.streams.IBuffer
    protocol_error: typing.Optional[int]

class GattReliableWriteTransaction(_winrt.winrt_base):
    ...
    def commit_async() -> winrt.windows.foundation.IAsyncOperation[GattCommunicationStatus]:
        ...
    def commit_with_result_async() -> winrt.windows.foundation.IAsyncOperation[GattWriteResult]:
        ...
    def write_value(characteristic: GattCharacteristic, value: winrt.windows.storage.streams.IBuffer) -> None:
        ...

class GattRequestStateChangedEventArgs(_winrt.winrt_base):
    ...
    error: winrt.windows.devices.bluetooth.BluetoothError
    state: GattRequestState

class GattServiceProvider(_winrt.winrt_base):
    ...
    advertisement_status: GattServiceProviderAdvertisementStatus
    service: GattLocalService
    def create_async(service_uuid: uuid.UUID) -> winrt.windows.foundation.IAsyncOperation[GattServiceProviderResult]:
        ...
    def start_advertising() -> None:
        ...
    def start_advertising(parameters: GattServiceProviderAdvertisingParameters) -> None:
        ...
    def stop_advertising() -> None:
        ...
    def add_advertisement_status_changed(handler: winrt.windows.foundation.TypedEventHandler[GattServiceProvider, GattServiceProviderAdvertisementStatusChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_advertisement_status_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GattServiceProviderAdvertisementStatusChangedEventArgs(_winrt.winrt_base):
    ...
    error: winrt.windows.devices.bluetooth.BluetoothError
    status: GattServiceProviderAdvertisementStatus

class GattServiceProviderAdvertisingParameters(_winrt.winrt_base):
    ...
    is_discoverable: bool
    is_connectable: bool
    service_data: winrt.windows.storage.streams.IBuffer

class GattServiceProviderResult(_winrt.winrt_base):
    ...
    error: winrt.windows.devices.bluetooth.BluetoothError
    service_provider: GattServiceProvider

class GattServiceUuids(_winrt.winrt_base):
    ...
    battery: uuid.UUID
    blood_pressure: uuid.UUID
    cycling_speed_and_cadence: uuid.UUID
    generic_access: uuid.UUID
    generic_attribute: uuid.UUID
    glucose: uuid.UUID
    health_thermometer: uuid.UUID
    heart_rate: uuid.UUID
    running_speed_and_cadence: uuid.UUID
    alert_notification: uuid.UUID
    current_time: uuid.UUID
    cycling_power: uuid.UUID
    device_information: uuid.UUID
    human_interface_device: uuid.UUID
    immediate_alert: uuid.UUID
    link_loss: uuid.UUID
    location_and_navigation: uuid.UUID
    next_dst_change: uuid.UUID
    phone_alert_status: uuid.UUID
    reference_time_update: uuid.UUID
    scan_parameters: uuid.UUID
    tx_power: uuid.UUID

class GattSession(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    maintain_connection: bool
    can_maintain_connection: bool
    device_id: winrt.windows.devices.bluetooth.BluetoothDeviceId
    max_pdu_size: int
    session_status: GattSessionStatus
    def close() -> None:
        ...
    def from_device_id_async(device_id: winrt.windows.devices.bluetooth.BluetoothDeviceId) -> winrt.windows.foundation.IAsyncOperation[GattSession]:
        ...
    def add_max_pdu_size_changed(handler: winrt.windows.foundation.TypedEventHandler[GattSession, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_max_pdu_size_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_session_status_changed(handler: winrt.windows.foundation.TypedEventHandler[GattSession, GattSessionStatusChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_session_status_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GattSessionStatusChangedEventArgs(_winrt.winrt_base):
    ...
    error: winrt.windows.devices.bluetooth.BluetoothError
    status: GattSessionStatus

class GattSubscribedClient(_winrt.winrt_base):
    ...
    max_notification_size: int
    session: GattSession
    def add_max_notification_size_changed(handler: winrt.windows.foundation.TypedEventHandler[GattSubscribedClient, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_max_notification_size_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GattValueChangedEventArgs(_winrt.winrt_base):
    ...
    characteristic_value: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.DateTime

class GattWriteRequest(_winrt.winrt_base):
    ...
    offset: int
    option: GattWriteOption
    state: GattRequestState
    value: winrt.windows.storage.streams.IBuffer
    def respond() -> None:
        ...
    def respond_with_protocol_error(protocol_error: int) -> None:
        ...
    def add_state_changed(handler: winrt.windows.foundation.TypedEventHandler[GattWriteRequest, GattRequestStateChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_state_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GattWriteRequestedEventArgs(_winrt.winrt_base):
    ...
    session: GattSession
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...
    def get_request_async() -> winrt.windows.foundation.IAsyncOperation[GattWriteRequest]:
        ...

class GattWriteResult(_winrt.winrt_base):
    ...
    protocol_error: typing.Optional[int]
    status: GattCommunicationStatus

