# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.devices.bluetooth.genericattributeprofile
except Exception:
    pass

try:
    import winrt.windows.devices.bluetooth.rfcomm
except Exception:
    pass

try:
    import winrt.windows.devices.enumeration
except Exception:
    pass

try:
    import winrt.windows.devices.radios
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
    import winrt.windows.storage.streams
except Exception:
    pass

class BluetoothAddressType(enum.IntEnum):
    PUBLIC = 0
    RANDOM = 1
    UNSPECIFIED = 2

class BluetoothCacheMode(enum.IntEnum):
    CACHED = 0
    UNCACHED = 1

class BluetoothConnectionStatus(enum.IntEnum):
    DISCONNECTED = 0
    CONNECTED = 1

class BluetoothError(enum.IntEnum):
    SUCCESS = 0
    RADIO_NOT_AVAILABLE = 1
    RESOURCE_IN_USE = 2
    DEVICE_NOT_CONNECTED = 3
    OTHER_ERROR = 4
    DISABLED_BY_POLICY = 5
    NOT_SUPPORTED = 6
    DISABLED_BY_USER = 7
    CONSENT_REQUIRED = 8
    TRANSPORT_NOT_SUPPORTED = 9

class BluetoothLEPreferredConnectionParametersRequestStatus(enum.IntEnum):
    UNSPECIFIED = 0
    SUCCESS = 1
    DEVICE_NOT_AVAILABLE = 2
    ACCESS_DENIED = 3

class BluetoothMajorClass(enum.IntEnum):
    MISCELLANEOUS = 0
    COMPUTER = 1
    PHONE = 2
    NETWORK_ACCESS_POINT = 3
    AUDIO_VIDEO = 4
    PERIPHERAL = 5
    IMAGING = 6
    WEARABLE = 7
    TOY = 8
    HEALTH = 9

class BluetoothMinorClass(enum.IntEnum):
    UNCATEGORIZED = 0
    COMPUTER_DESKTOP = 1
    COMPUTER_SERVER = 2
    COMPUTER_LAPTOP = 3
    COMPUTER_HANDHELD = 4
    COMPUTER_PALM_SIZE = 5
    COMPUTER_WEARABLE = 6
    COMPUTER_TABLET = 7
    PHONE_CELLULAR = 1
    PHONE_CORDLESS = 2
    PHONE_SMART_PHONE = 3
    PHONE_WIRED = 4
    PHONE_ISDN = 5
    NETWORK_FULLY_AVAILABLE = 0
    NETWORK_USED01_TO17_PERCENT = 8
    NETWORK_USED17_TO33_PERCENT = 16
    NETWORK_USED33_TO50_PERCENT = 24
    NETWORK_USED50_TO67_PERCENT = 32
    NETWORK_USED67_TO83_PERCENT = 40
    NETWORK_USED83_TO99_PERCENT = 48
    NETWORK_NO_SERVICE_AVAILABLE = 56
    AUDIO_VIDEO_WEARABLE_HEADSET = 1
    AUDIO_VIDEO_HANDS_FREE = 2
    AUDIO_VIDEO_MICROPHONE = 4
    AUDIO_VIDEO_LOUDSPEAKER = 5
    AUDIO_VIDEO_HEADPHONES = 6
    AUDIO_VIDEO_PORTABLE_AUDIO = 7
    AUDIO_VIDEO_CAR_AUDIO = 8
    AUDIO_VIDEO_SET_TOP_BOX = 9
    AUDIO_VIDEO_HIFI_AUDIO_DEVICE = 10
    AUDIO_VIDEO_VCR = 11
    AUDIO_VIDEO_VIDEO_CAMERA = 12
    AUDIO_VIDEO_CAMCORDER = 13
    AUDIO_VIDEO_VIDEO_MONITOR = 14
    AUDIO_VIDEO_VIDEO_DISPLAY_AND_LOUDSPEAKER = 15
    AUDIO_VIDEO_VIDEO_CONFERENCING = 16
    AUDIO_VIDEO_GAMING_OR_TOY = 18
    PERIPHERAL_JOYSTICK = 1
    PERIPHERAL_GAMEPAD = 2
    PERIPHERAL_REMOTE_CONTROL = 3
    PERIPHERAL_SENSING = 4
    PERIPHERAL_DIGITIZER_TABLET = 5
    PERIPHERAL_CARD_READER = 6
    PERIPHERAL_DIGITAL_PEN = 7
    PERIPHERAL_HANDHELD_SCANNER = 8
    PERIPHERAL_HANDHELD_GESTURE = 9
    WEARABLE_WRISTWATCH = 1
    WEARABLE_PAGER = 2
    WEARABLE_JACKET = 3
    WEARABLE_HELMET = 4
    WEARABLE_GLASSES = 5
    TOY_ROBOT = 1
    TOY_VEHICLE = 2
    TOY_DOLL = 3
    TOY_CONTROLLER = 4
    TOY_GAME = 5
    HEALTH_BLOOD_PRESSURE_MONITOR = 1
    HEALTH_THERMOMETER = 2
    HEALTH_WEIGHING_SCALE = 3
    HEALTH_GLUCOSE_METER = 4
    HEALTH_PULSE_OXIMETER = 5
    HEALTH_HEART_RATE_MONITOR = 6
    HEALTH_HEALTH_DATA_DISPLAY = 7
    HEALTH_STEP_COUNTER = 8
    HEALTH_BODY_COMPOSITION_ANALYZER = 9
    HEALTH_PEAK_FLOW_MONITOR = 10
    HEALTH_MEDICATION_MONITOR = 11
    HEALTH_KNEE_PROSTHESIS = 12
    HEALTH_ANKLE_PROSTHESIS = 13
    HEALTH_GENERIC_HEALTH_MANAGER = 14
    HEALTH_PERSONAL_MOBILITY_DEVICE = 15

class BluetoothServiceCapabilities(enum.IntFlag):
    NONE = 0
    LIMITED_DISCOVERABLE_MODE = 0x1
    POSITIONING_SERVICE = 0x8
    NETWORKING_SERVICE = 0x10
    RENDERING_SERVICE = 0x20
    CAPTURING_SERVICE = 0x40
    OBJECT_TRANSFER_SERVICE = 0x80
    AUDIO_SERVICE = 0x100
    TELEPHONE_SERVICE = 0x200
    INFORMATION_SERVICE = 0x400

class BluetoothAdapter(_winrt.winrt_base):
    ...
    bluetooth_address: int
    device_id: str
    is_advertisement_offload_supported: bool
    is_central_role_supported: bool
    is_classic_supported: bool
    is_low_energy_supported: bool
    is_peripheral_role_supported: bool
    are_classic_secure_connections_supported: bool
    are_low_energy_secure_connections_supported: bool
    is_extended_advertising_supported: bool
    max_advertisement_data_length: int
    def from_id_async(device_id: str) -> winrt.windows.foundation.IAsyncOperation[BluetoothAdapter]:
        ...
    def get_default_async() -> winrt.windows.foundation.IAsyncOperation[BluetoothAdapter]:
        ...
    def get_device_selector() -> str:
        ...
    def get_radio_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.radios.Radio]:
        ...

class BluetoothClassOfDevice(_winrt.winrt_base):
    ...
    major_class: BluetoothMajorClass
    minor_class: BluetoothMinorClass
    raw_value: int
    service_capabilities: BluetoothServiceCapabilities
    def from_parts(major_class: BluetoothMajorClass, minor_class: BluetoothMinorClass, service_capabilities: BluetoothServiceCapabilities) -> BluetoothClassOfDevice:
        ...
    def from_raw_value(raw_value: int) -> BluetoothClassOfDevice:
        ...

class BluetoothDevice(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    bluetooth_address: int
    class_of_device: BluetoothClassOfDevice
    connection_status: BluetoothConnectionStatus
    device_id: str
    host_name: winrt.windows.networking.HostName
    name: str
    rfcomm_services: winrt.windows.foundation.collections.IVectorView[winrt.windows.devices.bluetooth.rfcomm.RfcommDeviceService]
    sdp_records: winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.streams.IBuffer]
    device_information: winrt.windows.devices.enumeration.DeviceInformation
    device_access_information: winrt.windows.devices.enumeration.DeviceAccessInformation
    bluetooth_device_id: BluetoothDeviceId
    was_secure_connection_used_for_pairing: bool
    def close() -> None:
        ...
    def from_bluetooth_address_async(address: int) -> winrt.windows.foundation.IAsyncOperation[BluetoothDevice]:
        ...
    def from_host_name_async(host_name: winrt.windows.networking.HostName) -> winrt.windows.foundation.IAsyncOperation[BluetoothDevice]:
        ...
    def from_id_async(device_id: str) -> winrt.windows.foundation.IAsyncOperation[BluetoothDevice]:
        ...
    def get_device_selector() -> str:
        ...
    def get_device_selector_from_bluetooth_address(bluetooth_address: int) -> str:
        ...
    def get_device_selector_from_class_of_device(class_of_device: BluetoothClassOfDevice) -> str:
        ...
    def get_device_selector_from_connection_status(connection_status: BluetoothConnectionStatus) -> str:
        ...
    def get_device_selector_from_device_name(device_name: str) -> str:
        ...
    def get_device_selector_from_pairing_state(pairing_state: bool) -> str:
        ...
    def get_rfcomm_services_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.bluetooth.rfcomm.RfcommDeviceServicesResult]:
        ...
    def get_rfcomm_services_async(cache_mode: BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.bluetooth.rfcomm.RfcommDeviceServicesResult]:
        ...
    def get_rfcomm_services_for_id_async(service_id: winrt.windows.devices.bluetooth.rfcomm.RfcommServiceId) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.bluetooth.rfcomm.RfcommDeviceServicesResult]:
        ...
    def get_rfcomm_services_for_id_async(service_id: winrt.windows.devices.bluetooth.rfcomm.RfcommServiceId, cache_mode: BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.bluetooth.rfcomm.RfcommDeviceServicesResult]:
        ...
    def request_access_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.enumeration.DeviceAccessStatus]:
        ...
    def add_connection_status_changed(handler: winrt.windows.foundation.TypedEventHandler[BluetoothDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_connection_status_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_name_changed(handler: winrt.windows.foundation.TypedEventHandler[BluetoothDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_name_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_sdp_records_changed(handler: winrt.windows.foundation.TypedEventHandler[BluetoothDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_sdp_records_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class BluetoothDeviceId(_winrt.winrt_base):
    ...
    id: str
    is_classic_device: bool
    is_low_energy_device: bool
    def from_id(device_id: str) -> BluetoothDeviceId:
        ...

class BluetoothLEAppearance(_winrt.winrt_base):
    ...
    category: int
    raw_value: int
    sub_category: int
    def from_parts(appearance_category: int, appearance_sub_category: int) -> BluetoothLEAppearance:
        ...
    def from_raw_value(raw_value: int) -> BluetoothLEAppearance:
        ...

class BluetoothLEAppearanceCategories(_winrt.winrt_base):
    ...
    barcode_scanner: int
    blood_pressure: int
    clock: int
    computer: int
    cycling: int
    display: int
    eye_glasses: int
    glucose_meter: int
    heart_rate: int
    human_interface_device: int
    keyring: int
    media_player: int
    outdoor_sport_activity: int
    phone: int
    pulse_oximeter: int
    remote_control: int
    running_walking: int
    tag: int
    thermometer: int
    uncategorized: int
    watch: int
    weight_scale: int

class BluetoothLEAppearanceSubcategories(_winrt.winrt_base):
    ...
    barcode_scanner: int
    blood_pressure_arm: int
    blood_pressure_wrist: int
    card_reader: int
    cycling_cadence_sensor: int
    cycling_computer: int
    cycling_power_sensor: int
    cycling_speed_cadence_sensor: int
    cycling_speed_sensor: int
    digital_pen: int
    digitizer_tablet: int
    gamepad: int
    generic: int
    heart_rate_belt: int
    joystick: int
    keyboard: int
    location_display: int
    location_navigation_display: int
    location_navigation_pod: int
    location_pod: int
    mouse: int
    oximeter_fingertip: int
    oximeter_wrist_worn: int
    running_walking_in_shoe: int
    running_walking_on_hip: int
    running_walking_on_shoe: int
    sports_watch: int
    thermometer_ear: int

class BluetoothLEConnectionParameters(_winrt.winrt_base):
    ...
    connection_interval: int
    connection_latency: int
    link_timeout: int

class BluetoothLEConnectionPhy(_winrt.winrt_base):
    ...
    receive_info: BluetoothLEConnectionPhyInfo
    transmit_info: BluetoothLEConnectionPhyInfo

class BluetoothLEConnectionPhyInfo(_winrt.winrt_base):
    ...
    is_coded_phy: bool
    is_uncoded1_m_phy: bool
    is_uncoded2_m_phy: bool

class BluetoothLEDevice(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    bluetooth_address: int
    connection_status: BluetoothConnectionStatus
    device_id: str
    gatt_services: winrt.windows.foundation.collections.IVectorView[winrt.windows.devices.bluetooth.genericattributeprofile.GattDeviceService]
    name: str
    appearance: BluetoothLEAppearance
    bluetooth_address_type: BluetoothAddressType
    device_information: winrt.windows.devices.enumeration.DeviceInformation
    device_access_information: winrt.windows.devices.enumeration.DeviceAccessInformation
    bluetooth_device_id: BluetoothDeviceId
    was_secure_connection_used_for_pairing: bool
    def close() -> None:
        ...
    def from_bluetooth_address_async(bluetooth_address: int) -> winrt.windows.foundation.IAsyncOperation[BluetoothLEDevice]:
        ...
    def from_bluetooth_address_async(bluetooth_address: int, bluetooth_address_type: BluetoothAddressType) -> winrt.windows.foundation.IAsyncOperation[BluetoothLEDevice]:
        ...
    def from_id_async(device_id: str) -> winrt.windows.foundation.IAsyncOperation[BluetoothLEDevice]:
        ...
    def get_connection_parameters() -> BluetoothLEConnectionParameters:
        ...
    def get_connection_phy() -> BluetoothLEConnectionPhy:
        ...
    def get_device_selector() -> str:
        ...
    def get_device_selector_from_appearance(appearance: BluetoothLEAppearance) -> str:
        ...
    def get_device_selector_from_bluetooth_address(bluetooth_address: int) -> str:
        ...
    def get_device_selector_from_bluetooth_address(bluetooth_address: int, bluetooth_address_type: BluetoothAddressType) -> str:
        ...
    def get_device_selector_from_connection_status(connection_status: BluetoothConnectionStatus) -> str:
        ...
    def get_device_selector_from_device_name(device_name: str) -> str:
        ...
    def get_device_selector_from_pairing_state(pairing_state: bool) -> str:
        ...
    def get_gatt_service(service_uuid: uuid.UUID) -> winrt.windows.devices.bluetooth.genericattributeprofile.GattDeviceService:
        ...
    def get_gatt_services_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.bluetooth.genericattributeprofile.GattDeviceServicesResult]:
        ...
    def get_gatt_services_async(cache_mode: BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.bluetooth.genericattributeprofile.GattDeviceServicesResult]:
        ...
    def get_gatt_services_for_uuid_async(service_uuid: uuid.UUID) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.bluetooth.genericattributeprofile.GattDeviceServicesResult]:
        ...
    def get_gatt_services_for_uuid_async(service_uuid: uuid.UUID, cache_mode: BluetoothCacheMode) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.bluetooth.genericattributeprofile.GattDeviceServicesResult]:
        ...
    def request_access_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.devices.enumeration.DeviceAccessStatus]:
        ...
    def request_preferred_connection_parameters(preferred_connection_parameters: BluetoothLEPreferredConnectionParameters) -> BluetoothLEPreferredConnectionParametersRequest:
        ...
    def add_connection_status_changed(handler: winrt.windows.foundation.TypedEventHandler[BluetoothLEDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_connection_status_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_gatt_services_changed(handler: winrt.windows.foundation.TypedEventHandler[BluetoothLEDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_gatt_services_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_name_changed(handler: winrt.windows.foundation.TypedEventHandler[BluetoothLEDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_name_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_connection_parameters_changed(handler: winrt.windows.foundation.TypedEventHandler[BluetoothLEDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_connection_parameters_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_connection_phy_changed(handler: winrt.windows.foundation.TypedEventHandler[BluetoothLEDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_connection_phy_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class BluetoothLEPreferredConnectionParameters(_winrt.winrt_base):
    ...
    connection_latency: int
    link_timeout: int
    max_connection_interval: int
    min_connection_interval: int
    balanced: BluetoothLEPreferredConnectionParameters
    power_optimized: BluetoothLEPreferredConnectionParameters
    throughput_optimized: BluetoothLEPreferredConnectionParameters

class BluetoothLEPreferredConnectionParametersRequest(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    status: BluetoothLEPreferredConnectionParametersRequestStatus
    def close() -> None:
        ...

class BluetoothSignalStrengthFilter(_winrt.winrt_base):
    ...
    sampling_interval: typing.Optional[winrt.windows.foundation.TimeSpan]
    out_of_range_timeout: typing.Optional[winrt.windows.foundation.TimeSpan]
    out_of_range_threshold_in_d_bm: typing.Optional[int]
    in_range_threshold_in_d_bm: typing.Optional[int]

class BluetoothUuidHelper(_winrt.winrt_base):
    ...
    def from_short_id(short_id: int) -> uuid.UUID:
        ...
    def try_get_short_id(uuid: uuid.UUID) -> typing.Optional[int]:
        ...

