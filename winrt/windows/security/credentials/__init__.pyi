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
    import winrt.windows.security.cryptography.core
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

class KeyCredentialAttestationStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_ERROR = 1
    NOT_SUPPORTED = 2
    TEMPORARY_FAILURE = 3

class KeyCredentialCreationOption(enum.IntEnum):
    REPLACE_EXISTING = 0
    FAIL_IF_EXISTS = 1

class KeyCredentialStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_ERROR = 1
    NOT_FOUND = 2
    USER_CANCELED = 3
    USER_PREFERS_PASSWORD = 4
    CREDENTIAL_ALREADY_EXISTS = 5
    SECURITY_DEVICE_LOCKED = 6

class WebAccountPictureSize(enum.IntEnum):
    SIZE64X64 = 64
    SIZE208X208 = 208
    SIZE424X424 = 424
    SIZE1080X1080 = 1080

class WebAccountState(enum.IntEnum):
    NONE = 0
    CONNECTED = 1
    ERROR = 2

class KeyCredential(_winrt.winrt_base):
    ...
    name: str
    def get_attestation_async() -> winrt.windows.foundation.IAsyncOperation[KeyCredentialAttestationResult]:
        ...
    def request_sign_async(data: winrt.windows.storage.streams.IBuffer) -> winrt.windows.foundation.IAsyncOperation[KeyCredentialOperationResult]:
        ...
    def retrieve_public_key() -> winrt.windows.storage.streams.IBuffer:
        ...
    def retrieve_public_key(blob_type: winrt.windows.security.cryptography.core.CryptographicPublicKeyBlobType) -> winrt.windows.storage.streams.IBuffer:
        ...

class KeyCredentialAttestationResult(_winrt.winrt_base):
    ...
    attestation_buffer: winrt.windows.storage.streams.IBuffer
    certificate_chain_buffer: winrt.windows.storage.streams.IBuffer
    status: KeyCredentialAttestationStatus

class KeyCredentialManager(_winrt.winrt_base):
    ...
    def delete_async(name: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def is_supported_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def open_async(name: str) -> winrt.windows.foundation.IAsyncOperation[KeyCredentialRetrievalResult]:
        ...
    def renew_attestation_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def request_create_async(name: str, option: KeyCredentialCreationOption) -> winrt.windows.foundation.IAsyncOperation[KeyCredentialRetrievalResult]:
        ...

class KeyCredentialOperationResult(_winrt.winrt_base):
    ...
    result: winrt.windows.storage.streams.IBuffer
    status: KeyCredentialStatus

class KeyCredentialRetrievalResult(_winrt.winrt_base):
    ...
    credential: KeyCredential
    status: KeyCredentialStatus

class PasswordCredential(_winrt.winrt_base):
    ...
    user_name: str
    resource: str
    password: str
    properties: winrt.windows.foundation.collections.IPropertySet
    def retrieve_password() -> None:
        ...

class PasswordCredentialPropertyStore(winrt.windows.foundation.collections.IPropertySet, winrt.windows.foundation.collections.IObservableMap[str, _winrt.winrt_base], winrt.windows.foundation.collections.IMap[str, _winrt.winrt_base], winrt.windows.foundation.collections.IIterable[winrt.windows.foundation.collections.IKeyValuePair[str, _winrt.winrt_base]], _winrt.winrt_base):
    ...
    size: int
    def clear() -> None:
        ...
    def first() -> winrt.windows.foundation.collections.IIterator[winrt.windows.foundation.collections.IKeyValuePair[str, _winrt.winrt_base]]:
        ...
    def get_view() -> winrt.windows.foundation.collections.IMapView[str, _winrt.winrt_base]:
        ...
    def has_key(key: str) -> bool:
        ...
    def insert(key: str, value: _winrt.winrt_base) -> bool:
        ...
    def lookup(key: str) -> _winrt.winrt_base:
        ...
    def remove(key: str) -> None:
        ...
    def add_map_changed(vhnd: winrt.windows.foundation.collections.MapChangedEventHandler[str, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_map_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class PasswordVault(_winrt.winrt_base):
    ...
    def add(credential: PasswordCredential) -> None:
        ...
    def find_all_by_resource(resource: str) -> winrt.windows.foundation.collections.IVectorView[PasswordCredential]:
        ...
    def find_all_by_user_name(user_name: str) -> winrt.windows.foundation.collections.IVectorView[PasswordCredential]:
        ...
    def remove(credential: PasswordCredential) -> None:
        ...
    def retrieve(resource: str, user_name: str) -> PasswordCredential:
        ...
    def retrieve_all() -> winrt.windows.foundation.collections.IVectorView[PasswordCredential]:
        ...

class WebAccount(IWebAccount, _winrt.winrt_base):
    ...
    state: WebAccountState
    user_name: str
    web_account_provider: WebAccountProvider
    id: str
    properties: winrt.windows.foundation.collections.IMapView[str, str]
    def get_picture_async(desized_size: WebAccountPictureSize) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.IRandomAccessStream]:
        ...
    def sign_out_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def sign_out_async(client_id: str) -> winrt.windows.foundation.IAsyncAction:
        ...

class WebAccountProvider(_winrt.winrt_base):
    ...
    display_name: str
    icon_uri: winrt.windows.foundation.Uri
    id: str
    authority: str
    display_purpose: str
    user: winrt.windows.system.User
    is_system_provider: bool

class IWebAccount(_winrt.winrt_base):
    ...
    state: WebAccountState
    user_name: str
    web_account_provider: WebAccountProvider
