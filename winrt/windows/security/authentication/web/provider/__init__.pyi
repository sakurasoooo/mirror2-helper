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
    import winrt.windows.security.authentication.web
except Exception:
    pass

try:
    import winrt.windows.security.authentication.web.core
except Exception:
    pass

try:
    import winrt.windows.security.credentials
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

try:
    import winrt.windows.web.http
except Exception:
    pass

class WebAccountClientViewType(enum.IntEnum):
    ID_ONLY = 0
    ID_AND_PROPERTIES = 1

class WebAccountProviderOperationKind(enum.IntEnum):
    REQUEST_TOKEN = 0
    GET_TOKEN_SILENTLY = 1
    ADD_ACCOUNT = 2
    MANAGE_ACCOUNT = 3
    DELETE_ACCOUNT = 4
    RETRIEVE_COOKIES = 5
    SIGN_OUT_ACCOUNT = 6

class WebAccountScope(enum.IntEnum):
    PER_USER = 0
    PER_APPLICATION = 1

class WebAccountSelectionOptions(enum.IntFlag):
    DEFAULT = 0
    NEW = 0x1

class WebAccountClientView(_winrt.winrt_base):
    ...
    account_pairwise_id: str
    application_callback_uri: winrt.windows.foundation.Uri
    type: WebAccountClientViewType

class WebAccountManager(_winrt.winrt_base):
    ...
    def add_web_account_async(web_account_id: str, web_account_user_name: str, props: winrt.windows.foundation.collections.IMapView[str, str]) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.security.credentials.WebAccount]:
        ...
    def add_web_account_async(web_account_id: str, web_account_user_name: str, props: winrt.windows.foundation.collections.IMapView[str, str], scope: WebAccountScope) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.security.credentials.WebAccount]:
        ...
    def add_web_account_async(web_account_id: str, web_account_user_name: str, props: winrt.windows.foundation.collections.IMapView[str, str], scope: WebAccountScope, per_user_web_account_id: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.security.credentials.WebAccount]:
        ...
    def add_web_account_for_user_async(user: winrt.windows.system.User, web_account_id: str, web_account_user_name: str, props: winrt.windows.foundation.collections.IMapView[str, str]) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.security.credentials.WebAccount]:
        ...
    def add_web_account_for_user_async(user: winrt.windows.system.User, web_account_id: str, web_account_user_name: str, props: winrt.windows.foundation.collections.IMapView[str, str], scope: WebAccountScope) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.security.credentials.WebAccount]:
        ...
    def add_web_account_for_user_async(user: winrt.windows.system.User, web_account_id: str, web_account_user_name: str, props: winrt.windows.foundation.collections.IMapView[str, str], scope: WebAccountScope, per_user_web_account_id: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.security.credentials.WebAccount]:
        ...
    def clear_per_user_from_per_app_account_async(per_app_account: winrt.windows.security.credentials.WebAccount) -> winrt.windows.foundation.IAsyncAction:
        ...
    def clear_view_async(web_account: winrt.windows.security.credentials.WebAccount, application_callback_uri: winrt.windows.foundation.Uri) -> winrt.windows.foundation.IAsyncAction:
        ...
    def clear_web_account_picture_async(web_account: winrt.windows.security.credentials.WebAccount) -> winrt.windows.foundation.IAsyncAction:
        ...
    def delete_web_account_async(web_account: winrt.windows.security.credentials.WebAccount) -> winrt.windows.foundation.IAsyncAction:
        ...
    def find_all_provider_web_accounts_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.security.credentials.WebAccount]]:
        ...
    def find_all_provider_web_accounts_for_user_async(user: winrt.windows.system.User) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.security.credentials.WebAccount]]:
        ...
    def get_per_user_from_per_app_account_async(per_app_account: winrt.windows.security.credentials.WebAccount) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.security.credentials.WebAccount]:
        ...
    def get_scope(web_account: winrt.windows.security.credentials.WebAccount) -> WebAccountScope:
        ...
    def get_views_async(web_account: winrt.windows.security.credentials.WebAccount) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[WebAccountClientView]]:
        ...
    def invalidate_app_cache_for_account_async(web_account: winrt.windows.security.credentials.WebAccount) -> winrt.windows.foundation.IAsyncAction:
        ...
    def invalidate_app_cache_for_all_accounts_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def pull_cookies_async(uri_string: str, caller_p_f_n: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def push_cookies_async(uri: winrt.windows.foundation.Uri, cookies: winrt.windows.foundation.collections.IVectorView[winrt.windows.web.http.HttpCookie]) -> winrt.windows.foundation.IAsyncAction:
        ...
    def set_per_app_to_per_user_account_async(per_app_account: winrt.windows.security.credentials.WebAccount, per_user_web_account_id: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def set_scope_async(web_account: winrt.windows.security.credentials.WebAccount, scope: WebAccountScope) -> winrt.windows.foundation.IAsyncAction:
        ...
    def set_view_async(web_account: winrt.windows.security.credentials.WebAccount, view: WebAccountClientView) -> winrt.windows.foundation.IAsyncAction:
        ...
    def set_web_account_picture_async(web_account: winrt.windows.security.credentials.WebAccount, web_account_picture: winrt.windows.storage.streams.IRandomAccessStream) -> winrt.windows.foundation.IAsyncAction:
        ...
    def update_web_account_properties_async(web_account: winrt.windows.security.credentials.WebAccount, web_account_user_name: str, additional_properties: winrt.windows.foundation.collections.IMapView[str, str]) -> winrt.windows.foundation.IAsyncAction:
        ...

class WebAccountProviderAddAccountOperation(IWebAccountProviderOperation, _winrt.winrt_base):
    ...
    kind: WebAccountProviderOperationKind
    def report_completed() -> None:
        ...

class WebAccountProviderDeleteAccountOperation(IWebAccountProviderOperation, IWebAccountProviderBaseReportOperation, _winrt.winrt_base):
    ...
    web_account: winrt.windows.security.credentials.WebAccount
    kind: WebAccountProviderOperationKind
    def report_completed() -> None:
        ...
    def report_error(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...

class WebAccountProviderGetTokenSilentOperation(IWebAccountProviderTokenOperation, IWebAccountProviderOperation, IWebAccountProviderSilentReportOperation, IWebAccountProviderBaseReportOperation, _winrt.winrt_base):
    ...
    kind: WebAccountProviderOperationKind
    cache_expiration_time: winrt.windows.foundation.DateTime
    provider_request: WebProviderTokenRequest
    provider_responses: winrt.windows.foundation.collections.IVector[WebProviderTokenResponse]
    def report_completed() -> None:
        ...
    def report_error(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...
    def report_user_interaction_required() -> None:
        ...
    def report_user_interaction_required(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...

class WebAccountProviderManageAccountOperation(IWebAccountProviderOperation, _winrt.winrt_base):
    ...
    web_account: winrt.windows.security.credentials.WebAccount
    kind: WebAccountProviderOperationKind
    def report_completed() -> None:
        ...

class WebAccountProviderRequestTokenOperation(IWebAccountProviderTokenOperation, IWebAccountProviderOperation, IWebAccountProviderUIReportOperation, IWebAccountProviderBaseReportOperation, _winrt.winrt_base):
    ...
    kind: WebAccountProviderOperationKind
    cache_expiration_time: winrt.windows.foundation.DateTime
    provider_request: WebProviderTokenRequest
    provider_responses: winrt.windows.foundation.collections.IVector[WebProviderTokenResponse]
    def report_completed() -> None:
        ...
    def report_error(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...
    def report_user_canceled() -> None:
        ...

class WebAccountProviderRetrieveCookiesOperation(IWebAccountProviderOperation, IWebAccountProviderBaseReportOperation, _winrt.winrt_base):
    ...
    kind: WebAccountProviderOperationKind
    uri: winrt.windows.foundation.Uri
    application_callback_uri: winrt.windows.foundation.Uri
    context: winrt.windows.foundation.Uri
    cookies: winrt.windows.foundation.collections.IVector[winrt.windows.web.http.HttpCookie]
    def report_completed() -> None:
        ...
    def report_error(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...

class WebAccountProviderSignOutAccountOperation(IWebAccountProviderOperation, IWebAccountProviderBaseReportOperation, _winrt.winrt_base):
    ...
    kind: WebAccountProviderOperationKind
    application_callback_uri: winrt.windows.foundation.Uri
    client_id: str
    web_account: winrt.windows.security.credentials.WebAccount
    def report_completed() -> None:
        ...
    def report_error(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...

class WebAccountProviderTriggerDetails(IWebAccountProviderTokenObjects, IWebAccountProviderTokenObjects2, _winrt.winrt_base):
    ...
    operation: IWebAccountProviderOperation
    user: winrt.windows.system.User

class WebProviderTokenRequest(_winrt.winrt_base):
    ...
    application_callback_uri: winrt.windows.foundation.Uri
    client_request: winrt.windows.security.authentication.web.core.WebTokenRequest
    web_account_selection_options: WebAccountSelectionOptions
    web_accounts: winrt.windows.foundation.collections.IVectorView[winrt.windows.security.credentials.WebAccount]
    application_package_family_name: str
    application_process_name: str
    def check_application_for_capability_async(capability_name: str) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def get_application_token_binding_key_async(key_type: winrt.windows.security.authentication.web.TokenBindingKeyType, target: winrt.windows.foundation.Uri) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.security.cryptography.core.CryptographicKey]:
        ...
    def get_application_token_binding_key_id_async(key_type: winrt.windows.security.authentication.web.TokenBindingKeyType, target: winrt.windows.foundation.Uri) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.IBuffer]:
        ...

class WebProviderTokenResponse(_winrt.winrt_base):
    ...
    client_response: winrt.windows.security.authentication.web.core.WebTokenResponse

class IWebAccountProviderBaseReportOperation(_winrt.winrt_base):
    ...
    def report_completed() -> None:
        ...
    def report_error(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...

class IWebAccountProviderOperation(_winrt.winrt_base):
    ...
    kind: WebAccountProviderOperationKind

class IWebAccountProviderSilentReportOperation(IWebAccountProviderBaseReportOperation, _winrt.winrt_base):
    ...
    def report_user_interaction_required() -> None:
        ...
    def report_user_interaction_required(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...
    def report_completed() -> None:
        ...
    def report_error(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...

class IWebAccountProviderTokenObjects(_winrt.winrt_base):
    ...
    operation: IWebAccountProviderOperation

class IWebAccountProviderTokenObjects2(IWebAccountProviderTokenObjects, _winrt.winrt_base):
    ...
    user: winrt.windows.system.User
    operation: IWebAccountProviderOperation

class IWebAccountProviderTokenOperation(IWebAccountProviderOperation, _winrt.winrt_base):
    ...
    cache_expiration_time: winrt.windows.foundation.DateTime
    provider_request: WebProviderTokenRequest
    provider_responses: winrt.windows.foundation.collections.IVector[WebProviderTokenResponse]
    kind: WebAccountProviderOperationKind

class IWebAccountProviderUIReportOperation(IWebAccountProviderBaseReportOperation, _winrt.winrt_base):
    ...
    def report_user_canceled() -> None:
        ...
    def report_completed() -> None:
        ...
    def report_error(value: winrt.windows.security.authentication.web.core.WebProviderError) -> None:
        ...
