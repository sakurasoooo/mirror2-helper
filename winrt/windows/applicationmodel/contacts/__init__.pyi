# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.data.text
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

try:
    import winrt.windows.system
except Exception:
    pass

try:
    import winrt.windows.ui
except Exception:
    pass

try:
    import winrt.windows.ui.popups
except Exception:
    pass

try:
    import winrt.windows.ui.viewmanagement
except Exception:
    pass

class ContactAddressKind(enum.IntEnum):
    HOME = 0
    WORK = 1
    OTHER = 2

class ContactAnnotationOperations(enum.IntFlag):
    NONE = 0
    CONTACT_PROFILE = 0x1
    MESSAGE = 0x2
    AUDIO_CALL = 0x4
    VIDEO_CALL = 0x8
    SOCIAL_FEEDS = 0x10
    SHARE = 0x20

class ContactAnnotationStoreAccessType(enum.IntEnum):
    APP_ANNOTATIONS_READ_WRITE = 0
    ALL_ANNOTATIONS_READ_WRITE = 1

class ContactBatchStatus(enum.IntEnum):
    SUCCESS = 0
    SERVER_SEARCH_SYNC_MANAGER_ERROR = 1
    SERVER_SEARCH_UNKNOWN_ERROR = 2

class ContactCardHeaderKind(enum.IntEnum):
    DEFAULT = 0
    BASIC = 1
    ENTERPRISE = 2

class ContactCardTabKind(enum.IntEnum):
    DEFAULT = 0
    EMAIL = 1
    MESSAGING = 2
    PHONE = 3
    VIDEO = 4
    ORGANIZATIONAL_HIERARCHY = 5

class ContactChangeType(enum.IntEnum):
    CREATED = 0
    MODIFIED = 1
    DELETED = 2
    CHANGE_TRACKING_LOST = 3

class ContactDateKind(enum.IntEnum):
    BIRTHDAY = 0
    ANNIVERSARY = 1
    OTHER = 2

class ContactEmailKind(enum.IntEnum):
    PERSONAL = 0
    WORK = 1
    OTHER = 2

class ContactFieldCategory(enum.IntEnum):
    NONE = 0
    HOME = 1
    WORK = 2
    MOBILE = 3
    OTHER = 4

class ContactFieldType(enum.IntEnum):
    EMAIL = 0
    PHONE_NUMBER = 1
    LOCATION = 2
    INSTANT_MESSAGE = 3
    CUSTOM = 4
    CONNECTED_SERVICE_ACCOUNT = 5
    IMPORTANT_DATE = 6
    ADDRESS = 7
    SIGNIFICANT_OTHER = 8
    NOTES = 9
    WEBSITE = 10
    JOB_INFO = 11

class ContactListOtherAppReadAccess(enum.IntEnum):
    SYSTEM_ONLY = 0
    LIMITED = 1
    FULL = 2
    NONE = 3

class ContactListOtherAppWriteAccess(enum.IntEnum):
    NONE = 0
    SYSTEM_ONLY = 1
    LIMITED = 2

class ContactListSyncStatus(enum.IntEnum):
    IDLE = 0
    SYNCING = 1
    UP_TO_DATE = 2
    AUTHENTICATION_ERROR = 3
    POLICY_ERROR = 4
    UNKNOWN_ERROR = 5
    MANUAL_ACCOUNT_REMOVAL_REQUIRED = 6

class ContactMatchReasonKind(enum.IntEnum):
    NAME = 0
    EMAIL_ADDRESS = 1
    PHONE_NUMBER = 2
    JOB_INFO = 3
    YOMI_NAME = 4
    OTHER = 5

class ContactNameOrder(enum.IntEnum):
    FIRST_NAME_LAST_NAME = 0
    LAST_NAME_FIRST_NAME = 1

class ContactPhoneKind(enum.IntEnum):
    HOME = 0
    MOBILE = 1
    WORK = 2
    OTHER = 3
    PAGER = 4
    BUSINESS_FAX = 5
    HOME_FAX = 6
    COMPANY = 7
    ASSISTANT = 8
    RADIO = 9

class ContactQueryDesiredFields(enum.IntFlag):
    NONE = 0
    PHONE_NUMBER = 0x1
    EMAIL_ADDRESS = 0x2
    POSTAL_ADDRESS = 0x4

class ContactQuerySearchFields(enum.IntFlag):
    NONE = 0
    NAME = 0x1
    EMAIL = 0x2
    PHONE = 0x4
    ALL = 0xffffffff

class ContactQuerySearchScope(enum.IntEnum):
    LOCAL = 0
    SERVER = 1

class ContactRelationship(enum.IntEnum):
    OTHER = 0
    SPOUSE = 1
    PARTNER = 2
    SIBLING = 3
    PARENT = 4
    CHILD = 5

class ContactSelectionMode(enum.IntEnum):
    CONTACTS = 0
    FIELDS = 1

class ContactStoreAccessType(enum.IntEnum):
    APP_CONTACTS_READ_WRITE = 0
    ALL_CONTACTS_READ_ONLY = 1
    ALL_CONTACTS_READ_WRITE = 2

class PinnedContactSurface(enum.IntEnum):
    START_MENU = 0
    TASKBAR = 1

class AggregateContactManager(_winrt.winrt_base):
    ...
    def find_raw_contacts_async(contact: Contact) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Contact]]:
        ...
    def set_remote_identification_information_async(contact_list_id: str, remote_source_id: str, account_id: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def try_link_contacts_async(primary_contact: Contact, secondary_contact: Contact) -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def try_set_preferred_source_for_picture_async(aggregate_contact: Contact, raw_contact: Contact) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def unlink_raw_contact_async(contact: Contact) -> winrt.windows.foundation.IAsyncAction:
        ...

class Contact(_winrt.winrt_base):
    ...
    thumbnail: winrt.windows.storage.streams.IRandomAccessStreamReference
    name: str
    fields: winrt.windows.foundation.collections.IVector[IContactField]
    id: str
    notes: str
    connected_service_accounts: winrt.windows.foundation.collections.IVector[ContactConnectedServiceAccount]
    emails: winrt.windows.foundation.collections.IVector[ContactEmail]
    addresses: winrt.windows.foundation.collections.IVector[ContactAddress]
    important_dates: winrt.windows.foundation.collections.IVector[ContactDate]
    job_info: winrt.windows.foundation.collections.IVector[ContactJobInfo]
    data_suppliers: winrt.windows.foundation.collections.IVector[str]
    phones: winrt.windows.foundation.collections.IVector[ContactPhone]
    provider_properties: winrt.windows.foundation.collections.IPropertySet
    significant_others: winrt.windows.foundation.collections.IVector[ContactSignificantOther]
    websites: winrt.windows.foundation.collections.IVector[ContactWebsite]
    source_display_picture: winrt.windows.storage.streams.IRandomAccessStreamReference
    text_tone_token: str
    display_name_override: str
    display_picture_user_update_time: winrt.windows.foundation.DateTime
    nickname: str
    remote_id: str
    ring_tone_token: str
    contact_list_id: str
    large_display_picture: winrt.windows.storage.streams.IRandomAccessStreamReference
    small_display_picture: winrt.windows.storage.streams.IRandomAccessStreamReference
    sort_name: str
    aggregate_id: str
    full_name: str
    is_aggregate: bool
    is_display_picture_manually_set: bool
    is_me: bool
    yomi_given_name: str
    honorific_name_suffix: str
    yomi_family_name: str
    middle_name: str
    last_name: str
    honorific_name_prefix: str
    first_name: str
    display_name: str
    yomi_display_name: str

class ContactAddress(_winrt.winrt_base):
    ...
    street_address: str
    region: str
    postal_code: str
    locality: str
    kind: ContactAddressKind
    description: str
    country: str

class ContactAnnotation(_winrt.winrt_base):
    ...
    supported_operations: ContactAnnotationOperations
    remote_id: str
    contact_id: str
    annotation_list_id: str
    id: str
    is_disabled: bool
    provider_properties: winrt.windows.foundation.collections.ValueSet
    contact_list_id: str

class ContactAnnotationList(_winrt.winrt_base):
    ...
    id: str
    provider_package_family_name: str
    user_data_account_id: str
    def delete_annotation_async(annotation: ContactAnnotation) -> winrt.windows.foundation.IAsyncAction:
        ...
    def delete_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def find_annotations_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ContactAnnotation]]:
        ...
    def find_annotations_by_remote_id_async(remote_id: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ContactAnnotation]]:
        ...
    def get_annotation_async(annotation_id: str) -> winrt.windows.foundation.IAsyncOperation[ContactAnnotation]:
        ...
    def try_save_annotation_async(annotation: ContactAnnotation) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...

class ContactAnnotationStore(_winrt.winrt_base):
    ...
    def create_annotation_list_async() -> winrt.windows.foundation.IAsyncOperation[ContactAnnotationList]:
        ...
    def create_annotation_list_async(user_data_account_id: str) -> winrt.windows.foundation.IAsyncOperation[ContactAnnotationList]:
        ...
    def disable_annotation_async(annotation: ContactAnnotation) -> winrt.windows.foundation.IAsyncAction:
        ...
    def find_annotation_lists_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ContactAnnotationList]]:
        ...
    def find_annotations_for_contact_async(contact: Contact) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ContactAnnotation]]:
        ...
    def find_annotations_for_contact_list_async(contact_list_id: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ContactAnnotation]]:
        ...
    def find_contact_ids_by_email_async(email_address: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[str]]:
        ...
    def find_contact_ids_by_phone_number_async(phone_number: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[str]]:
        ...
    def get_annotation_list_async(annotation_list_id: str) -> winrt.windows.foundation.IAsyncOperation[ContactAnnotationList]:
        ...

class ContactBatch(_winrt.winrt_base):
    ...
    contacts: winrt.windows.foundation.collections.IVectorView[Contact]
    status: ContactBatchStatus

class ContactCardDelayedDataLoader(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def close() -> None:
        ...
    def set_data(contact: Contact) -> None:
        ...

class ContactCardOptions(_winrt.winrt_base):
    ...
    initial_tab_kind: ContactCardTabKind
    header_kind: ContactCardHeaderKind
    server_search_contact_list_ids: winrt.windows.foundation.collections.IVector[str]

class ContactChange(_winrt.winrt_base):
    ...
    change_type: ContactChangeType
    contact: Contact

class ContactChangeReader(_winrt.winrt_base):
    ...
    def accept_changes() -> None:
        ...
    def accept_changes_through(last_change_to_accept: ContactChange) -> None:
        ...
    def read_batch_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ContactChange]]:
        ...

class ContactChangeTracker(_winrt.winrt_base):
    ...
    is_tracking: bool
    def enable() -> None:
        ...
    def get_change_reader() -> ContactChangeReader:
        ...
    def reset() -> None:
        ...

class ContactChangedDeferral(_winrt.winrt_base):
    ...
    def complete() -> None:
        ...

class ContactChangedEventArgs(_winrt.winrt_base):
    ...
    def get_deferral() -> ContactChangedDeferral:
        ...

class ContactConnectedServiceAccount(_winrt.winrt_base):
    ...
    service_name: str
    id: str

class ContactDate(_winrt.winrt_base):
    ...
    year: typing.Optional[int]
    month: typing.Optional[int]
    kind: ContactDateKind
    description: str
    day: typing.Optional[int]

class ContactEmail(_winrt.winrt_base):
    ...
    kind: ContactEmailKind
    description: str
    address: str

class ContactField(IContactField, _winrt.winrt_base):
    ...
    category: ContactFieldCategory
    name: str
    type: ContactFieldType
    value: str

class ContactFieldFactory(IContactFieldFactory, IContactLocationFieldFactory, IContactInstantMessageFieldFactory, _winrt.winrt_base):
    ...
    def create_field(value: str, type: ContactFieldType) -> ContactField:
        ...
    def create_field(value: str, type: ContactFieldType, category: ContactFieldCategory) -> ContactField:
        ...
    def create_field(name: str, value: str, type: ContactFieldType, category: ContactFieldCategory) -> ContactField:
        ...
    def create_instant_message(user_name: str) -> ContactInstantMessageField:
        ...
    def create_instant_message(user_name: str, category: ContactFieldCategory) -> ContactInstantMessageField:
        ...
    def create_instant_message(user_name: str, category: ContactFieldCategory, service: str, display_text: str, verb: winrt.windows.foundation.Uri) -> ContactInstantMessageField:
        ...
    def create_location(unstructured_address: str) -> ContactLocationField:
        ...
    def create_location(unstructured_address: str, category: ContactFieldCategory) -> ContactLocationField:
        ...
    def create_location(unstructured_address: str, category: ContactFieldCategory, street: str, city: str, region: str, country: str, postal_code: str) -> ContactLocationField:
        ...

class ContactGroup(_winrt.winrt_base):
    ...

class ContactInformation(_winrt.winrt_base):
    ...
    custom_fields: winrt.windows.foundation.collections.IVectorView[ContactField]
    emails: winrt.windows.foundation.collections.IVectorView[ContactField]
    instant_messages: winrt.windows.foundation.collections.IVectorView[ContactInstantMessageField]
    locations: winrt.windows.foundation.collections.IVectorView[ContactLocationField]
    name: str
    phone_numbers: winrt.windows.foundation.collections.IVectorView[ContactField]
    def get_thumbnail_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.IRandomAccessStreamWithContentType]:
        ...
    def query_custom_fields(custom_name: str) -> winrt.windows.foundation.collections.IVectorView[ContactField]:
        ...

class ContactInstantMessageField(IContactField, _winrt.winrt_base):
    ...
    category: ContactFieldCategory
    name: str
    type: ContactFieldType
    value: str
    display_text: str
    launch_uri: winrt.windows.foundation.Uri
    service: str
    user_name: str

class ContactJobInfo(_winrt.winrt_base):
    ...
    title: str
    office: str
    manager: str
    description: str
    department: str
    company_yomi_name: str
    company_name: str
    company_address: str

class ContactLaunchActionVerbs(_winrt.winrt_base):
    ...
    call: str
    map: str
    message: str
    post: str
    video_call: str

class ContactList(_winrt.winrt_base):
    ...
    supports_server_search: bool
    is_hidden: bool
    other_app_write_access: ContactListOtherAppWriteAccess
    display_name: str
    other_app_read_access: ContactListOtherAppReadAccess
    change_tracker: ContactChangeTracker
    source_display_name: str
    id: str
    sync_manager: ContactListSyncManager
    user_data_account_id: str
    sync_constraints: ContactListSyncConstraints
    limited_write_operations: ContactListLimitedWriteOperations
    def delete_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def delete_contact_async(contact: Contact) -> winrt.windows.foundation.IAsyncAction:
        ...
    def get_change_tracker(identity: str) -> ContactChangeTracker:
        ...
    def get_contact_async(contact_id: str) -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def get_contact_from_remote_id_async(remote_id: str) -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def get_contact_reader() -> ContactReader:
        ...
    def get_contact_reader(options: ContactQueryOptions) -> ContactReader:
        ...
    def get_me_contact_async() -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def register_sync_manager_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def save_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def save_contact_async(contact: Contact) -> winrt.windows.foundation.IAsyncAction:
        ...
    def add_contact_changed(value: winrt.windows.foundation.TypedEventHandler[ContactList, ContactChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_contact_changed(value: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ContactListLimitedWriteOperations(_winrt.winrt_base):
    ...
    def try_create_or_update_contact_async(contact: Contact) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_delete_contact_async(contact_id: str) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...

class ContactListSyncConstraints(_winrt.winrt_base):
    ...
    can_sync_descriptions: bool
    max_company_phone_numbers: typing.Optional[int]
    max_child_relationships: typing.Optional[int]
    max_business_fax_phone_numbers: typing.Optional[int]
    max_birthday_dates: typing.Optional[int]
    max_assistant_phone_numbers: typing.Optional[int]
    max_other_addresses: typing.Optional[int]
    max_anniversary_dates: typing.Optional[int]
    max_home_addresses: typing.Optional[int]
    max_other_dates: typing.Optional[int]
    max_mobile_phone_numbers: typing.Optional[int]
    max_job_info: typing.Optional[int]
    max_home_phone_numbers: typing.Optional[int]
    max_home_fax_phone_numbers: typing.Optional[int]
    max_other_email_addresses: typing.Optional[int]
    max_personal_email_addresses: typing.Optional[int]
    max_partner_relationships: typing.Optional[int]
    max_parent_relationships: typing.Optional[int]
    max_pager_phone_numbers: typing.Optional[int]
    max_other_relationships: typing.Optional[int]
    max_other_phone_numbers: typing.Optional[int]
    max_radio_phone_numbers: typing.Optional[int]
    max_work_phone_numbers: typing.Optional[int]
    max_work_email_addresses: typing.Optional[int]
    max_work_addresses: typing.Optional[int]
    max_websites: typing.Optional[int]
    max_spouse_relationships: typing.Optional[int]
    max_sibling_relationships: typing.Optional[int]

class ContactListSyncManager(_winrt.winrt_base):
    ...
    status: ContactListSyncStatus
    last_successful_sync_time: winrt.windows.foundation.DateTime
    last_attempted_sync_time: winrt.windows.foundation.DateTime
    def sync_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def add_sync_status_changed(handler: winrt.windows.foundation.TypedEventHandler[ContactListSyncManager, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_sync_status_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ContactLocationField(IContactField, _winrt.winrt_base):
    ...
    category: ContactFieldCategory
    name: str
    type: ContactFieldType
    value: str
    city: str
    country: str
    postal_code: str
    region: str
    street: str
    unstructured_address: str

class ContactManager(_winrt.winrt_base):
    ...
    system_sort_order: ContactNameOrder
    system_display_name_order: ContactNameOrder
    include_middle_name_in_system_display_and_sort: bool
    def convert_contact_to_v_card_async(contact: Contact) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.RandomAccessStreamReference]:
        ...
    def convert_contact_to_v_card_async(contact: Contact, max_bytes: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.RandomAccessStreamReference]:
        ...
    def convert_v_card_to_contact_async(v_card: winrt.windows.storage.streams.IRandomAccessStreamReference) -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def get_for_user(user: winrt.windows.system.User) -> ContactManagerForUser:
        ...
    def is_show_contact_card_supported() -> bool:
        ...
    def is_show_delay_loaded_contact_card_supported() -> bool:
        ...
    def is_show_full_contact_card_supported_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def request_annotation_store_async(access_type: ContactAnnotationStoreAccessType) -> winrt.windows.foundation.IAsyncOperation[ContactAnnotationStore]:
        ...
    def request_store_async() -> winrt.windows.foundation.IAsyncOperation[ContactStore]:
        ...
    def request_store_async(access_type: ContactStoreAccessType) -> winrt.windows.foundation.IAsyncOperation[ContactStore]:
        ...
    def show_contact_card(contact: Contact, selection: winrt.windows.foundation.Rect) -> None:
        ...
    def show_contact_card(contact: Contact, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement) -> None:
        ...
    def show_contact_card(contact: Contact, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement, contact_card_options: ContactCardOptions) -> None:
        ...
    def show_delay_loaded_contact_card(contact: Contact, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement) -> ContactCardDelayedDataLoader:
        ...
    def show_delay_loaded_contact_card(contact: Contact, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement, contact_card_options: ContactCardOptions) -> ContactCardDelayedDataLoader:
        ...
    def show_full_contact_card(contact: Contact, full_contact_card_options: FullContactCardOptions) -> None:
        ...

class ContactManagerForUser(_winrt.winrt_base):
    ...
    system_sort_order: ContactNameOrder
    system_display_name_order: ContactNameOrder
    user: winrt.windows.system.User
    def convert_contact_to_v_card_async(contact: Contact) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.RandomAccessStreamReference]:
        ...
    def convert_contact_to_v_card_async(contact: Contact, max_bytes: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.RandomAccessStreamReference]:
        ...
    def convert_v_card_to_contact_async(v_card: winrt.windows.storage.streams.IRandomAccessStreamReference) -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def request_annotation_store_async(access_type: ContactAnnotationStoreAccessType) -> winrt.windows.foundation.IAsyncOperation[ContactAnnotationStore]:
        ...
    def request_store_async(access_type: ContactStoreAccessType) -> winrt.windows.foundation.IAsyncOperation[ContactStore]:
        ...
    def show_full_contact_card(contact: Contact, full_contact_card_options: FullContactCardOptions) -> None:
        ...

class ContactMatchReason(_winrt.winrt_base):
    ...
    field: ContactMatchReasonKind
    segments: winrt.windows.foundation.collections.IVectorView[winrt.windows.data.text.TextSegment]
    text: str

class ContactPanel(_winrt.winrt_base):
    ...
    header_color: typing.Optional[winrt.windows.ui.Color]
    def close_panel() -> None:
        ...
    def add_closing(handler: winrt.windows.foundation.TypedEventHandler[ContactPanel, ContactPanelClosingEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_closing(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_launch_full_app_requested(handler: winrt.windows.foundation.TypedEventHandler[ContactPanel, ContactPanelLaunchFullAppRequestedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_launch_full_app_requested(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ContactPanelClosingEventArgs(_winrt.winrt_base):
    ...
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...

class ContactPanelLaunchFullAppRequestedEventArgs(_winrt.winrt_base):
    ...
    handled: bool

class ContactPhone(_winrt.winrt_base):
    ...
    number: str
    kind: ContactPhoneKind
    description: str

class ContactPicker(_winrt.winrt_base):
    ...
    selection_mode: ContactSelectionMode
    commit_button_text: str
    desired_fields: winrt.windows.foundation.collections.IVector[str]
    desired_fields_with_contact_field_type: winrt.windows.foundation.collections.IVector[ContactFieldType]
    user: winrt.windows.system.User
    def create_for_user(user: winrt.windows.system.User) -> ContactPicker:
        ...
    def is_supported_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def pick_contact_async() -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def pick_contacts_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVector[Contact]]:
        ...
    def pick_multiple_contacts_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ContactInformation]]:
        ...
    def pick_single_contact_async() -> winrt.windows.foundation.IAsyncOperation[ContactInformation]:
        ...

class ContactQueryOptions(_winrt.winrt_base):
    ...
    include_contacts_from_hidden_lists: bool
    desired_operations: ContactAnnotationOperations
    desired_fields: ContactQueryDesiredFields
    annotation_list_ids: winrt.windows.foundation.collections.IVector[str]
    contact_list_ids: winrt.windows.foundation.collections.IVector[str]
    text_search: ContactQueryTextSearch

class ContactQueryTextSearch(_winrt.winrt_base):
    ...
    text: str
    search_scope: ContactQuerySearchScope
    fields: ContactQuerySearchFields

class ContactReader(_winrt.winrt_base):
    ...
    def get_matching_properties_with_match_reason(contact: Contact) -> winrt.windows.foundation.collections.IVectorView[ContactMatchReason]:
        ...
    def read_batch_async() -> winrt.windows.foundation.IAsyncOperation[ContactBatch]:
        ...

class ContactSignificantOther(_winrt.winrt_base):
    ...
    name: str
    description: str
    relationship: ContactRelationship

class ContactStore(_winrt.winrt_base):
    ...
    aggregate_contact_manager: AggregateContactManager
    change_tracker: ContactChangeTracker
    def create_contact_list_async(display_name: str) -> winrt.windows.foundation.IAsyncOperation[ContactList]:
        ...
    def create_contact_list_async(display_name: str, user_data_account_id: str) -> winrt.windows.foundation.IAsyncOperation[ContactList]:
        ...
    def find_contact_lists_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ContactList]]:
        ...
    def find_contacts_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Contact]]:
        ...
    def find_contacts_async(search_text: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Contact]]:
        ...
    def get_change_tracker(identity: str) -> ContactChangeTracker:
        ...
    def get_contact_async(contact_id: str) -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def get_contact_list_async(contact_list_id: str) -> winrt.windows.foundation.IAsyncOperation[ContactList]:
        ...
    def get_contact_reader() -> ContactReader:
        ...
    def get_contact_reader(options: ContactQueryOptions) -> ContactReader:
        ...
    def get_me_contact_async() -> winrt.windows.foundation.IAsyncOperation[Contact]:
        ...
    def add_contact_changed(value: winrt.windows.foundation.TypedEventHandler[ContactStore, ContactChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_contact_changed(value: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ContactStoreNotificationTriggerDetails(_winrt.winrt_base):
    ...

class ContactWebsite(_winrt.winrt_base):
    ...
    uri: winrt.windows.foundation.Uri
    description: str
    raw_value: str

class FullContactCardOptions(_winrt.winrt_base):
    ...
    desired_remaining_view: winrt.windows.ui.viewmanagement.ViewSizePreference

class KnownContactField(_winrt.winrt_base):
    ...
    email: str
    instant_message: str
    location: str
    phone_number: str
    def convert_name_to_type(name: str) -> ContactFieldType:
        ...
    def convert_type_to_name(type: ContactFieldType) -> str:
        ...

class PinnedContactIdsQueryResult(_winrt.winrt_base):
    ...
    contact_ids: winrt.windows.foundation.collections.IVector[str]

class PinnedContactManager(_winrt.winrt_base):
    ...
    user: winrt.windows.system.User
    def get_default() -> PinnedContactManager:
        ...
    def get_for_user(user: winrt.windows.system.User) -> PinnedContactManager:
        ...
    def get_pinned_contact_ids_async() -> winrt.windows.foundation.IAsyncOperation[PinnedContactIdsQueryResult]:
        ...
    def is_contact_pinned(contact: Contact, surface: PinnedContactSurface) -> bool:
        ...
    def is_pin_surface_supported(surface: PinnedContactSurface) -> bool:
        ...
    def is_supported() -> bool:
        ...
    def request_pin_contact_async(contact: Contact, surface: PinnedContactSurface) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def request_pin_contacts_async(contacts: typing.Iterable[Contact], surface: PinnedContactSurface) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def request_unpin_contact_async(contact: Contact, surface: PinnedContactSurface) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def signal_contact_activity(contact: Contact) -> None:
        ...

class IContactField(_winrt.winrt_base):
    ...
    category: ContactFieldCategory
    name: str
    type: ContactFieldType
    value: str

class IContactFieldFactory(_winrt.winrt_base):
    ...
    def create_field(value: str, type: ContactFieldType) -> ContactField:
        ...
    def create_field(value: str, type: ContactFieldType, category: ContactFieldCategory) -> ContactField:
        ...
    def create_field(name: str, value: str, type: ContactFieldType, category: ContactFieldCategory) -> ContactField:
        ...

class IContactInstantMessageFieldFactory(_winrt.winrt_base):
    ...
    def create_instant_message(user_name: str) -> ContactInstantMessageField:
        ...
    def create_instant_message(user_name: str, category: ContactFieldCategory) -> ContactInstantMessageField:
        ...
    def create_instant_message(user_name: str, category: ContactFieldCategory, service: str, display_text: str, verb: winrt.windows.foundation.Uri) -> ContactInstantMessageField:
        ...

class IContactLocationFieldFactory(_winrt.winrt_base):
    ...
    def create_location(unstructured_address: str) -> ContactLocationField:
        ...
    def create_location(unstructured_address: str, category: ContactFieldCategory) -> ContactLocationField:
        ...
    def create_location(unstructured_address: str, category: ContactFieldCategory, street: str, city: str, region: str, country: str, postal_code: str) -> ContactLocationField:
        ...
