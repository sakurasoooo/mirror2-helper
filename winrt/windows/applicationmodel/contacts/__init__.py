# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.ApplicationModel.Contacts")

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

AggregateContactManager = _ns_module.AggregateContactManager
Contact = _ns_module.Contact
ContactAddress = _ns_module.ContactAddress
ContactAnnotation = _ns_module.ContactAnnotation
ContactAnnotationList = _ns_module.ContactAnnotationList
ContactAnnotationStore = _ns_module.ContactAnnotationStore
ContactBatch = _ns_module.ContactBatch
ContactCardDelayedDataLoader = _ns_module.ContactCardDelayedDataLoader
ContactCardOptions = _ns_module.ContactCardOptions
ContactChange = _ns_module.ContactChange
ContactChangeReader = _ns_module.ContactChangeReader
ContactChangeTracker = _ns_module.ContactChangeTracker
ContactChangedDeferral = _ns_module.ContactChangedDeferral
ContactChangedEventArgs = _ns_module.ContactChangedEventArgs
ContactConnectedServiceAccount = _ns_module.ContactConnectedServiceAccount
ContactDate = _ns_module.ContactDate
ContactEmail = _ns_module.ContactEmail
ContactField = _ns_module.ContactField
ContactFieldFactory = _ns_module.ContactFieldFactory
ContactGroup = _ns_module.ContactGroup
ContactInformation = _ns_module.ContactInformation
ContactInstantMessageField = _ns_module.ContactInstantMessageField
ContactJobInfo = _ns_module.ContactJobInfo
ContactLaunchActionVerbs = _ns_module.ContactLaunchActionVerbs
ContactList = _ns_module.ContactList
ContactListLimitedWriteOperations = _ns_module.ContactListLimitedWriteOperations
ContactListSyncConstraints = _ns_module.ContactListSyncConstraints
ContactListSyncManager = _ns_module.ContactListSyncManager
ContactLocationField = _ns_module.ContactLocationField
ContactManager = _ns_module.ContactManager
ContactManagerForUser = _ns_module.ContactManagerForUser
ContactMatchReason = _ns_module.ContactMatchReason
ContactPanel = _ns_module.ContactPanel
ContactPanelClosingEventArgs = _ns_module.ContactPanelClosingEventArgs
ContactPanelLaunchFullAppRequestedEventArgs = _ns_module.ContactPanelLaunchFullAppRequestedEventArgs
ContactPhone = _ns_module.ContactPhone
ContactPicker = _ns_module.ContactPicker
ContactQueryOptions = _ns_module.ContactQueryOptions
ContactQueryTextSearch = _ns_module.ContactQueryTextSearch
ContactReader = _ns_module.ContactReader
ContactSignificantOther = _ns_module.ContactSignificantOther
ContactStore = _ns_module.ContactStore
ContactStoreNotificationTriggerDetails = _ns_module.ContactStoreNotificationTriggerDetails
ContactWebsite = _ns_module.ContactWebsite
FullContactCardOptions = _ns_module.FullContactCardOptions
KnownContactField = _ns_module.KnownContactField
PinnedContactIdsQueryResult = _ns_module.PinnedContactIdsQueryResult
PinnedContactManager = _ns_module.PinnedContactManager
IContactField = _ns_module.IContactField
IContactFieldFactory = _ns_module.IContactFieldFactory
IContactInstantMessageFieldFactory = _ns_module.IContactInstantMessageFieldFactory
IContactLocationFieldFactory = _ns_module.IContactLocationFieldFactory
