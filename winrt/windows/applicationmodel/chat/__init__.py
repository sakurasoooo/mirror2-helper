# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.ApplicationModel.Chat")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.media.mediaproperties
except Exception:
    pass

try:
    import winrt.windows.security.credentials
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class ChatConversationThreadingKind(enum.IntEnum):
    PARTICIPANTS = 0
    CONTACT_ID = 1
    CONVERSATION_ID = 2
    CUSTOM = 3

class ChatItemKind(enum.IntEnum):
    MESSAGE = 0
    CONVERSATION = 1

class ChatMessageChangeType(enum.IntEnum):
    MESSAGE_CREATED = 0
    MESSAGE_MODIFIED = 1
    MESSAGE_DELETED = 2
    CHANGE_TRACKING_LOST = 3

class ChatMessageKind(enum.IntEnum):
    STANDARD = 0
    FILE_TRANSFER_REQUEST = 1
    TRANSPORT_CUSTOM = 2
    JOINED_CONVERSATION = 3
    LEFT_CONVERSATION = 4
    OTHER_PARTICIPANT_JOINED_CONVERSATION = 5
    OTHER_PARTICIPANT_LEFT_CONVERSATION = 6

class ChatMessageOperatorKind(enum.IntEnum):
    UNSPECIFIED = 0
    SMS = 1
    MMS = 2
    RCS = 3

class ChatMessageStatus(enum.IntEnum):
    DRAFT = 0
    SENDING = 1
    SENT = 2
    SEND_RETRY_NEEDED = 3
    SEND_FAILED = 4
    RECEIVED = 5
    RECEIVE_DOWNLOAD_NEEDED = 6
    RECEIVE_DOWNLOAD_FAILED = 7
    RECEIVE_DOWNLOADING = 8
    DELETED = 9
    DECLINED = 10
    CANCELLED = 11
    RECALLED = 12
    RECEIVE_RETRY_NEEDED = 13

class ChatMessageTransportKind(enum.IntEnum):
    TEXT = 0
    UNTRIAGED = 1
    BLOCKED = 2
    CUSTOM = 3

class ChatMessageValidationStatus(enum.IntEnum):
    VALID = 0
    NO_RECIPIENTS = 1
    INVALID_DATA = 2
    MESSAGE_TOO_LARGE = 3
    TOO_MANY_RECIPIENTS = 4
    TRANSPORT_INACTIVE = 5
    TRANSPORT_NOT_FOUND = 6
    TOO_MANY_ATTACHMENTS = 7
    INVALID_RECIPIENTS = 8
    INVALID_BODY = 9
    INVALID_OTHER = 10
    VALID_WITH_LARGE_MESSAGE = 11
    VOICE_ROAMING_RESTRICTION = 12
    DATA_ROAMING_RESTRICTION = 13

class ChatRestoreHistorySpan(enum.IntEnum):
    LAST_MONTH = 0
    LAST_YEAR = 1
    ANY_TIME = 2

class ChatStoreChangedEventKind(enum.IntEnum):
    NOTIFICATIONS_MISSED = 0
    STORE_MODIFIED = 1
    MESSAGE_CREATED = 2
    MESSAGE_MODIFIED = 3
    MESSAGE_DELETED = 4
    CONVERSATION_MODIFIED = 5
    CONVERSATION_DELETED = 6
    CONVERSATION_TRANSPORT_DELETED = 7

class ChatTransportErrorCodeCategory(enum.IntEnum):
    NONE = 0
    HTTP = 1
    NETWORK = 2
    MMS_SERVER = 3

class ChatTransportInterpretedErrorCode(enum.IntEnum):
    NONE = 0
    UNKNOWN = 1
    INVALID_RECIPIENT_ADDRESS = 2
    NETWORK_CONNECTIVITY = 3
    SERVICE_DENIED = 4
    TIMEOUT = 5

class RcsServiceKind(enum.IntEnum):
    CHAT = 0
    GROUP_CHAT = 1
    FILE_TRANSFER = 2
    CAPABILITY = 3

ChatCapabilities = _ns_module.ChatCapabilities
ChatCapabilitiesManager = _ns_module.ChatCapabilitiesManager
ChatConversation = _ns_module.ChatConversation
ChatConversationReader = _ns_module.ChatConversationReader
ChatConversationThreadingInfo = _ns_module.ChatConversationThreadingInfo
ChatMessage = _ns_module.ChatMessage
ChatMessageAttachment = _ns_module.ChatMessageAttachment
ChatMessageBlocking = _ns_module.ChatMessageBlocking
ChatMessageChange = _ns_module.ChatMessageChange
ChatMessageChangeReader = _ns_module.ChatMessageChangeReader
ChatMessageChangeTracker = _ns_module.ChatMessageChangeTracker
ChatMessageChangedDeferral = _ns_module.ChatMessageChangedDeferral
ChatMessageChangedEventArgs = _ns_module.ChatMessageChangedEventArgs
ChatMessageManager = _ns_module.ChatMessageManager
ChatMessageNotificationTriggerDetails = _ns_module.ChatMessageNotificationTriggerDetails
ChatMessageReader = _ns_module.ChatMessageReader
ChatMessageStore = _ns_module.ChatMessageStore
ChatMessageStoreChangedEventArgs = _ns_module.ChatMessageStoreChangedEventArgs
ChatMessageTransport = _ns_module.ChatMessageTransport
ChatMessageTransportConfiguration = _ns_module.ChatMessageTransportConfiguration
ChatMessageValidationResult = _ns_module.ChatMessageValidationResult
ChatQueryOptions = _ns_module.ChatQueryOptions
ChatRecipientDeliveryInfo = _ns_module.ChatRecipientDeliveryInfo
ChatSearchReader = _ns_module.ChatSearchReader
ChatSyncConfiguration = _ns_module.ChatSyncConfiguration
ChatSyncManager = _ns_module.ChatSyncManager
RcsEndUserMessage = _ns_module.RcsEndUserMessage
RcsEndUserMessageAction = _ns_module.RcsEndUserMessageAction
RcsEndUserMessageAvailableEventArgs = _ns_module.RcsEndUserMessageAvailableEventArgs
RcsEndUserMessageAvailableTriggerDetails = _ns_module.RcsEndUserMessageAvailableTriggerDetails
RcsEndUserMessageManager = _ns_module.RcsEndUserMessageManager
RcsManager = _ns_module.RcsManager
RcsServiceKindSupportedChangedEventArgs = _ns_module.RcsServiceKindSupportedChangedEventArgs
RcsTransport = _ns_module.RcsTransport
RcsTransportConfiguration = _ns_module.RcsTransportConfiguration
RemoteParticipantComposingChangedEventArgs = _ns_module.RemoteParticipantComposingChangedEventArgs
IChatItem = _ns_module.IChatItem