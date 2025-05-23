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
    import winrt.windows.ui.popups
except Exception:
    pass

class AppointmentBusyStatus(enum.IntEnum):
    BUSY = 0
    TENTATIVE = 1
    FREE = 2
    OUT_OF_OFFICE = 3
    WORKING_ELSEWHERE = 4

class AppointmentCalendarOtherAppReadAccess(enum.IntEnum):
    SYSTEM_ONLY = 0
    LIMITED = 1
    FULL = 2
    NONE = 3

class AppointmentCalendarOtherAppWriteAccess(enum.IntEnum):
    NONE = 0
    SYSTEM_ONLY = 1
    LIMITED = 2

class AppointmentCalendarSyncStatus(enum.IntEnum):
    IDLE = 0
    SYNCING = 1
    UP_TO_DATE = 2
    AUTHENTICATION_ERROR = 3
    POLICY_ERROR = 4
    UNKNOWN_ERROR = 5
    MANUAL_ACCOUNT_REMOVAL_REQUIRED = 6

class AppointmentConflictType(enum.IntEnum):
    NONE = 0
    ADJACENT = 1
    OVERLAP = 2

class AppointmentDaysOfWeek(enum.IntFlag):
    NONE = 0
    SUNDAY = 0x1
    MONDAY = 0x2
    TUESDAY = 0x4
    WEDNESDAY = 0x8
    THURSDAY = 0x10
    FRIDAY = 0x20
    SATURDAY = 0x40

class AppointmentDetailsKind(enum.IntEnum):
    PLAIN_TEXT = 0
    HTML = 1

class AppointmentParticipantResponse(enum.IntEnum):
    NONE = 0
    TENTATIVE = 1
    ACCEPTED = 2
    DECLINED = 3
    UNKNOWN = 4

class AppointmentParticipantRole(enum.IntEnum):
    REQUIRED_ATTENDEE = 0
    OPTIONAL_ATTENDEE = 1
    RESOURCE = 2

class AppointmentRecurrenceUnit(enum.IntEnum):
    DAILY = 0
    WEEKLY = 1
    MONTHLY = 2
    MONTHLY_ON_DAY = 3
    YEARLY = 4
    YEARLY_ON_DAY = 5

class AppointmentSensitivity(enum.IntEnum):
    PUBLIC = 0
    PRIVATE = 1

class AppointmentStoreAccessType(enum.IntEnum):
    APP_CALENDARS_READ_WRITE = 0
    ALL_CALENDARS_READ_ONLY = 1
    ALL_CALENDARS_READ_WRITE = 2

class AppointmentStoreChangeType(enum.IntEnum):
    APPOINTMENT_CREATED = 0
    APPOINTMENT_MODIFIED = 1
    APPOINTMENT_DELETED = 2
    CHANGE_TRACKING_LOST = 3
    CALENDAR_CREATED = 4
    CALENDAR_MODIFIED = 5
    CALENDAR_DELETED = 6

class AppointmentSummaryCardView(enum.IntEnum):
    SYSTEM = 0
    APP = 1

class AppointmentWeekOfMonth(enum.IntEnum):
    FIRST = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    LAST = 4

class FindAppointmentCalendarsOptions(enum.IntFlag):
    NONE = 0
    INCLUDE_HIDDEN = 0x1

class RecurrenceType(enum.IntEnum):
    MASTER = 0
    INSTANCE = 1
    EXCEPTION_INSTANCE = 2

class Appointment(_winrt.winrt_base):
    ...
    location: str
    all_day: bool
    organizer: AppointmentOrganizer
    duration: winrt.windows.foundation.TimeSpan
    details: str
    busy_status: AppointmentBusyStatus
    recurrence: AppointmentRecurrence
    subject: str
    uri: winrt.windows.foundation.Uri
    start_time: winrt.windows.foundation.DateTime
    sensitivity: AppointmentSensitivity
    reminder: typing.Optional[winrt.windows.foundation.TimeSpan]
    invitees: winrt.windows.foundation.collections.IVector[AppointmentInvitee]
    allow_new_time_proposal: bool
    user_response: AppointmentParticipantResponse
    roaming_id: str
    reply_time: typing.Optional[winrt.windows.foundation.DateTime]
    is_response_requested: bool
    is_organized_by_user: bool
    is_canceled_meeting: bool
    online_meeting_link: str
    has_invitees: bool
    calendar_id: str
    local_id: str
    original_start_time: typing.Optional[winrt.windows.foundation.DateTime]
    remote_change_number: int
    details_kind: AppointmentDetailsKind
    change_number: int

class AppointmentCalendar(_winrt.winrt_base):
    ...
    summary_card_view: AppointmentSummaryCardView
    other_app_write_access: AppointmentCalendarOtherAppWriteAccess
    display_color: winrt.windows.ui.Color
    is_hidden: bool
    display_name: str
    other_app_read_access: AppointmentCalendarOtherAppReadAccess
    local_id: str
    source_display_name: str
    can_cancel_meetings: bool
    can_notify_invitees: bool
    remote_id: str
    must_nofity_invitees: bool
    can_update_meeting_responses: bool
    can_propose_new_time_for_meetings: bool
    can_create_or_update_appointments: bool
    can_forward_meetings: bool
    sync_manager: AppointmentCalendarSyncManager
    user_data_account_id: str
    def delete_appointment_async(local_id: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def delete_appointment_instance_async(local_id: str, instance_start_time: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncAction:
        ...
    def delete_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def find_all_instances_async(master_local_id: str, range_start: winrt.windows.foundation.DateTime, range_length: winrt.windows.foundation.TimeSpan) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Appointment]]:
        ...
    def find_all_instances_async(master_local_id: str, range_start: winrt.windows.foundation.DateTime, range_length: winrt.windows.foundation.TimeSpan, p_options: FindAppointmentsOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Appointment]]:
        ...
    def find_appointments_async(range_start: winrt.windows.foundation.DateTime, range_length: winrt.windows.foundation.TimeSpan) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Appointment]]:
        ...
    def find_appointments_async(range_start: winrt.windows.foundation.DateTime, range_length: winrt.windows.foundation.TimeSpan, options: FindAppointmentsOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Appointment]]:
        ...
    def find_exceptions_from_master_async(master_local_id: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[AppointmentException]]:
        ...
    def find_unexpanded_appointments_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Appointment]]:
        ...
    def find_unexpanded_appointments_async(options: FindAppointmentsOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Appointment]]:
        ...
    def get_appointment_async(local_id: str) -> winrt.windows.foundation.IAsyncOperation[Appointment]:
        ...
    def get_appointment_instance_async(local_id: str, instance_start_time: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[Appointment]:
        ...
    def register_sync_manager_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def save_appointment_async(p_appointment: Appointment) -> winrt.windows.foundation.IAsyncAction:
        ...
    def save_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def try_cancel_meeting_async(meeting: Appointment, subject: str, comment: str, notify_invitees: bool) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_create_or_update_appointment_async(appointment: Appointment, notify_invitees: bool) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_forward_meeting_async(meeting: Appointment, invitees: typing.Iterable[AppointmentInvitee], subject: str, forward_header: str, comment: str) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_propose_new_time_for_meeting_async(meeting: Appointment, new_start_time: winrt.windows.foundation.DateTime, new_duration: winrt.windows.foundation.TimeSpan, subject: str, comment: str) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_update_meeting_response_async(meeting: Appointment, response: AppointmentParticipantResponse, subject: str, comment: str, send_update: bool) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...

class AppointmentCalendarSyncManager(_winrt.winrt_base):
    ...
    status: AppointmentCalendarSyncStatus
    last_successful_sync_time: winrt.windows.foundation.DateTime
    last_attempted_sync_time: winrt.windows.foundation.DateTime
    def sync_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def add_sync_status_changed(handler: winrt.windows.foundation.TypedEventHandler[AppointmentCalendarSyncManager, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_sync_status_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class AppointmentConflictResult(_winrt.winrt_base):
    ...
    date: winrt.windows.foundation.DateTime
    type: AppointmentConflictType

class AppointmentException(_winrt.winrt_base):
    ...
    appointment: Appointment
    exception_properties: winrt.windows.foundation.collections.IVectorView[str]
    is_deleted: bool

class AppointmentInvitee(IAppointmentParticipant, _winrt.winrt_base):
    ...
    role: AppointmentParticipantRole
    response: AppointmentParticipantResponse
    display_name: str
    address: str

class AppointmentManager(_winrt.winrt_base):
    ...
    def get_for_user(user: winrt.windows.system.User) -> AppointmentManagerForUser:
        ...
    def request_store_async(options: AppointmentStoreAccessType) -> winrt.windows.foundation.IAsyncOperation[AppointmentStore]:
        ...
    def show_add_appointment_async(appointment: Appointment, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_add_appointment_async(appointment: Appointment, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_appointment_details_async(appointment_id: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def show_appointment_details_async(appointment_id: str, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncAction:
        ...
    def show_edit_new_appointment_async(appointment: Appointment) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_remove_appointment_async(appointment_id: str, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def show_remove_appointment_async(appointment_id: str, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def show_remove_appointment_async(appointment_id: str, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def show_replace_appointment_async(appointment_id: str, appointment: Appointment, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_replace_appointment_async(appointment_id: str, appointment: Appointment, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_replace_appointment_async(appointment_id: str, appointment: Appointment, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_time_frame_async(time_to_show: winrt.windows.foundation.DateTime, duration: winrt.windows.foundation.TimeSpan) -> winrt.windows.foundation.IAsyncAction:
        ...

class AppointmentManagerForUser(_winrt.winrt_base):
    ...
    user: winrt.windows.system.User
    def request_store_async(options: AppointmentStoreAccessType) -> winrt.windows.foundation.IAsyncOperation[AppointmentStore]:
        ...
    def show_add_appointment_async(appointment: Appointment, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_add_appointment_async(appointment: Appointment, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_appointment_details_async(appointment_id: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def show_appointment_details_async(appointment_id: str, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncAction:
        ...
    def show_edit_new_appointment_async(appointment: Appointment) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_remove_appointment_async(appointment_id: str, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def show_remove_appointment_async(appointment_id: str, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def show_remove_appointment_async(appointment_id: str, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def show_replace_appointment_async(appointment_id: str, appointment: Appointment, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_replace_appointment_async(appointment_id: str, appointment: Appointment, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_replace_appointment_async(appointment_id: str, appointment: Appointment, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_time_frame_async(time_to_show: winrt.windows.foundation.DateTime, duration: winrt.windows.foundation.TimeSpan) -> winrt.windows.foundation.IAsyncAction:
        ...

class AppointmentOrganizer(IAppointmentParticipant, _winrt.winrt_base):
    ...
    display_name: str
    address: str

class AppointmentProperties(_winrt.winrt_base):
    ...
    has_invitees: str
    all_day: str
    allow_new_time_proposal: str
    busy_status: str
    default_properties: winrt.windows.foundation.collections.IVector[str]
    details: str
    duration: str
    recurrence: str
    invitees: str
    is_canceled_meeting: str
    is_organized_by_user: str
    is_response_requested: str
    location: str
    online_meeting_link: str
    organizer: str
    original_start_time: str
    reminder: str
    reply_time: str
    sensitivity: str
    start_time: str
    subject: str
    uri: str
    user_response: str
    details_kind: str
    remote_change_number: str
    change_number: str

class AppointmentRecurrence(_winrt.winrt_base):
    ...
    unit: AppointmentRecurrenceUnit
    occurrences: typing.Optional[int]
    month: int
    interval: int
    days_of_week: AppointmentDaysOfWeek
    day: int
    week_of_month: AppointmentWeekOfMonth
    until: typing.Optional[winrt.windows.foundation.DateTime]
    time_zone: str
    recurrence_type: RecurrenceType
    calendar_identifier: str

class AppointmentStore(_winrt.winrt_base):
    ...
    change_tracker: AppointmentStoreChangeTracker
    def create_appointment_calendar_async(name: str) -> winrt.windows.foundation.IAsyncOperation[AppointmentCalendar]:
        ...
    def create_appointment_calendar_async(name: str, user_data_account_id: str) -> winrt.windows.foundation.IAsyncOperation[AppointmentCalendar]:
        ...
    def find_appointment_calendars_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[AppointmentCalendar]]:
        ...
    def find_appointment_calendars_async(options: FindAppointmentCalendarsOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[AppointmentCalendar]]:
        ...
    def find_appointments_async(range_start: winrt.windows.foundation.DateTime, range_length: winrt.windows.foundation.TimeSpan) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Appointment]]:
        ...
    def find_appointments_async(range_start: winrt.windows.foundation.DateTime, range_length: winrt.windows.foundation.TimeSpan, options: FindAppointmentsOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[Appointment]]:
        ...
    def find_conflict_async(appointment: Appointment) -> winrt.windows.foundation.IAsyncOperation[AppointmentConflictResult]:
        ...
    def find_conflict_async(appointment: Appointment, instance_start_time: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[AppointmentConflictResult]:
        ...
    def find_local_ids_from_roaming_id_async(roaming_id: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[str]]:
        ...
    def get_appointment_async(local_id: str) -> winrt.windows.foundation.IAsyncOperation[Appointment]:
        ...
    def get_appointment_calendar_async(calendar_id: str) -> winrt.windows.foundation.IAsyncOperation[AppointmentCalendar]:
        ...
    def get_appointment_instance_async(local_id: str, instance_start_time: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[Appointment]:
        ...
    def get_change_tracker(identity: str) -> AppointmentStoreChangeTracker:
        ...
    def move_appointment_async(appointment: Appointment, destination_calendar: AppointmentCalendar) -> winrt.windows.foundation.IAsyncAction:
        ...
    def show_add_appointment_async(appointment: Appointment, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_appointment_details_async(local_id: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def show_appointment_details_async(local_id: str, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncAction:
        ...
    def show_edit_new_appointment_async(appointment: Appointment) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_remove_appointment_async(local_id: str, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def show_remove_appointment_async(local_id: str, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def show_replace_appointment_async(local_id: str, appointment: Appointment, selection: winrt.windows.foundation.Rect) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def show_replace_appointment_async(local_id: str, appointment: Appointment, selection: winrt.windows.foundation.Rect, preferred_placement: winrt.windows.ui.popups.Placement, instance_start_date: winrt.windows.foundation.DateTime) -> winrt.windows.foundation.IAsyncOperation[str]:
        ...
    def add_store_changed(p_handler: winrt.windows.foundation.TypedEventHandler[AppointmentStore, AppointmentStoreChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_store_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class AppointmentStoreChange(_winrt.winrt_base):
    ...
    appointment: Appointment
    change_type: AppointmentStoreChangeType
    appointment_calendar: AppointmentCalendar

class AppointmentStoreChangeReader(_winrt.winrt_base):
    ...
    def accept_changes() -> None:
        ...
    def accept_changes_through(last_change_to_accept: AppointmentStoreChange) -> None:
        ...
    def read_batch_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[AppointmentStoreChange]]:
        ...

class AppointmentStoreChangeTracker(_winrt.winrt_base):
    ...
    is_tracking: bool
    def enable() -> None:
        ...
    def get_change_reader() -> AppointmentStoreChangeReader:
        ...
    def reset() -> None:
        ...

class AppointmentStoreChangedDeferral(_winrt.winrt_base):
    ...
    def complete() -> None:
        ...

class AppointmentStoreChangedEventArgs(_winrt.winrt_base):
    ...
    def get_deferral() -> AppointmentStoreChangedDeferral:
        ...

class AppointmentStoreNotificationTriggerDetails(_winrt.winrt_base):
    ...

class FindAppointmentsOptions(_winrt.winrt_base):
    ...
    max_count: int
    include_hidden: bool
    calendar_ids: winrt.windows.foundation.collections.IVector[str]
    fetch_properties: winrt.windows.foundation.collections.IVector[str]

class IAppointmentParticipant(_winrt.winrt_base):
    ...
    address: str
    display_name: str

