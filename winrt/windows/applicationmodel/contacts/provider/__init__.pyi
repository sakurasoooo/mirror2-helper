# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.applicationmodel.contacts
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

class AddContactResult(enum.IntEnum):
    ADDED = 0
    ALREADY_ADDED = 1
    UNAVAILABLE = 2

class ContactPickerUI(_winrt.winrt_base):
    ...
    desired_fields: winrt.windows.foundation.collections.IVectorView[str]
    selection_mode: winrt.windows.applicationmodel.contacts.ContactSelectionMode
    desired_fields_with_contact_field_type: winrt.windows.foundation.collections.IVector[winrt.windows.applicationmodel.contacts.ContactFieldType]
    def add_contact(contact: winrt.windows.applicationmodel.contacts.Contact) -> AddContactResult:
        ...
    def add_contact(id: str, contact: winrt.windows.applicationmodel.contacts.Contact) -> AddContactResult:
        ...
    def contains_contact(id: str) -> bool:
        ...
    def remove_contact(id: str) -> None:
        ...
    def add_contact_removed(handler: winrt.windows.foundation.TypedEventHandler[ContactPickerUI, ContactRemovedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_contact_removed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ContactRemovedEventArgs(_winrt.winrt_base):
    ...
    id: str

