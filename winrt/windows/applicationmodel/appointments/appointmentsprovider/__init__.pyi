# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.applicationmodel.appointments
except Exception:
    pass

try:
    import winrt.windows.foundation
except Exception:
    pass

class AddAppointmentOperation(_winrt.winrt_base):
    ...
    appointment_information: winrt.windows.applicationmodel.appointments.Appointment
    source_package_family_name: str
    def dismiss_u_i() -> None:
        ...
    def report_canceled() -> None:
        ...
    def report_completed(item_id: str) -> None:
        ...
    def report_error(value: str) -> None:
        ...

class AppointmentsProviderLaunchActionVerbs(_winrt.winrt_base):
    ...
    add_appointment: str
    remove_appointment: str
    replace_appointment: str
    show_time_frame: str
    show_appointment_details: str

class RemoveAppointmentOperation(_winrt.winrt_base):
    ...
    appointment_id: str
    instance_start_date: typing.Optional[winrt.windows.foundation.DateTime]
    source_package_family_name: str
    def dismiss_u_i() -> None:
        ...
    def report_canceled() -> None:
        ...
    def report_completed() -> None:
        ...
    def report_error(value: str) -> None:
        ...

class ReplaceAppointmentOperation(_winrt.winrt_base):
    ...
    appointment_id: str
    appointment_information: winrt.windows.applicationmodel.appointments.Appointment
    instance_start_date: typing.Optional[winrt.windows.foundation.DateTime]
    source_package_family_name: str
    def dismiss_u_i() -> None:
        ...
    def report_canceled() -> None:
        ...
    def report_completed(item_id: str) -> None:
        ...
    def report_error(value: str) -> None:
        ...

