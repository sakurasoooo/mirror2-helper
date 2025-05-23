// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.UI.Notifications.Management.h"

PyTypeObject* py::winrt_type<winrt::Windows::UI::Notifications::Management::UserNotificationListener>::python_type;

namespace py::cpp::Windows::UI::Notifications::Management
{
    // ----- UserNotificationListener class --------------------
    constexpr const char* const _type_name_UserNotificationListener = "UserNotificationListener";

    static PyObject* _new_UserNotificationListener(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_UserNotificationListener);
        return nullptr;
    }

    static void _dealloc_UserNotificationListener(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* UserNotificationListener_ClearNotifications(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.ClearNotifications();
                Py_RETURN_NONE;
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else
        {
            py::set_invalid_arg_count_error(arg_count);
            return nullptr;
        }
    }

    static PyObject* UserNotificationListener_GetAccessStatus(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetAccessStatus());
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else
        {
            py::set_invalid_arg_count_error(arg_count);
            return nullptr;
        }
    }

    static PyObject* UserNotificationListener_GetNotification(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<uint32_t>(args, 0);

                return py::convert(self->obj.GetNotification(param0));
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else
        {
            py::set_invalid_arg_count_error(arg_count);
            return nullptr;
        }
    }

    static PyObject* UserNotificationListener_GetNotificationsAsync(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::UI::Notifications::NotificationKinds>(args, 0);

                return py::convert(self->obj.GetNotificationsAsync(param0));
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else
        {
            py::set_invalid_arg_count_error(arg_count);
            return nullptr;
        }
    }

    static PyObject* UserNotificationListener_RemoveNotification(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<uint32_t>(args, 0);

                self->obj.RemoveNotification(param0);
                Py_RETURN_NONE;
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else
        {
            py::set_invalid_arg_count_error(arg_count);
            return nullptr;
        }
    }

    static PyObject* UserNotificationListener_RequestAccessAsync(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.RequestAccessAsync());
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else
        {
            py::set_invalid_arg_count_error(arg_count);
            return nullptr;
        }
    }

    static PyObject* UserNotificationListener_get_Current(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::UI::Notifications::Management::UserNotificationListener::Current());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* UserNotificationListener_add_NotificationChanged(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::TypedEventHandler<winrt::Windows::UI::Notifications::Management::UserNotificationListener, winrt::Windows::UI::Notifications::UserNotificationChangedEventArgs>>(arg);

            return py::convert(self->obj.NotificationChanged(param0));
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* UserNotificationListener_remove_NotificationChanged(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::event_token>(arg);

            self->obj.NotificationChanged(param0);
            Py_RETURN_NONE;
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_UserNotificationListener(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::Notifications::Management::UserNotificationListener>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_UserNotificationListener[] = {
        { "clear_notifications", (PyCFunction)UserNotificationListener_ClearNotifications, METH_VARARGS, nullptr },
        { "get_access_status", (PyCFunction)UserNotificationListener_GetAccessStatus, METH_VARARGS, nullptr },
        { "get_notification", (PyCFunction)UserNotificationListener_GetNotification, METH_VARARGS, nullptr },
        { "get_notifications_async", (PyCFunction)UserNotificationListener_GetNotificationsAsync, METH_VARARGS, nullptr },
        { "remove_notification", (PyCFunction)UserNotificationListener_RemoveNotification, METH_VARARGS, nullptr },
        { "request_access_async", (PyCFunction)UserNotificationListener_RequestAccessAsync, METH_VARARGS, nullptr },
        { "get_current", (PyCFunction)UserNotificationListener_get_Current, METH_NOARGS | METH_STATIC, nullptr },
        { "add_notification_changed", (PyCFunction)UserNotificationListener_add_NotificationChanged, METH_O, nullptr },
        { "remove_notification_changed", (PyCFunction)UserNotificationListener_remove_NotificationChanged, METH_O, nullptr },
        { "_from", (PyCFunction)_from_UserNotificationListener, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_UserNotificationListener[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_UserNotificationListener[] = 
    {
        { Py_tp_new, _new_UserNotificationListener },
        { Py_tp_dealloc, _dealloc_UserNotificationListener },
        { Py_tp_methods, _methods_UserNotificationListener },
        { Py_tp_getset, _getset_UserNotificationListener },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_UserNotificationListener =
    {
        "_winrt_Windows_UI_Notifications_Management.UserNotificationListener",
        sizeof(py::wrapper::Windows::UI::Notifications::Management::UserNotificationListener),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_UserNotificationListener
    };

    // ----- Windows.UI.Notifications.Management Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::UI::Notifications::Management::UserNotificationListener>::python_type = py::register_python_type(module, _type_name_UserNotificationListener, &_type_spec_UserNotificationListener, bases.get());

            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyModuleDef_Slot module_slots[] = {
        {Py_mod_exec, module_exec},
        {0, nullptr}
    };

    PyDoc_STRVAR(module_doc, "Windows.UI.Notifications.Management");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_UI_Notifications_Management",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::UI::Notifications::Management

PyMODINIT_FUNC
PyInit__winrt_Windows_UI_Notifications_Management (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::UI::Notifications::Management::module_def);
}
