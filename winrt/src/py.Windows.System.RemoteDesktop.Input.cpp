// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.System.RemoteDesktop.Input.h"

PyTypeObject* py::winrt_type<winrt::Windows::System::RemoteDesktop::Input::RemoteTextConnection>::python_type;

namespace py::cpp::Windows::System::RemoteDesktop::Input
{
    // ----- RemoteTextConnection class --------------------
    constexpr const char* const _type_name_RemoteTextConnection = "RemoteTextConnection";

    static PyObject* _new_RemoteTextConnection(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        if (kwds != nullptr)
        {
            py::set_invalid_kwd_args_error();
            return nullptr;
        }

        Py_ssize_t arg_count = PyTuple_Size(args);
        if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<winrt::guid>(args, 0);
                auto param1 = py::convert_to<winrt::Windows::System::RemoteDesktop::Input::RemoteTextConnectionDataHandler>(args, 1);

                winrt::Windows::System::RemoteDesktop::Input::RemoteTextConnection instance{ param0, param1 };
                return py::wrap(instance, type);
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

    static void _dealloc_RemoteTextConnection(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* RemoteTextConnection_Close(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.Close();
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

    static PyObject* RemoteTextConnection_RegisterThread(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<uint32_t>(args, 0);

                self->obj.RegisterThread(param0);
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

    static PyObject* RemoteTextConnection_ReportDataReceived(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::array_view<uint8_t>>(args, 0);

                self->obj.ReportDataReceived(param0);
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

    static PyObject* RemoteTextConnection_UnregisterThread(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<uint32_t>(args, 0);

                self->obj.UnregisterThread(param0);
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

    static PyObject* RemoteTextConnection_get_IsEnabled(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsEnabled());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int RemoteTextConnection_put_IsEnabled(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<bool>(arg);

            self->obj.IsEnabled(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* _from_RemoteTextConnection(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::RemoteDesktop::Input::RemoteTextConnection>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _enter_RemoteTextConnection(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self) noexcept
    {
        Py_INCREF(self);
        return (PyObject*)self;
    }

    static PyObject* _exit_RemoteTextConnection(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection* self) noexcept
    {
        try
        {
            self->obj.Close();
            Py_RETURN_FALSE;
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_RemoteTextConnection[] = {
        { "close", (PyCFunction)RemoteTextConnection_Close, METH_VARARGS, nullptr },
        { "register_thread", (PyCFunction)RemoteTextConnection_RegisterThread, METH_VARARGS, nullptr },
        { "report_data_received", (PyCFunction)RemoteTextConnection_ReportDataReceived, METH_VARARGS, nullptr },
        { "unregister_thread", (PyCFunction)RemoteTextConnection_UnregisterThread, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_RemoteTextConnection, METH_O | METH_STATIC, nullptr },
        { "__enter__", (PyCFunction)_enter_RemoteTextConnection, METH_NOARGS, nullptr },
        { "__exit__",  (PyCFunction)_exit_RemoteTextConnection, METH_VARARGS, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_RemoteTextConnection[] = {
        { const_cast<char*>("is_enabled"), (getter)RemoteTextConnection_get_IsEnabled, (setter)RemoteTextConnection_put_IsEnabled, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_RemoteTextConnection[] = 
    {
        { Py_tp_new, _new_RemoteTextConnection },
        { Py_tp_dealloc, _dealloc_RemoteTextConnection },
        { Py_tp_methods, _methods_RemoteTextConnection },
        { Py_tp_getset, _getset_RemoteTextConnection },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_RemoteTextConnection =
    {
        "_winrt_Windows_System_RemoteDesktop_Input.RemoteTextConnection",
        sizeof(py::wrapper::Windows::System::RemoteDesktop::Input::RemoteTextConnection),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_RemoteTextConnection
    };

    // ----- Windows.System.RemoteDesktop.Input Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::System::RemoteDesktop::Input::RemoteTextConnection>::python_type = py::register_python_type(module, _type_name_RemoteTextConnection, &_type_spec_RemoteTextConnection, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.System.RemoteDesktop.Input");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_System_RemoteDesktop_Input",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::System::RemoteDesktop::Input

PyMODINIT_FUNC
PyInit__winrt_Windows_System_RemoteDesktop_Input (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::System::RemoteDesktop::Input::module_def);
}