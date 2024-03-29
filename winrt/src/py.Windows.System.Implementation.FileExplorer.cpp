// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.System.Implementation.FileExplorer.h"

PyTypeObject* py::winrt_type<winrt::Windows::System::Implementation::FileExplorer::SysStorageProviderEventReceivedEventArgs>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderEventSource>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderHandlerFactory>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderHttpRequestProvider>::python_type;

namespace py::cpp::Windows::System::Implementation::FileExplorer
{
    // ----- SysStorageProviderEventReceivedEventArgs class --------------------
    constexpr const char* const _type_name_SysStorageProviderEventReceivedEventArgs = "SysStorageProviderEventReceivedEventArgs";

    static PyObject* _new_SysStorageProviderEventReceivedEventArgs(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        if (kwds != nullptr)
        {
            py::set_invalid_kwd_args_error();
            return nullptr;
        }

        Py_ssize_t arg_count = PyTuple_Size(args);
        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);

                winrt::Windows::System::Implementation::FileExplorer::SysStorageProviderEventReceivedEventArgs instance{ param0 };
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

    static void _dealloc_SysStorageProviderEventReceivedEventArgs(py::wrapper::Windows::System::Implementation::FileExplorer::SysStorageProviderEventReceivedEventArgs* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SysStorageProviderEventReceivedEventArgs_get_Json(py::wrapper::Windows::System::Implementation::FileExplorer::SysStorageProviderEventReceivedEventArgs* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Json());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SysStorageProviderEventReceivedEventArgs(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Implementation::FileExplorer::SysStorageProviderEventReceivedEventArgs>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SysStorageProviderEventReceivedEventArgs[] = {
        { "_from", (PyCFunction)_from_SysStorageProviderEventReceivedEventArgs, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SysStorageProviderEventReceivedEventArgs[] = {
        { const_cast<char*>("json"), (getter)SysStorageProviderEventReceivedEventArgs_get_Json, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SysStorageProviderEventReceivedEventArgs[] = 
    {
        { Py_tp_new, _new_SysStorageProviderEventReceivedEventArgs },
        { Py_tp_dealloc, _dealloc_SysStorageProviderEventReceivedEventArgs },
        { Py_tp_methods, _methods_SysStorageProviderEventReceivedEventArgs },
        { Py_tp_getset, _getset_SysStorageProviderEventReceivedEventArgs },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SysStorageProviderEventReceivedEventArgs =
    {
        "_winrt_Windows_System_Implementation_FileExplorer.SysStorageProviderEventReceivedEventArgs",
        sizeof(py::wrapper::Windows::System::Implementation::FileExplorer::SysStorageProviderEventReceivedEventArgs),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SysStorageProviderEventReceivedEventArgs
    };

    // ----- ISysStorageProviderEventSource interface --------------------
    constexpr const char* const _type_name_ISysStorageProviderEventSource = "ISysStorageProviderEventSource";

    static PyObject* _new_ISysStorageProviderEventSource(PyTypeObject* /* unused */, PyObject* /* unused */, PyObject* /* unused */)
    {
        py::set_invalid_activation_error(_type_name_ISysStorageProviderEventSource);
        return nullptr;
    }

    static void _dealloc_ISysStorageProviderEventSource(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderEventSource* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* ISysStorageProviderEventSource_add_EventReceived(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderEventSource* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::TypedEventHandler<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderEventSource, winrt::Windows::System::Implementation::FileExplorer::SysStorageProviderEventReceivedEventArgs>>(arg);

            return py::convert(self->obj.EventReceived(param0));
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* ISysStorageProviderEventSource_remove_EventReceived(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderEventSource* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::event_token>(arg);

            self->obj.EventReceived(param0);
            Py_RETURN_NONE;
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_ISysStorageProviderEventSource(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderEventSource>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_ISysStorageProviderEventSource[] = {
        { "add_event_received", (PyCFunction)ISysStorageProviderEventSource_add_EventReceived, METH_O, nullptr },
        { "remove_event_received", (PyCFunction)ISysStorageProviderEventSource_remove_EventReceived, METH_O, nullptr },
        { "_from", (PyCFunction)_from_ISysStorageProviderEventSource, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_ISysStorageProviderEventSource[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_ISysStorageProviderEventSource[] = 
    {
        { Py_tp_new, _new_ISysStorageProviderEventSource },
        { Py_tp_dealloc, _dealloc_ISysStorageProviderEventSource },
        { Py_tp_methods, _methods_ISysStorageProviderEventSource },
        { Py_tp_getset, _getset_ISysStorageProviderEventSource },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_ISysStorageProviderEventSource =
    {
        "_winrt_Windows_System_Implementation_FileExplorer.ISysStorageProviderEventSource",
        sizeof(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderEventSource),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_ISysStorageProviderEventSource
    };

    // ----- ISysStorageProviderHandlerFactory interface --------------------
    constexpr const char* const _type_name_ISysStorageProviderHandlerFactory = "ISysStorageProviderHandlerFactory";

    static PyObject* _new_ISysStorageProviderHandlerFactory(PyTypeObject* /* unused */, PyObject* /* unused */, PyObject* /* unused */)
    {
        py::set_invalid_activation_error(_type_name_ISysStorageProviderHandlerFactory);
        return nullptr;
    }

    static void _dealloc_ISysStorageProviderHandlerFactory(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderHandlerFactory* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* ISysStorageProviderHandlerFactory_GetEventSource(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderHandlerFactory* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::hstring>(args, 1);

                return py::convert(self->obj.GetEventSource(param0, param1));
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

    static PyObject* ISysStorageProviderHandlerFactory_GetHttpRequestProvider(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderHandlerFactory* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);

                return py::convert(self->obj.GetHttpRequestProvider(param0));
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

    static PyObject* _from_ISysStorageProviderHandlerFactory(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderHandlerFactory>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_ISysStorageProviderHandlerFactory[] = {
        { "get_event_source", (PyCFunction)ISysStorageProviderHandlerFactory_GetEventSource, METH_VARARGS, nullptr },
        { "get_http_request_provider", (PyCFunction)ISysStorageProviderHandlerFactory_GetHttpRequestProvider, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_ISysStorageProviderHandlerFactory, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_ISysStorageProviderHandlerFactory[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_ISysStorageProviderHandlerFactory[] = 
    {
        { Py_tp_new, _new_ISysStorageProviderHandlerFactory },
        { Py_tp_dealloc, _dealloc_ISysStorageProviderHandlerFactory },
        { Py_tp_methods, _methods_ISysStorageProviderHandlerFactory },
        { Py_tp_getset, _getset_ISysStorageProviderHandlerFactory },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_ISysStorageProviderHandlerFactory =
    {
        "_winrt_Windows_System_Implementation_FileExplorer.ISysStorageProviderHandlerFactory",
        sizeof(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderHandlerFactory),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_ISysStorageProviderHandlerFactory
    };

    // ----- ISysStorageProviderHttpRequestProvider interface --------------------
    constexpr const char* const _type_name_ISysStorageProviderHttpRequestProvider = "ISysStorageProviderHttpRequestProvider";

    static PyObject* _new_ISysStorageProviderHttpRequestProvider(PyTypeObject* /* unused */, PyObject* /* unused */, PyObject* /* unused */)
    {
        py::set_invalid_activation_error(_type_name_ISysStorageProviderHttpRequestProvider);
        return nullptr;
    }

    static void _dealloc_ISysStorageProviderHttpRequestProvider(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderHttpRequestProvider* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* ISysStorageProviderHttpRequestProvider_SendRequestAsync(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderHttpRequestProvider* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Web::Http::HttpRequestMessage>(args, 0);

                return py::convert(self->obj.SendRequestAsync(param0));
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

    static PyObject* _from_ISysStorageProviderHttpRequestProvider(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderHttpRequestProvider>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_ISysStorageProviderHttpRequestProvider[] = {
        { "send_request_async", (PyCFunction)ISysStorageProviderHttpRequestProvider_SendRequestAsync, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_ISysStorageProviderHttpRequestProvider, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_ISysStorageProviderHttpRequestProvider[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_ISysStorageProviderHttpRequestProvider[] = 
    {
        { Py_tp_new, _new_ISysStorageProviderHttpRequestProvider },
        { Py_tp_dealloc, _dealloc_ISysStorageProviderHttpRequestProvider },
        { Py_tp_methods, _methods_ISysStorageProviderHttpRequestProvider },
        { Py_tp_getset, _getset_ISysStorageProviderHttpRequestProvider },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_ISysStorageProviderHttpRequestProvider =
    {
        "_winrt_Windows_System_Implementation_FileExplorer.ISysStorageProviderHttpRequestProvider",
        sizeof(py::wrapper::Windows::System::Implementation::FileExplorer::ISysStorageProviderHttpRequestProvider),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_ISysStorageProviderHttpRequestProvider
    };

    // ----- Windows.System.Implementation.FileExplorer Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::System::Implementation::FileExplorer::SysStorageProviderEventReceivedEventArgs>::python_type = py::register_python_type(module, _type_name_SysStorageProviderEventReceivedEventArgs, &_type_spec_SysStorageProviderEventReceivedEventArgs, bases.get());
            py::winrt_type<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderEventSource>::python_type = py::register_python_type(module, _type_name_ISysStorageProviderEventSource, &_type_spec_ISysStorageProviderEventSource, bases.get());
            py::winrt_type<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderHandlerFactory>::python_type = py::register_python_type(module, _type_name_ISysStorageProviderHandlerFactory, &_type_spec_ISysStorageProviderHandlerFactory, bases.get());
            py::winrt_type<winrt::Windows::System::Implementation::FileExplorer::ISysStorageProviderHttpRequestProvider>::python_type = py::register_python_type(module, _type_name_ISysStorageProviderHttpRequestProvider, &_type_spec_ISysStorageProviderHttpRequestProvider, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.System.Implementation.FileExplorer");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_System_Implementation_FileExplorer",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::System::Implementation::FileExplorer

PyMODINIT_FUNC
PyInit__winrt_Windows_System_Implementation_FileExplorer (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::System::Implementation::FileExplorer::module_def);
}
