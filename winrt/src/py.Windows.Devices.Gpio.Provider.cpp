// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.Devices.Gpio.Provider.h"

PyTypeObject* py::winrt_type<winrt::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioControllerProvider>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioPinProvider>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioProvider>::python_type;

namespace py::cpp::Windows::Devices::Gpio::Provider
{
    // ----- GpioPinProviderValueChangedEventArgs class --------------------
    constexpr const char* const _type_name_GpioPinProviderValueChangedEventArgs = "GpioPinProviderValueChangedEventArgs";

    static PyObject* _new_GpioPinProviderValueChangedEventArgs(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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
                auto param0 = py::convert_to<winrt::Windows::Devices::Gpio::Provider::ProviderGpioPinEdge>(args, 0);

                winrt::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs instance{ param0 };
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

    static void _dealloc_GpioPinProviderValueChangedEventArgs(py::wrapper::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* GpioPinProviderValueChangedEventArgs_get_Edge(py::wrapper::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Edge());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_GpioPinProviderValueChangedEventArgs(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_GpioPinProviderValueChangedEventArgs[] = {
        { "_from", (PyCFunction)_from_GpioPinProviderValueChangedEventArgs, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_GpioPinProviderValueChangedEventArgs[] = {
        { const_cast<char*>("edge"), (getter)GpioPinProviderValueChangedEventArgs_get_Edge, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_GpioPinProviderValueChangedEventArgs[] = 
    {
        { Py_tp_new, _new_GpioPinProviderValueChangedEventArgs },
        { Py_tp_dealloc, _dealloc_GpioPinProviderValueChangedEventArgs },
        { Py_tp_methods, _methods_GpioPinProviderValueChangedEventArgs },
        { Py_tp_getset, _getset_GpioPinProviderValueChangedEventArgs },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_GpioPinProviderValueChangedEventArgs =
    {
        "_winrt_Windows_Devices_Gpio_Provider.GpioPinProviderValueChangedEventArgs",
        sizeof(py::wrapper::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_GpioPinProviderValueChangedEventArgs
    };

    // ----- IGpioControllerProvider interface --------------------
    constexpr const char* const _type_name_IGpioControllerProvider = "IGpioControllerProvider";

    static PyObject* _new_IGpioControllerProvider(PyTypeObject* /* unused */, PyObject* /* unused */, PyObject* /* unused */)
    {
        py::set_invalid_activation_error(_type_name_IGpioControllerProvider);
        return nullptr;
    }

    static void _dealloc_IGpioControllerProvider(py::wrapper::Windows::Devices::Gpio::Provider::IGpioControllerProvider* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* IGpioControllerProvider_OpenPinProvider(py::wrapper::Windows::Devices::Gpio::Provider::IGpioControllerProvider* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<int32_t>(args, 0);
                auto param1 = py::convert_to<winrt::Windows::Devices::Gpio::Provider::ProviderGpioSharingMode>(args, 1);

                return py::convert(self->obj.OpenPinProvider(param0, param1));
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

    static PyObject* IGpioControllerProvider_get_PinCount(py::wrapper::Windows::Devices::Gpio::Provider::IGpioControllerProvider* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.PinCount());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_IGpioControllerProvider(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Gpio::Provider::IGpioControllerProvider>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_IGpioControllerProvider[] = {
        { "open_pin_provider", (PyCFunction)IGpioControllerProvider_OpenPinProvider, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_IGpioControllerProvider, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_IGpioControllerProvider[] = {
        { const_cast<char*>("pin_count"), (getter)IGpioControllerProvider_get_PinCount, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_IGpioControllerProvider[] = 
    {
        { Py_tp_new, _new_IGpioControllerProvider },
        { Py_tp_dealloc, _dealloc_IGpioControllerProvider },
        { Py_tp_methods, _methods_IGpioControllerProvider },
        { Py_tp_getset, _getset_IGpioControllerProvider },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_IGpioControllerProvider =
    {
        "_winrt_Windows_Devices_Gpio_Provider.IGpioControllerProvider",
        sizeof(py::wrapper::Windows::Devices::Gpio::Provider::IGpioControllerProvider),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_IGpioControllerProvider
    };

    // ----- IGpioPinProvider interface --------------------
    constexpr const char* const _type_name_IGpioPinProvider = "IGpioPinProvider";

    static PyObject* _new_IGpioPinProvider(PyTypeObject* /* unused */, PyObject* /* unused */, PyObject* /* unused */)
    {
        py::set_invalid_activation_error(_type_name_IGpioPinProvider);
        return nullptr;
    }

    static void _dealloc_IGpioPinProvider(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* IGpioPinProvider_GetDriveMode(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetDriveMode());
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

    static PyObject* IGpioPinProvider_IsDriveModeSupported(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Devices::Gpio::Provider::ProviderGpioPinDriveMode>(args, 0);

                return py::convert(self->obj.IsDriveModeSupported(param0));
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

    static PyObject* IGpioPinProvider_Read(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.Read());
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

    static PyObject* IGpioPinProvider_SetDriveMode(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Devices::Gpio::Provider::ProviderGpioPinDriveMode>(args, 0);

                self->obj.SetDriveMode(param0);
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

    static PyObject* IGpioPinProvider_Write(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Devices::Gpio::Provider::ProviderGpioPinValue>(args, 0);

                self->obj.Write(param0);
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

    static PyObject* IGpioPinProvider_get_DebounceTimeout(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.DebounceTimeout());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int IGpioPinProvider_put_DebounceTimeout(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::TimeSpan>(arg);

            self->obj.DebounceTimeout(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* IGpioPinProvider_get_PinNumber(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.PinNumber());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* IGpioPinProvider_get_SharingMode(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SharingMode());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* IGpioPinProvider_add_ValueChanged(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::TypedEventHandler<winrt::Windows::Devices::Gpio::Provider::IGpioPinProvider, winrt::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs>>(arg);

            return py::convert(self->obj.ValueChanged(param0));
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* IGpioPinProvider_remove_ValueChanged(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::event_token>(arg);

            self->obj.ValueChanged(param0);
            Py_RETURN_NONE;
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_IGpioPinProvider(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Gpio::Provider::IGpioPinProvider>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_IGpioPinProvider[] = {
        { "get_drive_mode", (PyCFunction)IGpioPinProvider_GetDriveMode, METH_VARARGS, nullptr },
        { "is_drive_mode_supported", (PyCFunction)IGpioPinProvider_IsDriveModeSupported, METH_VARARGS, nullptr },
        { "read", (PyCFunction)IGpioPinProvider_Read, METH_VARARGS, nullptr },
        { "set_drive_mode", (PyCFunction)IGpioPinProvider_SetDriveMode, METH_VARARGS, nullptr },
        { "write", (PyCFunction)IGpioPinProvider_Write, METH_VARARGS, nullptr },
        { "add_value_changed", (PyCFunction)IGpioPinProvider_add_ValueChanged, METH_O, nullptr },
        { "remove_value_changed", (PyCFunction)IGpioPinProvider_remove_ValueChanged, METH_O, nullptr },
        { "_from", (PyCFunction)_from_IGpioPinProvider, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_IGpioPinProvider[] = {
        { const_cast<char*>("debounce_timeout"), (getter)IGpioPinProvider_get_DebounceTimeout, (setter)IGpioPinProvider_put_DebounceTimeout, nullptr, nullptr },
        { const_cast<char*>("pin_number"), (getter)IGpioPinProvider_get_PinNumber, nullptr, nullptr, nullptr },
        { const_cast<char*>("sharing_mode"), (getter)IGpioPinProvider_get_SharingMode, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_IGpioPinProvider[] = 
    {
        { Py_tp_new, _new_IGpioPinProvider },
        { Py_tp_dealloc, _dealloc_IGpioPinProvider },
        { Py_tp_methods, _methods_IGpioPinProvider },
        { Py_tp_getset, _getset_IGpioPinProvider },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_IGpioPinProvider =
    {
        "_winrt_Windows_Devices_Gpio_Provider.IGpioPinProvider",
        sizeof(py::wrapper::Windows::Devices::Gpio::Provider::IGpioPinProvider),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_IGpioPinProvider
    };

    // ----- IGpioProvider interface --------------------
    constexpr const char* const _type_name_IGpioProvider = "IGpioProvider";

    static PyObject* _new_IGpioProvider(PyTypeObject* /* unused */, PyObject* /* unused */, PyObject* /* unused */)
    {
        py::set_invalid_activation_error(_type_name_IGpioProvider);
        return nullptr;
    }

    static void _dealloc_IGpioProvider(py::wrapper::Windows::Devices::Gpio::Provider::IGpioProvider* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* IGpioProvider_GetControllers(py::wrapper::Windows::Devices::Gpio::Provider::IGpioProvider* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetControllers());
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

    static PyObject* _from_IGpioProvider(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Gpio::Provider::IGpioProvider>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_IGpioProvider[] = {
        { "get_controllers", (PyCFunction)IGpioProvider_GetControllers, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_IGpioProvider, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_IGpioProvider[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_IGpioProvider[] = 
    {
        { Py_tp_new, _new_IGpioProvider },
        { Py_tp_dealloc, _dealloc_IGpioProvider },
        { Py_tp_methods, _methods_IGpioProvider },
        { Py_tp_getset, _getset_IGpioProvider },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_IGpioProvider =
    {
        "_winrt_Windows_Devices_Gpio_Provider.IGpioProvider",
        sizeof(py::wrapper::Windows::Devices::Gpio::Provider::IGpioProvider),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_IGpioProvider
    };

    // ----- Windows.Devices.Gpio.Provider Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs>::python_type = py::register_python_type(module, _type_name_GpioPinProviderValueChangedEventArgs, &_type_spec_GpioPinProviderValueChangedEventArgs, bases.get());
            py::winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioControllerProvider>::python_type = py::register_python_type(module, _type_name_IGpioControllerProvider, &_type_spec_IGpioControllerProvider, bases.get());
            py::winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioPinProvider>::python_type = py::register_python_type(module, _type_name_IGpioPinProvider, &_type_spec_IGpioPinProvider, bases.get());
            py::winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioProvider>::python_type = py::register_python_type(module, _type_name_IGpioProvider, &_type_spec_IGpioProvider, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.Devices.Gpio.Provider");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_Devices_Gpio_Provider",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::Devices::Gpio::Provider

PyMODINIT_FUNC
PyInit__winrt_Windows_Devices_Gpio_Provider (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::Devices::Gpio::Provider::module_def);
}