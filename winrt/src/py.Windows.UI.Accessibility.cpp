// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.UI.Accessibility.h"

PyTypeObject* py::winrt_type<winrt::Windows::UI::Accessibility::ScreenReaderPositionChangedEventArgs>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::UI::Accessibility::ScreenReaderService>::python_type;

namespace py::cpp::Windows::UI::Accessibility
{
    // ----- ScreenReaderPositionChangedEventArgs class --------------------
    constexpr const char* const _type_name_ScreenReaderPositionChangedEventArgs = "ScreenReaderPositionChangedEventArgs";

    static PyObject* _new_ScreenReaderPositionChangedEventArgs(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_ScreenReaderPositionChangedEventArgs);
        return nullptr;
    }

    static void _dealloc_ScreenReaderPositionChangedEventArgs(py::wrapper::Windows::UI::Accessibility::ScreenReaderPositionChangedEventArgs* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* ScreenReaderPositionChangedEventArgs_get_IsReadingText(py::wrapper::Windows::UI::Accessibility::ScreenReaderPositionChangedEventArgs* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsReadingText());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* ScreenReaderPositionChangedEventArgs_get_ScreenPositionInRawPixels(py::wrapper::Windows::UI::Accessibility::ScreenReaderPositionChangedEventArgs* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.ScreenPositionInRawPixels());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_ScreenReaderPositionChangedEventArgs(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::Accessibility::ScreenReaderPositionChangedEventArgs>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_ScreenReaderPositionChangedEventArgs[] = {
        { "_from", (PyCFunction)_from_ScreenReaderPositionChangedEventArgs, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_ScreenReaderPositionChangedEventArgs[] = {
        { const_cast<char*>("is_reading_text"), (getter)ScreenReaderPositionChangedEventArgs_get_IsReadingText, nullptr, nullptr, nullptr },
        { const_cast<char*>("screen_position_in_raw_pixels"), (getter)ScreenReaderPositionChangedEventArgs_get_ScreenPositionInRawPixels, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_ScreenReaderPositionChangedEventArgs[] = 
    {
        { Py_tp_new, _new_ScreenReaderPositionChangedEventArgs },
        { Py_tp_dealloc, _dealloc_ScreenReaderPositionChangedEventArgs },
        { Py_tp_methods, _methods_ScreenReaderPositionChangedEventArgs },
        { Py_tp_getset, _getset_ScreenReaderPositionChangedEventArgs },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_ScreenReaderPositionChangedEventArgs =
    {
        "_winrt_Windows_UI_Accessibility.ScreenReaderPositionChangedEventArgs",
        sizeof(py::wrapper::Windows::UI::Accessibility::ScreenReaderPositionChangedEventArgs),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_ScreenReaderPositionChangedEventArgs
    };

    // ----- ScreenReaderService class --------------------
    constexpr const char* const _type_name_ScreenReaderService = "ScreenReaderService";

    static PyObject* _new_ScreenReaderService(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        if (kwds != nullptr)
        {
            py::set_invalid_kwd_args_error();
            return nullptr;
        }

        Py_ssize_t arg_count = PyTuple_Size(args);
        if (arg_count == 0)
        {
            try
            {
                winrt::Windows::UI::Accessibility::ScreenReaderService instance{  };
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

    static void _dealloc_ScreenReaderService(py::wrapper::Windows::UI::Accessibility::ScreenReaderService* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* ScreenReaderService_get_CurrentScreenReaderPosition(py::wrapper::Windows::UI::Accessibility::ScreenReaderService* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.CurrentScreenReaderPosition());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* ScreenReaderService_add_ScreenReaderPositionChanged(py::wrapper::Windows::UI::Accessibility::ScreenReaderService* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::TypedEventHandler<winrt::Windows::UI::Accessibility::ScreenReaderService, winrt::Windows::UI::Accessibility::ScreenReaderPositionChangedEventArgs>>(arg);

            return py::convert(self->obj.ScreenReaderPositionChanged(param0));
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* ScreenReaderService_remove_ScreenReaderPositionChanged(py::wrapper::Windows::UI::Accessibility::ScreenReaderService* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::event_token>(arg);

            self->obj.ScreenReaderPositionChanged(param0);
            Py_RETURN_NONE;
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_ScreenReaderService(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::Accessibility::ScreenReaderService>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_ScreenReaderService[] = {
        { "add_screen_reader_position_changed", (PyCFunction)ScreenReaderService_add_ScreenReaderPositionChanged, METH_O, nullptr },
        { "remove_screen_reader_position_changed", (PyCFunction)ScreenReaderService_remove_ScreenReaderPositionChanged, METH_O, nullptr },
        { "_from", (PyCFunction)_from_ScreenReaderService, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_ScreenReaderService[] = {
        { const_cast<char*>("current_screen_reader_position"), (getter)ScreenReaderService_get_CurrentScreenReaderPosition, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_ScreenReaderService[] = 
    {
        { Py_tp_new, _new_ScreenReaderService },
        { Py_tp_dealloc, _dealloc_ScreenReaderService },
        { Py_tp_methods, _methods_ScreenReaderService },
        { Py_tp_getset, _getset_ScreenReaderService },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_ScreenReaderService =
    {
        "_winrt_Windows_UI_Accessibility.ScreenReaderService",
        sizeof(py::wrapper::Windows::UI::Accessibility::ScreenReaderService),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_ScreenReaderService
    };

    // ----- Windows.UI.Accessibility Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::UI::Accessibility::ScreenReaderPositionChangedEventArgs>::python_type = py::register_python_type(module, _type_name_ScreenReaderPositionChangedEventArgs, &_type_spec_ScreenReaderPositionChangedEventArgs, bases.get());
            py::winrt_type<winrt::Windows::UI::Accessibility::ScreenReaderService>::python_type = py::register_python_type(module, _type_name_ScreenReaderService, &_type_spec_ScreenReaderService, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.UI.Accessibility");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_UI_Accessibility",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::UI::Accessibility

PyMODINIT_FUNC
PyInit__winrt_Windows_UI_Accessibility (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::UI::Accessibility::module_def);
}