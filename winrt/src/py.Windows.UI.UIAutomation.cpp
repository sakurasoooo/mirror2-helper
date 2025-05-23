// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.UI.UIAutomation.h"

PyTypeObject* py::winrt_type<winrt::Windows::UI::UIAutomation::AutomationConnection>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::UI::UIAutomation::AutomationConnectionBoundObject>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::UI::UIAutomation::AutomationElement>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::UI::UIAutomation::AutomationTextRange>::python_type;

namespace py::cpp::Windows::UI::UIAutomation
{
    // ----- AutomationConnection class --------------------
    constexpr const char* const _type_name_AutomationConnection = "AutomationConnection";

    static PyObject* _new_AutomationConnection(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_AutomationConnection);
        return nullptr;
    }

    static void _dealloc_AutomationConnection(py::wrapper::Windows::UI::UIAutomation::AutomationConnection* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* AutomationConnection_get_AppUserModelId(py::wrapper::Windows::UI::UIAutomation::AutomationConnection* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.AppUserModelId());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* AutomationConnection_get_ExecutableFileName(py::wrapper::Windows::UI::UIAutomation::AutomationConnection* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.ExecutableFileName());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* AutomationConnection_get_IsRemoteSystem(py::wrapper::Windows::UI::UIAutomation::AutomationConnection* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsRemoteSystem());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_AutomationConnection(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::UIAutomation::AutomationConnection>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_AutomationConnection[] = {
        { "_from", (PyCFunction)_from_AutomationConnection, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_AutomationConnection[] = {
        { const_cast<char*>("app_user_model_id"), (getter)AutomationConnection_get_AppUserModelId, nullptr, nullptr, nullptr },
        { const_cast<char*>("executable_file_name"), (getter)AutomationConnection_get_ExecutableFileName, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_remote_system"), (getter)AutomationConnection_get_IsRemoteSystem, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_AutomationConnection[] = 
    {
        { Py_tp_new, _new_AutomationConnection },
        { Py_tp_dealloc, _dealloc_AutomationConnection },
        { Py_tp_methods, _methods_AutomationConnection },
        { Py_tp_getset, _getset_AutomationConnection },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_AutomationConnection =
    {
        "_winrt_Windows_UI_UIAutomation.AutomationConnection",
        sizeof(py::wrapper::Windows::UI::UIAutomation::AutomationConnection),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_AutomationConnection
    };

    // ----- AutomationConnectionBoundObject class --------------------
    constexpr const char* const _type_name_AutomationConnectionBoundObject = "AutomationConnectionBoundObject";

    static PyObject* _new_AutomationConnectionBoundObject(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_AutomationConnectionBoundObject);
        return nullptr;
    }

    static void _dealloc_AutomationConnectionBoundObject(py::wrapper::Windows::UI::UIAutomation::AutomationConnectionBoundObject* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* AutomationConnectionBoundObject_get_Connection(py::wrapper::Windows::UI::UIAutomation::AutomationConnectionBoundObject* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Connection());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_AutomationConnectionBoundObject(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::UIAutomation::AutomationConnectionBoundObject>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_AutomationConnectionBoundObject[] = {
        { "_from", (PyCFunction)_from_AutomationConnectionBoundObject, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_AutomationConnectionBoundObject[] = {
        { const_cast<char*>("connection"), (getter)AutomationConnectionBoundObject_get_Connection, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_AutomationConnectionBoundObject[] = 
    {
        { Py_tp_new, _new_AutomationConnectionBoundObject },
        { Py_tp_dealloc, _dealloc_AutomationConnectionBoundObject },
        { Py_tp_methods, _methods_AutomationConnectionBoundObject },
        { Py_tp_getset, _getset_AutomationConnectionBoundObject },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_AutomationConnectionBoundObject =
    {
        "_winrt_Windows_UI_UIAutomation.AutomationConnectionBoundObject",
        sizeof(py::wrapper::Windows::UI::UIAutomation::AutomationConnectionBoundObject),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_AutomationConnectionBoundObject
    };

    // ----- AutomationElement class --------------------
    constexpr const char* const _type_name_AutomationElement = "AutomationElement";

    static PyObject* _new_AutomationElement(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_AutomationElement);
        return nullptr;
    }

    static void _dealloc_AutomationElement(py::wrapper::Windows::UI::UIAutomation::AutomationElement* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* AutomationElement_get_AppUserModelId(py::wrapper::Windows::UI::UIAutomation::AutomationElement* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.AppUserModelId());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* AutomationElement_get_ExecutableFileName(py::wrapper::Windows::UI::UIAutomation::AutomationElement* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.ExecutableFileName());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* AutomationElement_get_IsRemoteSystem(py::wrapper::Windows::UI::UIAutomation::AutomationElement* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsRemoteSystem());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_AutomationElement(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::UIAutomation::AutomationElement>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_AutomationElement[] = {
        { "_from", (PyCFunction)_from_AutomationElement, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_AutomationElement[] = {
        { const_cast<char*>("app_user_model_id"), (getter)AutomationElement_get_AppUserModelId, nullptr, nullptr, nullptr },
        { const_cast<char*>("executable_file_name"), (getter)AutomationElement_get_ExecutableFileName, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_remote_system"), (getter)AutomationElement_get_IsRemoteSystem, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_AutomationElement[] = 
    {
        { Py_tp_new, _new_AutomationElement },
        { Py_tp_dealloc, _dealloc_AutomationElement },
        { Py_tp_methods, _methods_AutomationElement },
        { Py_tp_getset, _getset_AutomationElement },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_AutomationElement =
    {
        "_winrt_Windows_UI_UIAutomation.AutomationElement",
        sizeof(py::wrapper::Windows::UI::UIAutomation::AutomationElement),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_AutomationElement
    };

    // ----- AutomationTextRange class --------------------
    constexpr const char* const _type_name_AutomationTextRange = "AutomationTextRange";

    static PyObject* _new_AutomationTextRange(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_AutomationTextRange);
        return nullptr;
    }

    static void _dealloc_AutomationTextRange(py::wrapper::Windows::UI::UIAutomation::AutomationTextRange* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* _from_AutomationTextRange(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::UIAutomation::AutomationTextRange>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_AutomationTextRange[] = {
        { "_from", (PyCFunction)_from_AutomationTextRange, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_AutomationTextRange[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_AutomationTextRange[] = 
    {
        { Py_tp_new, _new_AutomationTextRange },
        { Py_tp_dealloc, _dealloc_AutomationTextRange },
        { Py_tp_methods, _methods_AutomationTextRange },
        { Py_tp_getset, _getset_AutomationTextRange },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_AutomationTextRange =
    {
        "_winrt_Windows_UI_UIAutomation.AutomationTextRange",
        sizeof(py::wrapper::Windows::UI::UIAutomation::AutomationTextRange),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_AutomationTextRange
    };

    // ----- Windows.UI.UIAutomation Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::UI::UIAutomation::AutomationConnection>::python_type = py::register_python_type(module, _type_name_AutomationConnection, &_type_spec_AutomationConnection, bases.get());
            py::winrt_type<winrt::Windows::UI::UIAutomation::AutomationConnectionBoundObject>::python_type = py::register_python_type(module, _type_name_AutomationConnectionBoundObject, &_type_spec_AutomationConnectionBoundObject, bases.get());
            py::winrt_type<winrt::Windows::UI::UIAutomation::AutomationElement>::python_type = py::register_python_type(module, _type_name_AutomationElement, &_type_spec_AutomationElement, bases.get());
            py::winrt_type<winrt::Windows::UI::UIAutomation::AutomationTextRange>::python_type = py::register_python_type(module, _type_name_AutomationTextRange, &_type_spec_AutomationTextRange, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.UI.UIAutomation");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_UI_UIAutomation",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::UI::UIAutomation

PyMODINIT_FUNC
PyInit__winrt_Windows_UI_UIAutomation (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::UI::UIAutomation::module_def);
}
