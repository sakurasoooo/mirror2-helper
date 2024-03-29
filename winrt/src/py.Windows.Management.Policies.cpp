// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.Management.Policies.h"

PyTypeObject* py::winrt_type<winrt::Windows::Management::Policies::NamedPolicy>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Management::Policies::NamedPolicyData>::python_type;

namespace py::cpp::Windows::Management::Policies
{
    // ----- NamedPolicy class --------------------
    constexpr const char* const _type_name_NamedPolicy = "NamedPolicy";

    static PyObject* _new_NamedPolicy(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_NamedPolicy);
        return nullptr;
    }

    static PyObject* NamedPolicy_GetPolicyFromPath(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::hstring>(args, 1);

                return py::convert(winrt::Windows::Management::Policies::NamedPolicy::GetPolicyFromPath(param0, param1));
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

    static PyObject* NamedPolicy_GetPolicyFromPathForUser(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 3)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::System::User>(args, 0);
                auto param1 = py::convert_to<winrt::hstring>(args, 1);
                auto param2 = py::convert_to<winrt::hstring>(args, 2);

                return py::convert(winrt::Windows::Management::Policies::NamedPolicy::GetPolicyFromPathForUser(param0, param1, param2));
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

    static PyMethodDef _methods_NamedPolicy[] = {
        { "get_policy_from_path", (PyCFunction)NamedPolicy_GetPolicyFromPath, METH_VARARGS | METH_STATIC, nullptr },
        { "get_policy_from_path_for_user", (PyCFunction)NamedPolicy_GetPolicyFromPathForUser, METH_VARARGS | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_NamedPolicy[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_NamedPolicy[] = 
    {
        { Py_tp_new, _new_NamedPolicy },
        { Py_tp_methods, _methods_NamedPolicy },
        { Py_tp_getset, _getset_NamedPolicy },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_NamedPolicy =
    {
        "_winrt_Windows_Management_Policies.NamedPolicy",
        0,
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_NamedPolicy
    };

    // ----- NamedPolicyData class --------------------
    constexpr const char* const _type_name_NamedPolicyData = "NamedPolicyData";

    static PyObject* _new_NamedPolicyData(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_NamedPolicyData);
        return nullptr;
    }

    static void _dealloc_NamedPolicyData(py::wrapper::Windows::Management::Policies::NamedPolicyData* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* NamedPolicyData_GetBinary(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetBinary());
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

    static PyObject* NamedPolicyData_GetBoolean(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetBoolean());
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

    static PyObject* NamedPolicyData_GetInt32(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetInt32());
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

    static PyObject* NamedPolicyData_GetInt64(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetInt64());
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

    static PyObject* NamedPolicyData_GetString(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetString());
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

    static PyObject* NamedPolicyData_get_Area(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Area());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* NamedPolicyData_get_IsManaged(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsManaged());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* NamedPolicyData_get_IsUserPolicy(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsUserPolicy());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* NamedPolicyData_get_Kind(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Kind());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* NamedPolicyData_get_Name(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Name());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* NamedPolicyData_get_User(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.User());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* NamedPolicyData_add_Changed(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::TypedEventHandler<winrt::Windows::Management::Policies::NamedPolicyData, winrt::Windows::Foundation::IInspectable>>(arg);

            return py::convert(self->obj.Changed(param0));
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* NamedPolicyData_remove_Changed(py::wrapper::Windows::Management::Policies::NamedPolicyData* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::event_token>(arg);

            self->obj.Changed(param0);
            Py_RETURN_NONE;
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_NamedPolicyData(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Management::Policies::NamedPolicyData>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_NamedPolicyData[] = {
        { "get_binary", (PyCFunction)NamedPolicyData_GetBinary, METH_VARARGS, nullptr },
        { "get_boolean", (PyCFunction)NamedPolicyData_GetBoolean, METH_VARARGS, nullptr },
        { "get_int32", (PyCFunction)NamedPolicyData_GetInt32, METH_VARARGS, nullptr },
        { "get_int64", (PyCFunction)NamedPolicyData_GetInt64, METH_VARARGS, nullptr },
        { "get_string", (PyCFunction)NamedPolicyData_GetString, METH_VARARGS, nullptr },
        { "add_changed", (PyCFunction)NamedPolicyData_add_Changed, METH_O, nullptr },
        { "remove_changed", (PyCFunction)NamedPolicyData_remove_Changed, METH_O, nullptr },
        { "_from", (PyCFunction)_from_NamedPolicyData, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_NamedPolicyData[] = {
        { const_cast<char*>("area"), (getter)NamedPolicyData_get_Area, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_managed"), (getter)NamedPolicyData_get_IsManaged, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_user_policy"), (getter)NamedPolicyData_get_IsUserPolicy, nullptr, nullptr, nullptr },
        { const_cast<char*>("kind"), (getter)NamedPolicyData_get_Kind, nullptr, nullptr, nullptr },
        { const_cast<char*>("name"), (getter)NamedPolicyData_get_Name, nullptr, nullptr, nullptr },
        { const_cast<char*>("user"), (getter)NamedPolicyData_get_User, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_NamedPolicyData[] = 
    {
        { Py_tp_new, _new_NamedPolicyData },
        { Py_tp_dealloc, _dealloc_NamedPolicyData },
        { Py_tp_methods, _methods_NamedPolicyData },
        { Py_tp_getset, _getset_NamedPolicyData },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_NamedPolicyData =
    {
        "_winrt_Windows_Management_Policies.NamedPolicyData",
        sizeof(py::wrapper::Windows::Management::Policies::NamedPolicyData),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_NamedPolicyData
    };

    // ----- Windows.Management.Policies Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::Management::Policies::NamedPolicy>::python_type = py::register_python_type(module, _type_name_NamedPolicy, &_type_spec_NamedPolicy, nullptr);
            py::winrt_type<winrt::Windows::Management::Policies::NamedPolicyData>::python_type = py::register_python_type(module, _type_name_NamedPolicyData, &_type_spec_NamedPolicyData, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.Management.Policies");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_Management_Policies",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::Management::Policies

PyMODINIT_FUNC
PyInit__winrt_Windows_Management_Policies (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::Management::Policies::module_def);
}
