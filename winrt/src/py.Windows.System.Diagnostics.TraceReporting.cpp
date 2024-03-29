// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.System.Diagnostics.TraceReporting.h"

PyTypeObject* py::winrt_type<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceRuntimeInfo>::python_type;

namespace py::cpp::Windows::System::Diagnostics::TraceReporting
{
    // ----- PlatformDiagnosticActions class --------------------
    constexpr const char* const _type_name_PlatformDiagnosticActions = "PlatformDiagnosticActions";

    static PyObject* _new_PlatformDiagnosticActions(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_PlatformDiagnosticActions);
        return nullptr;
    }

    static PyObject* PlatformDiagnosticActions_DownloadLatestSettingsForNamespace(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 5)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::hstring>(args, 1);
                auto param2 = py::convert_to<bool>(args, 2);
                auto param3 = py::convert_to<bool>(args, 3);
                auto param4 = py::convert_to<bool>(args, 4);

                return py::convert(winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions::DownloadLatestSettingsForNamespace(param0, param1, param2, param3, param4));
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

    static PyObject* PlatformDiagnosticActions_ForceUpload(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 3)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticEventBufferLatencies>(args, 0);
                auto param1 = py::convert_to<bool>(args, 1);
                auto param2 = py::convert_to<bool>(args, 2);

                return py::convert(winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions::ForceUpload(param0, param1, param2));
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

    static PyObject* PlatformDiagnosticActions_GetActiveScenarioList(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions::GetActiveScenarioList());
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

    static PyObject* PlatformDiagnosticActions_GetActiveTraceRuntime(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceSlotType>(args, 0);

                return py::convert(winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions::GetActiveTraceRuntime(param0));
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

    static PyObject* PlatformDiagnosticActions_GetKnownTraceList(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceSlotType>(args, 0);

                return py::convert(winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions::GetKnownTraceList(param0));
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

    static PyObject* PlatformDiagnosticActions_IsScenarioEnabled(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::guid>(args, 0);

                return py::convert(winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions::IsScenarioEnabled(param0));
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

    static PyObject* PlatformDiagnosticActions_IsTraceRunning(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 3)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceSlotType>(args, 0);
                auto param1 = py::convert_to<winrt::guid>(args, 1);
                auto param2 = py::convert_to<uint64_t>(args, 2);

                return py::convert(winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions::IsTraceRunning(param0, param1, param2));
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

    static PyObject* PlatformDiagnosticActions_TryEscalateScenario(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 6)
        {
            try
            {
                auto param0 = py::convert_to<winrt::guid>(args, 0);
                auto param1 = py::convert_to<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticEscalationType>(args, 1);
                auto param2 = py::convert_to<winrt::hstring>(args, 2);
                auto param3 = py::convert_to<bool>(args, 3);
                auto param4 = py::convert_to<bool>(args, 4);
                auto param5 = py::convert_to<winrt::Windows::Foundation::Collections::IMapView<winrt::hstring, winrt::hstring>>(args, 5);

                return py::convert(winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions::TryEscalateScenario(param0, param1, param2, param3, param4, param5));
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

    static PyMethodDef _methods_PlatformDiagnosticActions[] = {
        { "download_latest_settings_for_namespace", (PyCFunction)PlatformDiagnosticActions_DownloadLatestSettingsForNamespace, METH_VARARGS | METH_STATIC, nullptr },
        { "force_upload", (PyCFunction)PlatformDiagnosticActions_ForceUpload, METH_VARARGS | METH_STATIC, nullptr },
        { "get_active_scenario_list", (PyCFunction)PlatformDiagnosticActions_GetActiveScenarioList, METH_VARARGS | METH_STATIC, nullptr },
        { "get_active_trace_runtime", (PyCFunction)PlatformDiagnosticActions_GetActiveTraceRuntime, METH_VARARGS | METH_STATIC, nullptr },
        { "get_known_trace_list", (PyCFunction)PlatformDiagnosticActions_GetKnownTraceList, METH_VARARGS | METH_STATIC, nullptr },
        { "is_scenario_enabled", (PyCFunction)PlatformDiagnosticActions_IsScenarioEnabled, METH_VARARGS | METH_STATIC, nullptr },
        { "is_trace_running", (PyCFunction)PlatformDiagnosticActions_IsTraceRunning, METH_VARARGS | METH_STATIC, nullptr },
        { "try_escalate_scenario", (PyCFunction)PlatformDiagnosticActions_TryEscalateScenario, METH_VARARGS | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_PlatformDiagnosticActions[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_PlatformDiagnosticActions[] = 
    {
        { Py_tp_new, _new_PlatformDiagnosticActions },
        { Py_tp_methods, _methods_PlatformDiagnosticActions },
        { Py_tp_getset, _getset_PlatformDiagnosticActions },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_PlatformDiagnosticActions =
    {
        "_winrt_Windows_System_Diagnostics_TraceReporting.PlatformDiagnosticActions",
        0,
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_PlatformDiagnosticActions
    };

    // ----- PlatformDiagnosticTraceInfo class --------------------
    constexpr const char* const _type_name_PlatformDiagnosticTraceInfo = "PlatformDiagnosticTraceInfo";

    static PyObject* _new_PlatformDiagnosticTraceInfo(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_PlatformDiagnosticTraceInfo);
        return nullptr;
    }

    static void _dealloc_PlatformDiagnosticTraceInfo(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* PlatformDiagnosticTraceInfo_get_IsAutoLogger(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsAutoLogger());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PlatformDiagnosticTraceInfo_get_IsExclusive(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsExclusive());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PlatformDiagnosticTraceInfo_get_MaxTraceDurationFileTime(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.MaxTraceDurationFileTime());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PlatformDiagnosticTraceInfo_get_Priority(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Priority());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PlatformDiagnosticTraceInfo_get_ProfileHash(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.ProfileHash());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PlatformDiagnosticTraceInfo_get_ScenarioId(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.ScenarioId());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_PlatformDiagnosticTraceInfo(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_PlatformDiagnosticTraceInfo[] = {
        { "_from", (PyCFunction)_from_PlatformDiagnosticTraceInfo, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_PlatformDiagnosticTraceInfo[] = {
        { const_cast<char*>("is_auto_logger"), (getter)PlatformDiagnosticTraceInfo_get_IsAutoLogger, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_exclusive"), (getter)PlatformDiagnosticTraceInfo_get_IsExclusive, nullptr, nullptr, nullptr },
        { const_cast<char*>("max_trace_duration_file_time"), (getter)PlatformDiagnosticTraceInfo_get_MaxTraceDurationFileTime, nullptr, nullptr, nullptr },
        { const_cast<char*>("priority"), (getter)PlatformDiagnosticTraceInfo_get_Priority, nullptr, nullptr, nullptr },
        { const_cast<char*>("profile_hash"), (getter)PlatformDiagnosticTraceInfo_get_ProfileHash, nullptr, nullptr, nullptr },
        { const_cast<char*>("scenario_id"), (getter)PlatformDiagnosticTraceInfo_get_ScenarioId, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_PlatformDiagnosticTraceInfo[] = 
    {
        { Py_tp_new, _new_PlatformDiagnosticTraceInfo },
        { Py_tp_dealloc, _dealloc_PlatformDiagnosticTraceInfo },
        { Py_tp_methods, _methods_PlatformDiagnosticTraceInfo },
        { Py_tp_getset, _getset_PlatformDiagnosticTraceInfo },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_PlatformDiagnosticTraceInfo =
    {
        "_winrt_Windows_System_Diagnostics_TraceReporting.PlatformDiagnosticTraceInfo",
        sizeof(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_PlatformDiagnosticTraceInfo
    };

    // ----- PlatformDiagnosticTraceRuntimeInfo class --------------------
    constexpr const char* const _type_name_PlatformDiagnosticTraceRuntimeInfo = "PlatformDiagnosticTraceRuntimeInfo";

    static PyObject* _new_PlatformDiagnosticTraceRuntimeInfo(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_PlatformDiagnosticTraceRuntimeInfo);
        return nullptr;
    }

    static void _dealloc_PlatformDiagnosticTraceRuntimeInfo(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceRuntimeInfo* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* PlatformDiagnosticTraceRuntimeInfo_get_EtwRuntimeFileTime(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceRuntimeInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.EtwRuntimeFileTime());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PlatformDiagnosticTraceRuntimeInfo_get_RuntimeFileTime(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceRuntimeInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.RuntimeFileTime());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_PlatformDiagnosticTraceRuntimeInfo(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceRuntimeInfo>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_PlatformDiagnosticTraceRuntimeInfo[] = {
        { "_from", (PyCFunction)_from_PlatformDiagnosticTraceRuntimeInfo, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_PlatformDiagnosticTraceRuntimeInfo[] = {
        { const_cast<char*>("etw_runtime_file_time"), (getter)PlatformDiagnosticTraceRuntimeInfo_get_EtwRuntimeFileTime, nullptr, nullptr, nullptr },
        { const_cast<char*>("runtime_file_time"), (getter)PlatformDiagnosticTraceRuntimeInfo_get_RuntimeFileTime, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_PlatformDiagnosticTraceRuntimeInfo[] = 
    {
        { Py_tp_new, _new_PlatformDiagnosticTraceRuntimeInfo },
        { Py_tp_dealloc, _dealloc_PlatformDiagnosticTraceRuntimeInfo },
        { Py_tp_methods, _methods_PlatformDiagnosticTraceRuntimeInfo },
        { Py_tp_getset, _getset_PlatformDiagnosticTraceRuntimeInfo },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_PlatformDiagnosticTraceRuntimeInfo =
    {
        "_winrt_Windows_System_Diagnostics_TraceReporting.PlatformDiagnosticTraceRuntimeInfo",
        sizeof(py::wrapper::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceRuntimeInfo),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_PlatformDiagnosticTraceRuntimeInfo
    };

    // ----- Windows.System.Diagnostics.TraceReporting Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticActions>::python_type = py::register_python_type(module, _type_name_PlatformDiagnosticActions, &_type_spec_PlatformDiagnosticActions, nullptr);
            py::winrt_type<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceInfo>::python_type = py::register_python_type(module, _type_name_PlatformDiagnosticTraceInfo, &_type_spec_PlatformDiagnosticTraceInfo, bases.get());
            py::winrt_type<winrt::Windows::System::Diagnostics::TraceReporting::PlatformDiagnosticTraceRuntimeInfo>::python_type = py::register_python_type(module, _type_name_PlatformDiagnosticTraceRuntimeInfo, &_type_spec_PlatformDiagnosticTraceRuntimeInfo, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.System.Diagnostics.TraceReporting");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_System_Diagnostics_TraceReporting",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::System::Diagnostics::TraceReporting

PyMODINIT_FUNC
PyInit__winrt_Windows_System_Diagnostics_TraceReporting (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::System::Diagnostics::TraceReporting::module_def);
}
