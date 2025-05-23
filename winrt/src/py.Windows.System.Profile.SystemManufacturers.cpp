// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.System.Profile.SystemManufacturers.h"

PyTypeObject* py::winrt_type<winrt::Windows::System::Profile::SystemManufacturers::OemSupportInfo>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::System::Profile::SystemManufacturers::SmbiosInformation>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::System::Profile::SystemManufacturers::SystemSupportInfo>::python_type;

namespace py::cpp::Windows::System::Profile::SystemManufacturers
{
    // ----- OemSupportInfo class --------------------
    constexpr const char* const _type_name_OemSupportInfo = "OemSupportInfo";

    static PyObject* _new_OemSupportInfo(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_OemSupportInfo);
        return nullptr;
    }

    static void _dealloc_OemSupportInfo(py::wrapper::Windows::System::Profile::SystemManufacturers::OemSupportInfo* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* OemSupportInfo_get_SupportAppLink(py::wrapper::Windows::System::Profile::SystemManufacturers::OemSupportInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SupportAppLink());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* OemSupportInfo_get_SupportLink(py::wrapper::Windows::System::Profile::SystemManufacturers::OemSupportInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SupportLink());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* OemSupportInfo_get_SupportProvider(py::wrapper::Windows::System::Profile::SystemManufacturers::OemSupportInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SupportProvider());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_OemSupportInfo(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Profile::SystemManufacturers::OemSupportInfo>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_OemSupportInfo[] = {
        { "_from", (PyCFunction)_from_OemSupportInfo, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_OemSupportInfo[] = {
        { const_cast<char*>("support_app_link"), (getter)OemSupportInfo_get_SupportAppLink, nullptr, nullptr, nullptr },
        { const_cast<char*>("support_link"), (getter)OemSupportInfo_get_SupportLink, nullptr, nullptr, nullptr },
        { const_cast<char*>("support_provider"), (getter)OemSupportInfo_get_SupportProvider, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_OemSupportInfo[] = 
    {
        { Py_tp_new, _new_OemSupportInfo },
        { Py_tp_dealloc, _dealloc_OemSupportInfo },
        { Py_tp_methods, _methods_OemSupportInfo },
        { Py_tp_getset, _getset_OemSupportInfo },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_OemSupportInfo =
    {
        "_winrt_Windows_System_Profile_SystemManufacturers.OemSupportInfo",
        sizeof(py::wrapper::Windows::System::Profile::SystemManufacturers::OemSupportInfo),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_OemSupportInfo
    };

    // ----- SmbiosInformation class --------------------
    constexpr const char* const _type_name_SmbiosInformation = "SmbiosInformation";

    static PyObject* _new_SmbiosInformation(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SmbiosInformation);
        return nullptr;
    }

    static PyObject* SmbiosInformation_get_SerialNumber(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::System::Profile::SystemManufacturers::SmbiosInformation::SerialNumber());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SmbiosInformation[] = {
        { "get_serial_number", (PyCFunction)SmbiosInformation_get_SerialNumber, METH_NOARGS | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SmbiosInformation[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_SmbiosInformation[] = 
    {
        { Py_tp_new, _new_SmbiosInformation },
        { Py_tp_methods, _methods_SmbiosInformation },
        { Py_tp_getset, _getset_SmbiosInformation },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SmbiosInformation =
    {
        "_winrt_Windows_System_Profile_SystemManufacturers.SmbiosInformation",
        0,
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SmbiosInformation
    };

    // ----- SystemSupportDeviceInfo class --------------------
    constexpr const char* const _type_name_SystemSupportDeviceInfo = "SystemSupportDeviceInfo";

    static PyObject* _new_SystemSupportDeviceInfo(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SystemSupportDeviceInfo);
        return nullptr;
    }

    static void _dealloc_SystemSupportDeviceInfo(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SystemSupportDeviceInfo_get_FriendlyName(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.FriendlyName());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SystemSupportDeviceInfo_get_OperatingSystem(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.OperatingSystem());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SystemSupportDeviceInfo_get_SystemFirmwareVersion(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SystemFirmwareVersion());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SystemSupportDeviceInfo_get_SystemHardwareVersion(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SystemHardwareVersion());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SystemSupportDeviceInfo_get_SystemManufacturer(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SystemManufacturer());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SystemSupportDeviceInfo_get_SystemProductName(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SystemProductName());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SystemSupportDeviceInfo_get_SystemSku(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SystemSku());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SystemSupportDeviceInfo(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SystemSupportDeviceInfo[] = {
        { "_from", (PyCFunction)_from_SystemSupportDeviceInfo, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SystemSupportDeviceInfo[] = {
        { const_cast<char*>("friendly_name"), (getter)SystemSupportDeviceInfo_get_FriendlyName, nullptr, nullptr, nullptr },
        { const_cast<char*>("operating_system"), (getter)SystemSupportDeviceInfo_get_OperatingSystem, nullptr, nullptr, nullptr },
        { const_cast<char*>("system_firmware_version"), (getter)SystemSupportDeviceInfo_get_SystemFirmwareVersion, nullptr, nullptr, nullptr },
        { const_cast<char*>("system_hardware_version"), (getter)SystemSupportDeviceInfo_get_SystemHardwareVersion, nullptr, nullptr, nullptr },
        { const_cast<char*>("system_manufacturer"), (getter)SystemSupportDeviceInfo_get_SystemManufacturer, nullptr, nullptr, nullptr },
        { const_cast<char*>("system_product_name"), (getter)SystemSupportDeviceInfo_get_SystemProductName, nullptr, nullptr, nullptr },
        { const_cast<char*>("system_sku"), (getter)SystemSupportDeviceInfo_get_SystemSku, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SystemSupportDeviceInfo[] = 
    {
        { Py_tp_new, _new_SystemSupportDeviceInfo },
        { Py_tp_dealloc, _dealloc_SystemSupportDeviceInfo },
        { Py_tp_methods, _methods_SystemSupportDeviceInfo },
        { Py_tp_getset, _getset_SystemSupportDeviceInfo },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SystemSupportDeviceInfo =
    {
        "_winrt_Windows_System_Profile_SystemManufacturers.SystemSupportDeviceInfo",
        sizeof(py::wrapper::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SystemSupportDeviceInfo
    };

    // ----- SystemSupportInfo class --------------------
    constexpr const char* const _type_name_SystemSupportInfo = "SystemSupportInfo";

    static PyObject* _new_SystemSupportInfo(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SystemSupportInfo);
        return nullptr;
    }

    static PyObject* SystemSupportInfo_get_LocalSystemEdition(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::System::Profile::SystemManufacturers::SystemSupportInfo::LocalSystemEdition());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SystemSupportInfo_get_OemSupportInfo(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::System::Profile::SystemManufacturers::SystemSupportInfo::OemSupportInfo());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SystemSupportInfo_get_LocalDeviceInfo(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::System::Profile::SystemManufacturers::SystemSupportInfo::LocalDeviceInfo());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SystemSupportInfo[] = {
        { "get_local_system_edition", (PyCFunction)SystemSupportInfo_get_LocalSystemEdition, METH_NOARGS | METH_STATIC, nullptr },
        { "get_oem_support_info", (PyCFunction)SystemSupportInfo_get_OemSupportInfo, METH_NOARGS | METH_STATIC, nullptr },
        { "get_local_device_info", (PyCFunction)SystemSupportInfo_get_LocalDeviceInfo, METH_NOARGS | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SystemSupportInfo[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_SystemSupportInfo[] = 
    {
        { Py_tp_new, _new_SystemSupportInfo },
        { Py_tp_methods, _methods_SystemSupportInfo },
        { Py_tp_getset, _getset_SystemSupportInfo },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SystemSupportInfo =
    {
        "_winrt_Windows_System_Profile_SystemManufacturers.SystemSupportInfo",
        0,
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SystemSupportInfo
    };

    // ----- Windows.System.Profile.SystemManufacturers Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::System::Profile::SystemManufacturers::OemSupportInfo>::python_type = py::register_python_type(module, _type_name_OemSupportInfo, &_type_spec_OemSupportInfo, bases.get());
            py::winrt_type<winrt::Windows::System::Profile::SystemManufacturers::SmbiosInformation>::python_type = py::register_python_type(module, _type_name_SmbiosInformation, &_type_spec_SmbiosInformation, nullptr);
            py::winrt_type<winrt::Windows::System::Profile::SystemManufacturers::SystemSupportDeviceInfo>::python_type = py::register_python_type(module, _type_name_SystemSupportDeviceInfo, &_type_spec_SystemSupportDeviceInfo, bases.get());
            py::winrt_type<winrt::Windows::System::Profile::SystemManufacturers::SystemSupportInfo>::python_type = py::register_python_type(module, _type_name_SystemSupportInfo, &_type_spec_SystemSupportInfo, nullptr);

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

    PyDoc_STRVAR(module_doc, "Windows.System.Profile.SystemManufacturers");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_System_Profile_SystemManufacturers",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::System::Profile::SystemManufacturers

PyMODINIT_FUNC
PyInit__winrt_Windows_System_Profile_SystemManufacturers (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::System::Profile::SystemManufacturers::module_def);
}
