// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.ApplicationModel.SocialInfo.Provider.h"

PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialInfoProviderManager>::python_type;

namespace py::cpp::Windows::ApplicationModel::SocialInfo::Provider
{
    // ----- SocialDashboardItemUpdater class --------------------
    constexpr const char* const _type_name_SocialDashboardItemUpdater = "SocialDashboardItemUpdater";

    static PyObject* _new_SocialDashboardItemUpdater(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SocialDashboardItemUpdater);
        return nullptr;
    }

    static void _dealloc_SocialDashboardItemUpdater(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SocialDashboardItemUpdater_CommitAsync(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.CommitAsync());
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

    static PyObject* SocialDashboardItemUpdater_get_Timestamp(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Timestamp());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int SocialDashboardItemUpdater_put_Timestamp(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::DateTime>(arg);

            self->obj.Timestamp(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* SocialDashboardItemUpdater_get_Thumbnail(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Thumbnail());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int SocialDashboardItemUpdater_put_Thumbnail(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::ApplicationModel::SocialInfo::SocialItemThumbnail>(arg);

            self->obj.Thumbnail(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* SocialDashboardItemUpdater_get_TargetUri(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.TargetUri());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int SocialDashboardItemUpdater_put_TargetUri(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::Uri>(arg);

            self->obj.TargetUri(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* SocialDashboardItemUpdater_get_Content(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Content());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SocialDashboardItemUpdater_get_OwnerRemoteId(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.OwnerRemoteId());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SocialDashboardItemUpdater(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SocialDashboardItemUpdater[] = {
        { "commit_async", (PyCFunction)SocialDashboardItemUpdater_CommitAsync, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_SocialDashboardItemUpdater, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SocialDashboardItemUpdater[] = {
        { const_cast<char*>("timestamp"), (getter)SocialDashboardItemUpdater_get_Timestamp, (setter)SocialDashboardItemUpdater_put_Timestamp, nullptr, nullptr },
        { const_cast<char*>("thumbnail"), (getter)SocialDashboardItemUpdater_get_Thumbnail, (setter)SocialDashboardItemUpdater_put_Thumbnail, nullptr, nullptr },
        { const_cast<char*>("target_uri"), (getter)SocialDashboardItemUpdater_get_TargetUri, (setter)SocialDashboardItemUpdater_put_TargetUri, nullptr, nullptr },
        { const_cast<char*>("content"), (getter)SocialDashboardItemUpdater_get_Content, nullptr, nullptr, nullptr },
        { const_cast<char*>("owner_remote_id"), (getter)SocialDashboardItemUpdater_get_OwnerRemoteId, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SocialDashboardItemUpdater[] = 
    {
        { Py_tp_new, _new_SocialDashboardItemUpdater },
        { Py_tp_dealloc, _dealloc_SocialDashboardItemUpdater },
        { Py_tp_methods, _methods_SocialDashboardItemUpdater },
        { Py_tp_getset, _getset_SocialDashboardItemUpdater },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SocialDashboardItemUpdater =
    {
        "_winrt_Windows_ApplicationModel_SocialInfo_Provider.SocialDashboardItemUpdater",
        sizeof(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SocialDashboardItemUpdater
    };

    // ----- SocialFeedUpdater class --------------------
    constexpr const char* const _type_name_SocialFeedUpdater = "SocialFeedUpdater";

    static PyObject* _new_SocialFeedUpdater(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SocialFeedUpdater);
        return nullptr;
    }

    static void _dealloc_SocialFeedUpdater(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SocialFeedUpdater_CommitAsync(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.CommitAsync());
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

    static PyObject* SocialFeedUpdater_get_Items(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Items());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SocialFeedUpdater_get_Kind(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater* self, void* /*unused*/) noexcept
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

    static PyObject* SocialFeedUpdater_get_OwnerRemoteId(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.OwnerRemoteId());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SocialFeedUpdater(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SocialFeedUpdater[] = {
        { "commit_async", (PyCFunction)SocialFeedUpdater_CommitAsync, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_SocialFeedUpdater, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SocialFeedUpdater[] = {
        { const_cast<char*>("items"), (getter)SocialFeedUpdater_get_Items, nullptr, nullptr, nullptr },
        { const_cast<char*>("kind"), (getter)SocialFeedUpdater_get_Kind, nullptr, nullptr, nullptr },
        { const_cast<char*>("owner_remote_id"), (getter)SocialFeedUpdater_get_OwnerRemoteId, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SocialFeedUpdater[] = 
    {
        { Py_tp_new, _new_SocialFeedUpdater },
        { Py_tp_dealloc, _dealloc_SocialFeedUpdater },
        { Py_tp_methods, _methods_SocialFeedUpdater },
        { Py_tp_getset, _getset_SocialFeedUpdater },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SocialFeedUpdater =
    {
        "_winrt_Windows_ApplicationModel_SocialInfo_Provider.SocialFeedUpdater",
        sizeof(py::wrapper::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SocialFeedUpdater
    };

    // ----- SocialInfoProviderManager class --------------------
    constexpr const char* const _type_name_SocialInfoProviderManager = "SocialInfoProviderManager";

    static PyObject* _new_SocialInfoProviderManager(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SocialInfoProviderManager);
        return nullptr;
    }

    static PyObject* SocialInfoProviderManager_CreateDashboardItemUpdaterAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);

                return py::convert(winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialInfoProviderManager::CreateDashboardItemUpdaterAsync(param0));
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

    static PyObject* SocialInfoProviderManager_CreateSocialFeedUpdaterAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 3)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::ApplicationModel::SocialInfo::SocialFeedKind>(args, 0);
                auto param1 = py::convert_to<winrt::Windows::ApplicationModel::SocialInfo::SocialFeedUpdateMode>(args, 1);
                auto param2 = py::convert_to<winrt::hstring>(args, 2);

                return py::convert(winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialInfoProviderManager::CreateSocialFeedUpdaterAsync(param0, param1, param2));
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

    static PyObject* SocialInfoProviderManager_DeprovisionAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialInfoProviderManager::DeprovisionAsync());
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

    static PyObject* SocialInfoProviderManager_ProvisionAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialInfoProviderManager::ProvisionAsync());
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

    static PyObject* SocialInfoProviderManager_ReportNewContentAvailable(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::Windows::ApplicationModel::SocialInfo::SocialFeedKind>(args, 1);

                winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialInfoProviderManager::ReportNewContentAvailable(param0, param1);
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

    static PyObject* SocialInfoProviderManager_UpdateBadgeCountValue(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<int32_t>(args, 1);

                winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialInfoProviderManager::UpdateBadgeCountValue(param0, param1);
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

    static PyMethodDef _methods_SocialInfoProviderManager[] = {
        { "create_dashboard_item_updater_async", (PyCFunction)SocialInfoProviderManager_CreateDashboardItemUpdaterAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "create_social_feed_updater_async", (PyCFunction)SocialInfoProviderManager_CreateSocialFeedUpdaterAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "deprovision_async", (PyCFunction)SocialInfoProviderManager_DeprovisionAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "provision_async", (PyCFunction)SocialInfoProviderManager_ProvisionAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "report_new_content_available", (PyCFunction)SocialInfoProviderManager_ReportNewContentAvailable, METH_VARARGS | METH_STATIC, nullptr },
        { "update_badge_count_value", (PyCFunction)SocialInfoProviderManager_UpdateBadgeCountValue, METH_VARARGS | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SocialInfoProviderManager[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_SocialInfoProviderManager[] = 
    {
        { Py_tp_new, _new_SocialInfoProviderManager },
        { Py_tp_methods, _methods_SocialInfoProviderManager },
        { Py_tp_getset, _getset_SocialInfoProviderManager },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SocialInfoProviderManager =
    {
        "_winrt_Windows_ApplicationModel_SocialInfo_Provider.SocialInfoProviderManager",
        0,
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SocialInfoProviderManager
    };

    // ----- Windows.ApplicationModel.SocialInfo.Provider Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialDashboardItemUpdater>::python_type = py::register_python_type(module, _type_name_SocialDashboardItemUpdater, &_type_spec_SocialDashboardItemUpdater, bases.get());
            py::winrt_type<winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialFeedUpdater>::python_type = py::register_python_type(module, _type_name_SocialFeedUpdater, &_type_spec_SocialFeedUpdater, bases.get());
            py::winrt_type<winrt::Windows::ApplicationModel::SocialInfo::Provider::SocialInfoProviderManager>::python_type = py::register_python_type(module, _type_name_SocialInfoProviderManager, &_type_spec_SocialInfoProviderManager, nullptr);

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

    PyDoc_STRVAR(module_doc, "Windows.ApplicationModel.SocialInfo.Provider");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_ApplicationModel_SocialInfo_Provider",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::ApplicationModel::SocialInfo::Provider

PyMODINIT_FUNC
PyInit__winrt_Windows_ApplicationModel_SocialInfo_Provider (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::ApplicationModel::SocialInfo::Provider::module_def);
}
