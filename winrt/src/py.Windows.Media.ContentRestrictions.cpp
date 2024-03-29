// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.Media.ContentRestrictions.h"

PyTypeObject* py::winrt_type<winrt::Windows::Media::ContentRestrictions::ContentRestrictionsBrowsePolicy>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Media::ContentRestrictions::RatedContentDescription>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Media::ContentRestrictions::RatedContentRestrictions>::python_type;

namespace py::cpp::Windows::Media::ContentRestrictions
{
    // ----- ContentRestrictionsBrowsePolicy class --------------------
    constexpr const char* const _type_name_ContentRestrictionsBrowsePolicy = "ContentRestrictionsBrowsePolicy";

    static PyObject* _new_ContentRestrictionsBrowsePolicy(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_ContentRestrictionsBrowsePolicy);
        return nullptr;
    }

    static void _dealloc_ContentRestrictionsBrowsePolicy(py::wrapper::Windows::Media::ContentRestrictions::ContentRestrictionsBrowsePolicy* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* ContentRestrictionsBrowsePolicy_get_GeographicRegion(py::wrapper::Windows::Media::ContentRestrictions::ContentRestrictionsBrowsePolicy* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.GeographicRegion());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* ContentRestrictionsBrowsePolicy_get_MaxBrowsableAgeRating(py::wrapper::Windows::Media::ContentRestrictions::ContentRestrictionsBrowsePolicy* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.MaxBrowsableAgeRating());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* ContentRestrictionsBrowsePolicy_get_PreferredAgeRating(py::wrapper::Windows::Media::ContentRestrictions::ContentRestrictionsBrowsePolicy* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.PreferredAgeRating());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_ContentRestrictionsBrowsePolicy(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Media::ContentRestrictions::ContentRestrictionsBrowsePolicy>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_ContentRestrictionsBrowsePolicy[] = {
        { "_from", (PyCFunction)_from_ContentRestrictionsBrowsePolicy, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_ContentRestrictionsBrowsePolicy[] = {
        { const_cast<char*>("geographic_region"), (getter)ContentRestrictionsBrowsePolicy_get_GeographicRegion, nullptr, nullptr, nullptr },
        { const_cast<char*>("max_browsable_age_rating"), (getter)ContentRestrictionsBrowsePolicy_get_MaxBrowsableAgeRating, nullptr, nullptr, nullptr },
        { const_cast<char*>("preferred_age_rating"), (getter)ContentRestrictionsBrowsePolicy_get_PreferredAgeRating, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_ContentRestrictionsBrowsePolicy[] = 
    {
        { Py_tp_new, _new_ContentRestrictionsBrowsePolicy },
        { Py_tp_dealloc, _dealloc_ContentRestrictionsBrowsePolicy },
        { Py_tp_methods, _methods_ContentRestrictionsBrowsePolicy },
        { Py_tp_getset, _getset_ContentRestrictionsBrowsePolicy },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_ContentRestrictionsBrowsePolicy =
    {
        "_winrt_Windows_Media_ContentRestrictions.ContentRestrictionsBrowsePolicy",
        sizeof(py::wrapper::Windows::Media::ContentRestrictions::ContentRestrictionsBrowsePolicy),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_ContentRestrictionsBrowsePolicy
    };

    // ----- RatedContentDescription class --------------------
    constexpr const char* const _type_name_RatedContentDescription = "RatedContentDescription";

    static PyObject* _new_RatedContentDescription(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        if (kwds != nullptr)
        {
            py::set_invalid_kwd_args_error();
            return nullptr;
        }

        Py_ssize_t arg_count = PyTuple_Size(args);
        if (arg_count == 3)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::hstring>(args, 1);
                auto param2 = py::convert_to<winrt::Windows::Media::ContentRestrictions::RatedContentCategory>(args, 2);

                winrt::Windows::Media::ContentRestrictions::RatedContentDescription instance{ param0, param1, param2 };
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

    static void _dealloc_RatedContentDescription(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* RatedContentDescription_get_Title(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Title());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int RatedContentDescription_put_Title(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::hstring>(arg);

            self->obj.Title(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* RatedContentDescription_get_Ratings(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Ratings());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int RatedContentDescription_put_Ratings(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::Collections::IVector<winrt::hstring>>(arg);

            self->obj.Ratings(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* RatedContentDescription_get_Image(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Image());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int RatedContentDescription_put_Image(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Storage::Streams::IRandomAccessStreamReference>(arg);

            self->obj.Image(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* RatedContentDescription_get_Id(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Id());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int RatedContentDescription_put_Id(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::hstring>(arg);

            self->obj.Id(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* RatedContentDescription_get_Category(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Category());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int RatedContentDescription_put_Category(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Media::ContentRestrictions::RatedContentCategory>(arg);

            self->obj.Category(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* _from_RatedContentDescription(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Media::ContentRestrictions::RatedContentDescription>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_RatedContentDescription[] = {
        { "_from", (PyCFunction)_from_RatedContentDescription, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_RatedContentDescription[] = {
        { const_cast<char*>("title"), (getter)RatedContentDescription_get_Title, (setter)RatedContentDescription_put_Title, nullptr, nullptr },
        { const_cast<char*>("ratings"), (getter)RatedContentDescription_get_Ratings, (setter)RatedContentDescription_put_Ratings, nullptr, nullptr },
        { const_cast<char*>("image"), (getter)RatedContentDescription_get_Image, (setter)RatedContentDescription_put_Image, nullptr, nullptr },
        { const_cast<char*>("id"), (getter)RatedContentDescription_get_Id, (setter)RatedContentDescription_put_Id, nullptr, nullptr },
        { const_cast<char*>("category"), (getter)RatedContentDescription_get_Category, (setter)RatedContentDescription_put_Category, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_RatedContentDescription[] = 
    {
        { Py_tp_new, _new_RatedContentDescription },
        { Py_tp_dealloc, _dealloc_RatedContentDescription },
        { Py_tp_methods, _methods_RatedContentDescription },
        { Py_tp_getset, _getset_RatedContentDescription },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_RatedContentDescription =
    {
        "_winrt_Windows_Media_ContentRestrictions.RatedContentDescription",
        sizeof(py::wrapper::Windows::Media::ContentRestrictions::RatedContentDescription),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_RatedContentDescription
    };

    // ----- RatedContentRestrictions class --------------------
    constexpr const char* const _type_name_RatedContentRestrictions = "RatedContentRestrictions";

    static PyObject* _new_RatedContentRestrictions(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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
                auto param0 = py::convert_to<uint32_t>(args, 0);

                winrt::Windows::Media::ContentRestrictions::RatedContentRestrictions instance{ param0 };
                return py::wrap(instance, type);
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else if (arg_count == 0)
        {
            try
            {
                winrt::Windows::Media::ContentRestrictions::RatedContentRestrictions instance{  };
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

    static void _dealloc_RatedContentRestrictions(py::wrapper::Windows::Media::ContentRestrictions::RatedContentRestrictions* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* RatedContentRestrictions_GetBrowsePolicyAsync(py::wrapper::Windows::Media::ContentRestrictions::RatedContentRestrictions* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetBrowsePolicyAsync());
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

    static PyObject* RatedContentRestrictions_GetRestrictionLevelAsync(py::wrapper::Windows::Media::ContentRestrictions::RatedContentRestrictions* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Media::ContentRestrictions::RatedContentDescription>(args, 0);

                return py::convert(self->obj.GetRestrictionLevelAsync(param0));
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

    static PyObject* RatedContentRestrictions_RequestContentAccessAsync(py::wrapper::Windows::Media::ContentRestrictions::RatedContentRestrictions* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Media::ContentRestrictions::RatedContentDescription>(args, 0);

                return py::convert(self->obj.RequestContentAccessAsync(param0));
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

    static PyObject* RatedContentRestrictions_add_RestrictionsChanged(py::wrapper::Windows::Media::ContentRestrictions::RatedContentRestrictions* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::EventHandler<winrt::Windows::Foundation::IInspectable>>(arg);

            return py::convert(self->obj.RestrictionsChanged(param0));
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* RatedContentRestrictions_remove_RestrictionsChanged(py::wrapper::Windows::Media::ContentRestrictions::RatedContentRestrictions* self, PyObject* arg) noexcept
    {
        try
        {
            auto param0 = py::convert_to<winrt::event_token>(arg);

            self->obj.RestrictionsChanged(param0);
            Py_RETURN_NONE;
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_RatedContentRestrictions(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Media::ContentRestrictions::RatedContentRestrictions>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_RatedContentRestrictions[] = {
        { "get_browse_policy_async", (PyCFunction)RatedContentRestrictions_GetBrowsePolicyAsync, METH_VARARGS, nullptr },
        { "get_restriction_level_async", (PyCFunction)RatedContentRestrictions_GetRestrictionLevelAsync, METH_VARARGS, nullptr },
        { "request_content_access_async", (PyCFunction)RatedContentRestrictions_RequestContentAccessAsync, METH_VARARGS, nullptr },
        { "add_restrictions_changed", (PyCFunction)RatedContentRestrictions_add_RestrictionsChanged, METH_O, nullptr },
        { "remove_restrictions_changed", (PyCFunction)RatedContentRestrictions_remove_RestrictionsChanged, METH_O, nullptr },
        { "_from", (PyCFunction)_from_RatedContentRestrictions, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_RatedContentRestrictions[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_RatedContentRestrictions[] = 
    {
        { Py_tp_new, _new_RatedContentRestrictions },
        { Py_tp_dealloc, _dealloc_RatedContentRestrictions },
        { Py_tp_methods, _methods_RatedContentRestrictions },
        { Py_tp_getset, _getset_RatedContentRestrictions },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_RatedContentRestrictions =
    {
        "_winrt_Windows_Media_ContentRestrictions.RatedContentRestrictions",
        sizeof(py::wrapper::Windows::Media::ContentRestrictions::RatedContentRestrictions),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_RatedContentRestrictions
    };

    // ----- Windows.Media.ContentRestrictions Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::Media::ContentRestrictions::ContentRestrictionsBrowsePolicy>::python_type = py::register_python_type(module, _type_name_ContentRestrictionsBrowsePolicy, &_type_spec_ContentRestrictionsBrowsePolicy, bases.get());
            py::winrt_type<winrt::Windows::Media::ContentRestrictions::RatedContentDescription>::python_type = py::register_python_type(module, _type_name_RatedContentDescription, &_type_spec_RatedContentDescription, bases.get());
            py::winrt_type<winrt::Windows::Media::ContentRestrictions::RatedContentRestrictions>::python_type = py::register_python_type(module, _type_name_RatedContentRestrictions, &_type_spec_RatedContentRestrictions, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.Media.ContentRestrictions");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_Media_ContentRestrictions",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::Media::ContentRestrictions

PyMODINIT_FUNC
PyInit__winrt_Windows_Media_ContentRestrictions (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::Media::ContentRestrictions::module_def);
}
