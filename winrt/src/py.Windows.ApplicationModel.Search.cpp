// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.ApplicationModel.Search.h"

PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::Search::LocalContentSuggestionSettings>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchPaneQueryLinguisticDetails>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchSuggestionCollection>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchSuggestionsRequest>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchSuggestionsRequestDeferral>::python_type;

namespace py::cpp::Windows::ApplicationModel::Search
{
    // ----- LocalContentSuggestionSettings class --------------------
    constexpr const char* const _type_name_LocalContentSuggestionSettings = "LocalContentSuggestionSettings";

    static PyObject* _new_LocalContentSuggestionSettings(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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
                winrt::Windows::ApplicationModel::Search::LocalContentSuggestionSettings instance{  };
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

    static void _dealloc_LocalContentSuggestionSettings(py::wrapper::Windows::ApplicationModel::Search::LocalContentSuggestionSettings* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* LocalContentSuggestionSettings_get_Enabled(py::wrapper::Windows::ApplicationModel::Search::LocalContentSuggestionSettings* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Enabled());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int LocalContentSuggestionSettings_put_Enabled(py::wrapper::Windows::ApplicationModel::Search::LocalContentSuggestionSettings* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<bool>(arg);

            self->obj.Enabled(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* LocalContentSuggestionSettings_get_AqsFilter(py::wrapper::Windows::ApplicationModel::Search::LocalContentSuggestionSettings* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.AqsFilter());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int LocalContentSuggestionSettings_put_AqsFilter(py::wrapper::Windows::ApplicationModel::Search::LocalContentSuggestionSettings* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::hstring>(arg);

            self->obj.AqsFilter(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* LocalContentSuggestionSettings_get_Locations(py::wrapper::Windows::ApplicationModel::Search::LocalContentSuggestionSettings* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Locations());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* LocalContentSuggestionSettings_get_PropertiesToMatch(py::wrapper::Windows::ApplicationModel::Search::LocalContentSuggestionSettings* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.PropertiesToMatch());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_LocalContentSuggestionSettings(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::ApplicationModel::Search::LocalContentSuggestionSettings>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_LocalContentSuggestionSettings[] = {
        { "_from", (PyCFunction)_from_LocalContentSuggestionSettings, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_LocalContentSuggestionSettings[] = {
        { const_cast<char*>("enabled"), (getter)LocalContentSuggestionSettings_get_Enabled, (setter)LocalContentSuggestionSettings_put_Enabled, nullptr, nullptr },
        { const_cast<char*>("aqs_filter"), (getter)LocalContentSuggestionSettings_get_AqsFilter, (setter)LocalContentSuggestionSettings_put_AqsFilter, nullptr, nullptr },
        { const_cast<char*>("locations"), (getter)LocalContentSuggestionSettings_get_Locations, nullptr, nullptr, nullptr },
        { const_cast<char*>("properties_to_match"), (getter)LocalContentSuggestionSettings_get_PropertiesToMatch, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_LocalContentSuggestionSettings[] = 
    {
        { Py_tp_new, _new_LocalContentSuggestionSettings },
        { Py_tp_dealloc, _dealloc_LocalContentSuggestionSettings },
        { Py_tp_methods, _methods_LocalContentSuggestionSettings },
        { Py_tp_getset, _getset_LocalContentSuggestionSettings },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_LocalContentSuggestionSettings =
    {
        "_winrt_Windows_ApplicationModel_Search.LocalContentSuggestionSettings",
        sizeof(py::wrapper::Windows::ApplicationModel::Search::LocalContentSuggestionSettings),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_LocalContentSuggestionSettings
    };

    // ----- SearchPaneQueryLinguisticDetails class --------------------
    constexpr const char* const _type_name_SearchPaneQueryLinguisticDetails = "SearchPaneQueryLinguisticDetails";

    static PyObject* _new_SearchPaneQueryLinguisticDetails(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SearchPaneQueryLinguisticDetails);
        return nullptr;
    }

    static void _dealloc_SearchPaneQueryLinguisticDetails(py::wrapper::Windows::ApplicationModel::Search::SearchPaneQueryLinguisticDetails* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SearchPaneQueryLinguisticDetails_get_QueryTextAlternatives(py::wrapper::Windows::ApplicationModel::Search::SearchPaneQueryLinguisticDetails* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.QueryTextAlternatives());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SearchPaneQueryLinguisticDetails_get_QueryTextCompositionLength(py::wrapper::Windows::ApplicationModel::Search::SearchPaneQueryLinguisticDetails* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.QueryTextCompositionLength());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SearchPaneQueryLinguisticDetails_get_QueryTextCompositionStart(py::wrapper::Windows::ApplicationModel::Search::SearchPaneQueryLinguisticDetails* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.QueryTextCompositionStart());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SearchPaneQueryLinguisticDetails(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::ApplicationModel::Search::SearchPaneQueryLinguisticDetails>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SearchPaneQueryLinguisticDetails[] = {
        { "_from", (PyCFunction)_from_SearchPaneQueryLinguisticDetails, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SearchPaneQueryLinguisticDetails[] = {
        { const_cast<char*>("query_text_alternatives"), (getter)SearchPaneQueryLinguisticDetails_get_QueryTextAlternatives, nullptr, nullptr, nullptr },
        { const_cast<char*>("query_text_composition_length"), (getter)SearchPaneQueryLinguisticDetails_get_QueryTextCompositionLength, nullptr, nullptr, nullptr },
        { const_cast<char*>("query_text_composition_start"), (getter)SearchPaneQueryLinguisticDetails_get_QueryTextCompositionStart, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SearchPaneQueryLinguisticDetails[] = 
    {
        { Py_tp_new, _new_SearchPaneQueryLinguisticDetails },
        { Py_tp_dealloc, _dealloc_SearchPaneQueryLinguisticDetails },
        { Py_tp_methods, _methods_SearchPaneQueryLinguisticDetails },
        { Py_tp_getset, _getset_SearchPaneQueryLinguisticDetails },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SearchPaneQueryLinguisticDetails =
    {
        "_winrt_Windows_ApplicationModel_Search.SearchPaneQueryLinguisticDetails",
        sizeof(py::wrapper::Windows::ApplicationModel::Search::SearchPaneQueryLinguisticDetails),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SearchPaneQueryLinguisticDetails
    };

    // ----- SearchQueryLinguisticDetails class --------------------
    constexpr const char* const _type_name_SearchQueryLinguisticDetails = "SearchQueryLinguisticDetails";

    static PyObject* _new_SearchQueryLinguisticDetails(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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
                auto param0 = py::convert_to<winrt::Windows::Foundation::Collections::IIterable<winrt::hstring>>(args, 0);
                auto param1 = py::convert_to<uint32_t>(args, 1);
                auto param2 = py::convert_to<uint32_t>(args, 2);

                winrt::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails instance{ param0, param1, param2 };
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

    static void _dealloc_SearchQueryLinguisticDetails(py::wrapper::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SearchQueryLinguisticDetails_get_QueryTextAlternatives(py::wrapper::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.QueryTextAlternatives());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SearchQueryLinguisticDetails_get_QueryTextCompositionLength(py::wrapper::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.QueryTextCompositionLength());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SearchQueryLinguisticDetails_get_QueryTextCompositionStart(py::wrapper::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.QueryTextCompositionStart());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SearchQueryLinguisticDetails(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SearchQueryLinguisticDetails[] = {
        { "_from", (PyCFunction)_from_SearchQueryLinguisticDetails, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SearchQueryLinguisticDetails[] = {
        { const_cast<char*>("query_text_alternatives"), (getter)SearchQueryLinguisticDetails_get_QueryTextAlternatives, nullptr, nullptr, nullptr },
        { const_cast<char*>("query_text_composition_length"), (getter)SearchQueryLinguisticDetails_get_QueryTextCompositionLength, nullptr, nullptr, nullptr },
        { const_cast<char*>("query_text_composition_start"), (getter)SearchQueryLinguisticDetails_get_QueryTextCompositionStart, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SearchQueryLinguisticDetails[] = 
    {
        { Py_tp_new, _new_SearchQueryLinguisticDetails },
        { Py_tp_dealloc, _dealloc_SearchQueryLinguisticDetails },
        { Py_tp_methods, _methods_SearchQueryLinguisticDetails },
        { Py_tp_getset, _getset_SearchQueryLinguisticDetails },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SearchQueryLinguisticDetails =
    {
        "_winrt_Windows_ApplicationModel_Search.SearchQueryLinguisticDetails",
        sizeof(py::wrapper::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SearchQueryLinguisticDetails
    };

    // ----- SearchSuggestionCollection class --------------------
    constexpr const char* const _type_name_SearchSuggestionCollection = "SearchSuggestionCollection";

    static PyObject* _new_SearchSuggestionCollection(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SearchSuggestionCollection);
        return nullptr;
    }

    static void _dealloc_SearchSuggestionCollection(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionCollection* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SearchSuggestionCollection_AppendQuerySuggestion(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionCollection* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);

                self->obj.AppendQuerySuggestion(param0);
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

    static PyObject* SearchSuggestionCollection_AppendQuerySuggestions(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionCollection* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Foundation::Collections::IIterable<winrt::hstring>>(args, 0);

                self->obj.AppendQuerySuggestions(param0);
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

    static PyObject* SearchSuggestionCollection_AppendResultSuggestion(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionCollection* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 5)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::hstring>(args, 1);
                auto param2 = py::convert_to<winrt::hstring>(args, 2);
                auto param3 = py::convert_to<winrt::Windows::Storage::Streams::IRandomAccessStreamReference>(args, 3);
                auto param4 = py::convert_to<winrt::hstring>(args, 4);

                self->obj.AppendResultSuggestion(param0, param1, param2, param3, param4);
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

    static PyObject* SearchSuggestionCollection_AppendSearchSeparator(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionCollection* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);

                self->obj.AppendSearchSeparator(param0);
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

    static PyObject* SearchSuggestionCollection_get_Size(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionCollection* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Size());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SearchSuggestionCollection(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::ApplicationModel::Search::SearchSuggestionCollection>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SearchSuggestionCollection[] = {
        { "append_query_suggestion", (PyCFunction)SearchSuggestionCollection_AppendQuerySuggestion, METH_VARARGS, nullptr },
        { "append_query_suggestions", (PyCFunction)SearchSuggestionCollection_AppendQuerySuggestions, METH_VARARGS, nullptr },
        { "append_result_suggestion", (PyCFunction)SearchSuggestionCollection_AppendResultSuggestion, METH_VARARGS, nullptr },
        { "append_search_separator", (PyCFunction)SearchSuggestionCollection_AppendSearchSeparator, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_SearchSuggestionCollection, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SearchSuggestionCollection[] = {
        { const_cast<char*>("size"), (getter)SearchSuggestionCollection_get_Size, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SearchSuggestionCollection[] = 
    {
        { Py_tp_new, _new_SearchSuggestionCollection },
        { Py_tp_dealloc, _dealloc_SearchSuggestionCollection },
        { Py_tp_methods, _methods_SearchSuggestionCollection },
        { Py_tp_getset, _getset_SearchSuggestionCollection },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SearchSuggestionCollection =
    {
        "_winrt_Windows_ApplicationModel_Search.SearchSuggestionCollection",
        sizeof(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionCollection),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SearchSuggestionCollection
    };

    // ----- SearchSuggestionsRequest class --------------------
    constexpr const char* const _type_name_SearchSuggestionsRequest = "SearchSuggestionsRequest";

    static PyObject* _new_SearchSuggestionsRequest(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SearchSuggestionsRequest);
        return nullptr;
    }

    static void _dealloc_SearchSuggestionsRequest(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionsRequest* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SearchSuggestionsRequest_GetDeferral(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionsRequest* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetDeferral());
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

    static PyObject* SearchSuggestionsRequest_get_IsCanceled(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionsRequest* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsCanceled());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SearchSuggestionsRequest_get_SearchSuggestionCollection(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionsRequest* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SearchSuggestionCollection());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SearchSuggestionsRequest(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::ApplicationModel::Search::SearchSuggestionsRequest>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SearchSuggestionsRequest[] = {
        { "get_deferral", (PyCFunction)SearchSuggestionsRequest_GetDeferral, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_SearchSuggestionsRequest, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SearchSuggestionsRequest[] = {
        { const_cast<char*>("is_canceled"), (getter)SearchSuggestionsRequest_get_IsCanceled, nullptr, nullptr, nullptr },
        { const_cast<char*>("search_suggestion_collection"), (getter)SearchSuggestionsRequest_get_SearchSuggestionCollection, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SearchSuggestionsRequest[] = 
    {
        { Py_tp_new, _new_SearchSuggestionsRequest },
        { Py_tp_dealloc, _dealloc_SearchSuggestionsRequest },
        { Py_tp_methods, _methods_SearchSuggestionsRequest },
        { Py_tp_getset, _getset_SearchSuggestionsRequest },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SearchSuggestionsRequest =
    {
        "_winrt_Windows_ApplicationModel_Search.SearchSuggestionsRequest",
        sizeof(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionsRequest),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SearchSuggestionsRequest
    };

    // ----- SearchSuggestionsRequestDeferral class --------------------
    constexpr const char* const _type_name_SearchSuggestionsRequestDeferral = "SearchSuggestionsRequestDeferral";

    static PyObject* _new_SearchSuggestionsRequestDeferral(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SearchSuggestionsRequestDeferral);
        return nullptr;
    }

    static void _dealloc_SearchSuggestionsRequestDeferral(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionsRequestDeferral* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SearchSuggestionsRequestDeferral_Complete(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionsRequestDeferral* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.Complete();
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

    static PyObject* _from_SearchSuggestionsRequestDeferral(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::ApplicationModel::Search::SearchSuggestionsRequestDeferral>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SearchSuggestionsRequestDeferral[] = {
        { "complete", (PyCFunction)SearchSuggestionsRequestDeferral_Complete, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_SearchSuggestionsRequestDeferral, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SearchSuggestionsRequestDeferral[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_SearchSuggestionsRequestDeferral[] = 
    {
        { Py_tp_new, _new_SearchSuggestionsRequestDeferral },
        { Py_tp_dealloc, _dealloc_SearchSuggestionsRequestDeferral },
        { Py_tp_methods, _methods_SearchSuggestionsRequestDeferral },
        { Py_tp_getset, _getset_SearchSuggestionsRequestDeferral },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SearchSuggestionsRequestDeferral =
    {
        "_winrt_Windows_ApplicationModel_Search.SearchSuggestionsRequestDeferral",
        sizeof(py::wrapper::Windows::ApplicationModel::Search::SearchSuggestionsRequestDeferral),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SearchSuggestionsRequestDeferral
    };

    // ----- Windows.ApplicationModel.Search Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::ApplicationModel::Search::LocalContentSuggestionSettings>::python_type = py::register_python_type(module, _type_name_LocalContentSuggestionSettings, &_type_spec_LocalContentSuggestionSettings, bases.get());
            py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchPaneQueryLinguisticDetails>::python_type = py::register_python_type(module, _type_name_SearchPaneQueryLinguisticDetails, &_type_spec_SearchPaneQueryLinguisticDetails, bases.get());
            py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchQueryLinguisticDetails>::python_type = py::register_python_type(module, _type_name_SearchQueryLinguisticDetails, &_type_spec_SearchQueryLinguisticDetails, bases.get());
            py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchSuggestionCollection>::python_type = py::register_python_type(module, _type_name_SearchSuggestionCollection, &_type_spec_SearchSuggestionCollection, bases.get());
            py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchSuggestionsRequest>::python_type = py::register_python_type(module, _type_name_SearchSuggestionsRequest, &_type_spec_SearchSuggestionsRequest, bases.get());
            py::winrt_type<winrt::Windows::ApplicationModel::Search::SearchSuggestionsRequestDeferral>::python_type = py::register_python_type(module, _type_name_SearchSuggestionsRequestDeferral, &_type_spec_SearchSuggestionsRequestDeferral, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.ApplicationModel.Search");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_ApplicationModel_Search",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::ApplicationModel::Search

PyMODINIT_FUNC
PyInit__winrt_Windows_ApplicationModel_Search (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::ApplicationModel::Search::module_def);
}