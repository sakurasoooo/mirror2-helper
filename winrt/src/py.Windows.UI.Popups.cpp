// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.UI.Popups.h"

PyTypeObject* py::winrt_type<winrt::Windows::UI::Popups::MessageDialog>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::UI::Popups::PopupMenu>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::UI::Popups::UICommand>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::UI::Popups::UICommandSeparator>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::UI::Popups::IUICommand>::python_type;

namespace py::cpp::Windows::UI::Popups
{
    // ----- MessageDialog class --------------------
    constexpr const char* const _type_name_MessageDialog = "MessageDialog";

    static PyObject* _new_MessageDialog(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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

                winrt::Windows::UI::Popups::MessageDialog instance{ param0 };
                return py::wrap(instance, type);
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::hstring>(args, 1);

                winrt::Windows::UI::Popups::MessageDialog instance{ param0, param1 };
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

    static void _dealloc_MessageDialog(py::wrapper::Windows::UI::Popups::MessageDialog* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* MessageDialog_ShowAsync(py::wrapper::Windows::UI::Popups::MessageDialog* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.ShowAsync());
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

    static PyObject* MessageDialog_get_Title(py::wrapper::Windows::UI::Popups::MessageDialog* self, void* /*unused*/) noexcept
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

    static int MessageDialog_put_Title(py::wrapper::Windows::UI::Popups::MessageDialog* self, PyObject* arg, void* /*unused*/) noexcept
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

    static PyObject* MessageDialog_get_Options(py::wrapper::Windows::UI::Popups::MessageDialog* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Options());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int MessageDialog_put_Options(py::wrapper::Windows::UI::Popups::MessageDialog* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::UI::Popups::MessageDialogOptions>(arg);

            self->obj.Options(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* MessageDialog_get_DefaultCommandIndex(py::wrapper::Windows::UI::Popups::MessageDialog* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.DefaultCommandIndex());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int MessageDialog_put_DefaultCommandIndex(py::wrapper::Windows::UI::Popups::MessageDialog* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<uint32_t>(arg);

            self->obj.DefaultCommandIndex(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* MessageDialog_get_Content(py::wrapper::Windows::UI::Popups::MessageDialog* self, void* /*unused*/) noexcept
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

    static int MessageDialog_put_Content(py::wrapper::Windows::UI::Popups::MessageDialog* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::hstring>(arg);

            self->obj.Content(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* MessageDialog_get_CancelCommandIndex(py::wrapper::Windows::UI::Popups::MessageDialog* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.CancelCommandIndex());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int MessageDialog_put_CancelCommandIndex(py::wrapper::Windows::UI::Popups::MessageDialog* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<uint32_t>(arg);

            self->obj.CancelCommandIndex(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* MessageDialog_get_Commands(py::wrapper::Windows::UI::Popups::MessageDialog* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Commands());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_MessageDialog(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::Popups::MessageDialog>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_MessageDialog[] = {
        { "show_async", (PyCFunction)MessageDialog_ShowAsync, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_MessageDialog, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_MessageDialog[] = {
        { const_cast<char*>("title"), (getter)MessageDialog_get_Title, (setter)MessageDialog_put_Title, nullptr, nullptr },
        { const_cast<char*>("options"), (getter)MessageDialog_get_Options, (setter)MessageDialog_put_Options, nullptr, nullptr },
        { const_cast<char*>("default_command_index"), (getter)MessageDialog_get_DefaultCommandIndex, (setter)MessageDialog_put_DefaultCommandIndex, nullptr, nullptr },
        { const_cast<char*>("content"), (getter)MessageDialog_get_Content, (setter)MessageDialog_put_Content, nullptr, nullptr },
        { const_cast<char*>("cancel_command_index"), (getter)MessageDialog_get_CancelCommandIndex, (setter)MessageDialog_put_CancelCommandIndex, nullptr, nullptr },
        { const_cast<char*>("commands"), (getter)MessageDialog_get_Commands, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_MessageDialog[] = 
    {
        { Py_tp_new, _new_MessageDialog },
        { Py_tp_dealloc, _dealloc_MessageDialog },
        { Py_tp_methods, _methods_MessageDialog },
        { Py_tp_getset, _getset_MessageDialog },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_MessageDialog =
    {
        "_winrt_Windows_UI_Popups.MessageDialog",
        sizeof(py::wrapper::Windows::UI::Popups::MessageDialog),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_MessageDialog
    };

    // ----- PopupMenu class --------------------
    constexpr const char* const _type_name_PopupMenu = "PopupMenu";

    static PyObject* _new_PopupMenu(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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
                winrt::Windows::UI::Popups::PopupMenu instance{  };
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

    static void _dealloc_PopupMenu(py::wrapper::Windows::UI::Popups::PopupMenu* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* PopupMenu_ShowAsync(py::wrapper::Windows::UI::Popups::PopupMenu* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Foundation::Point>(args, 0);

                return py::convert(self->obj.ShowAsync(param0));
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

    static PyObject* PopupMenu_ShowForSelectionAsync(py::wrapper::Windows::UI::Popups::PopupMenu* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Foundation::Rect>(args, 0);

                return py::convert(self->obj.ShowForSelectionAsync(param0));
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Foundation::Rect>(args, 0);
                auto param1 = py::convert_to<winrt::Windows::UI::Popups::Placement>(args, 1);

                return py::convert(self->obj.ShowForSelectionAsync(param0, param1));
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

    static PyObject* PopupMenu_get_Commands(py::wrapper::Windows::UI::Popups::PopupMenu* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Commands());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_PopupMenu(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::Popups::PopupMenu>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_PopupMenu[] = {
        { "show_async", (PyCFunction)PopupMenu_ShowAsync, METH_VARARGS, nullptr },
        { "show_for_selection_async", (PyCFunction)PopupMenu_ShowForSelectionAsync, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_PopupMenu, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_PopupMenu[] = {
        { const_cast<char*>("commands"), (getter)PopupMenu_get_Commands, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_PopupMenu[] = 
    {
        { Py_tp_new, _new_PopupMenu },
        { Py_tp_dealloc, _dealloc_PopupMenu },
        { Py_tp_methods, _methods_PopupMenu },
        { Py_tp_getset, _getset_PopupMenu },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_PopupMenu =
    {
        "_winrt_Windows_UI_Popups.PopupMenu",
        sizeof(py::wrapper::Windows::UI::Popups::PopupMenu),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_PopupMenu
    };

    // ----- UICommand class --------------------
    constexpr const char* const _type_name_UICommand = "UICommand";

    static PyObject* _new_UICommand(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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

                winrt::Windows::UI::Popups::UICommand instance{ param0 };
                return py::wrap(instance, type);
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else if (arg_count == 2)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::Windows::UI::Popups::UICommandInvokedHandler>(args, 1);

                winrt::Windows::UI::Popups::UICommand instance{ param0, param1 };
                return py::wrap(instance, type);
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else if (arg_count == 3)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);
                auto param1 = py::convert_to<winrt::Windows::UI::Popups::UICommandInvokedHandler>(args, 1);
                auto param2 = py::convert_to<winrt::Windows::Foundation::IInspectable>(args, 2);

                winrt::Windows::UI::Popups::UICommand instance{ param0, param1, param2 };
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
                winrt::Windows::UI::Popups::UICommand instance{  };
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

    static void _dealloc_UICommand(py::wrapper::Windows::UI::Popups::UICommand* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* UICommand_get_Label(py::wrapper::Windows::UI::Popups::UICommand* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Label());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int UICommand_put_Label(py::wrapper::Windows::UI::Popups::UICommand* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::hstring>(arg);

            self->obj.Label(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* UICommand_get_Invoked(py::wrapper::Windows::UI::Popups::UICommand* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Invoked());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int UICommand_put_Invoked(py::wrapper::Windows::UI::Popups::UICommand* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::UI::Popups::UICommandInvokedHandler>(arg);

            self->obj.Invoked(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* UICommand_get_Id(py::wrapper::Windows::UI::Popups::UICommand* self, void* /*unused*/) noexcept
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

    static int UICommand_put_Id(py::wrapper::Windows::UI::Popups::UICommand* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);

            self->obj.Id(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* _from_UICommand(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::Popups::UICommand>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_UICommand[] = {
        { "_from", (PyCFunction)_from_UICommand, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_UICommand[] = {
        { const_cast<char*>("label"), (getter)UICommand_get_Label, (setter)UICommand_put_Label, nullptr, nullptr },
        { const_cast<char*>("invoked"), (getter)UICommand_get_Invoked, (setter)UICommand_put_Invoked, nullptr, nullptr },
        { const_cast<char*>("id"), (getter)UICommand_get_Id, (setter)UICommand_put_Id, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_UICommand[] = 
    {
        { Py_tp_new, _new_UICommand },
        { Py_tp_dealloc, _dealloc_UICommand },
        { Py_tp_methods, _methods_UICommand },
        { Py_tp_getset, _getset_UICommand },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_UICommand =
    {
        "_winrt_Windows_UI_Popups.UICommand",
        sizeof(py::wrapper::Windows::UI::Popups::UICommand),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_UICommand
    };

    // ----- UICommandSeparator class --------------------
    constexpr const char* const _type_name_UICommandSeparator = "UICommandSeparator";

    static PyObject* _new_UICommandSeparator(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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
                winrt::Windows::UI::Popups::UICommandSeparator instance{  };
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

    static void _dealloc_UICommandSeparator(py::wrapper::Windows::UI::Popups::UICommandSeparator* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* UICommandSeparator_get_Label(py::wrapper::Windows::UI::Popups::UICommandSeparator* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Label());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int UICommandSeparator_put_Label(py::wrapper::Windows::UI::Popups::UICommandSeparator* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::hstring>(arg);

            self->obj.Label(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* UICommandSeparator_get_Invoked(py::wrapper::Windows::UI::Popups::UICommandSeparator* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Invoked());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int UICommandSeparator_put_Invoked(py::wrapper::Windows::UI::Popups::UICommandSeparator* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::UI::Popups::UICommandInvokedHandler>(arg);

            self->obj.Invoked(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* UICommandSeparator_get_Id(py::wrapper::Windows::UI::Popups::UICommandSeparator* self, void* /*unused*/) noexcept
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

    static int UICommandSeparator_put_Id(py::wrapper::Windows::UI::Popups::UICommandSeparator* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);

            self->obj.Id(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* _from_UICommandSeparator(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::Popups::UICommandSeparator>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_UICommandSeparator[] = {
        { "_from", (PyCFunction)_from_UICommandSeparator, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_UICommandSeparator[] = {
        { const_cast<char*>("label"), (getter)UICommandSeparator_get_Label, (setter)UICommandSeparator_put_Label, nullptr, nullptr },
        { const_cast<char*>("invoked"), (getter)UICommandSeparator_get_Invoked, (setter)UICommandSeparator_put_Invoked, nullptr, nullptr },
        { const_cast<char*>("id"), (getter)UICommandSeparator_get_Id, (setter)UICommandSeparator_put_Id, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_UICommandSeparator[] = 
    {
        { Py_tp_new, _new_UICommandSeparator },
        { Py_tp_dealloc, _dealloc_UICommandSeparator },
        { Py_tp_methods, _methods_UICommandSeparator },
        { Py_tp_getset, _getset_UICommandSeparator },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_UICommandSeparator =
    {
        "_winrt_Windows_UI_Popups.UICommandSeparator",
        sizeof(py::wrapper::Windows::UI::Popups::UICommandSeparator),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_UICommandSeparator
    };

    // ----- IUICommand interface --------------------
    constexpr const char* const _type_name_IUICommand = "IUICommand";

    static PyObject* _new_IUICommand(PyTypeObject* /* unused */, PyObject* /* unused */, PyObject* /* unused */)
    {
        py::set_invalid_activation_error(_type_name_IUICommand);
        return nullptr;
    }

    static void _dealloc_IUICommand(py::wrapper::Windows::UI::Popups::IUICommand* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* IUICommand_get_Id(py::wrapper::Windows::UI::Popups::IUICommand* self, void* /*unused*/) noexcept
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

    static int IUICommand_put_Id(py::wrapper::Windows::UI::Popups::IUICommand* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);

            self->obj.Id(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* IUICommand_get_Invoked(py::wrapper::Windows::UI::Popups::IUICommand* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Invoked());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int IUICommand_put_Invoked(py::wrapper::Windows::UI::Popups::IUICommand* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::UI::Popups::UICommandInvokedHandler>(arg);

            self->obj.Invoked(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* IUICommand_get_Label(py::wrapper::Windows::UI::Popups::IUICommand* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Label());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int IUICommand_put_Label(py::wrapper::Windows::UI::Popups::IUICommand* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::hstring>(arg);

            self->obj.Label(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* _from_IUICommand(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::UI::Popups::IUICommand>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_IUICommand[] = {
        { "_from", (PyCFunction)_from_IUICommand, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_IUICommand[] = {
        { const_cast<char*>("id"), (getter)IUICommand_get_Id, (setter)IUICommand_put_Id, nullptr, nullptr },
        { const_cast<char*>("invoked"), (getter)IUICommand_get_Invoked, (setter)IUICommand_put_Invoked, nullptr, nullptr },
        { const_cast<char*>("label"), (getter)IUICommand_get_Label, (setter)IUICommand_put_Label, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_IUICommand[] = 
    {
        { Py_tp_new, _new_IUICommand },
        { Py_tp_dealloc, _dealloc_IUICommand },
        { Py_tp_methods, _methods_IUICommand },
        { Py_tp_getset, _getset_IUICommand },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_IUICommand =
    {
        "_winrt_Windows_UI_Popups.IUICommand",
        sizeof(py::wrapper::Windows::UI::Popups::IUICommand),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_IUICommand
    };

    // ----- Windows.UI.Popups Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::UI::Popups::MessageDialog>::python_type = py::register_python_type(module, _type_name_MessageDialog, &_type_spec_MessageDialog, bases.get());
            py::winrt_type<winrt::Windows::UI::Popups::PopupMenu>::python_type = py::register_python_type(module, _type_name_PopupMenu, &_type_spec_PopupMenu, bases.get());
            py::winrt_type<winrt::Windows::UI::Popups::UICommand>::python_type = py::register_python_type(module, _type_name_UICommand, &_type_spec_UICommand, bases.get());
            py::winrt_type<winrt::Windows::UI::Popups::UICommandSeparator>::python_type = py::register_python_type(module, _type_name_UICommandSeparator, &_type_spec_UICommandSeparator, bases.get());
            py::winrt_type<winrt::Windows::UI::Popups::IUICommand>::python_type = py::register_python_type(module, _type_name_IUICommand, &_type_spec_IUICommand, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.UI.Popups");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_UI_Popups",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::UI::Popups

PyMODINIT_FUNC
PyInit__winrt_Windows_UI_Popups (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::UI::Popups::module_def);
}