// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.System.Display.h"

PyTypeObject* py::winrt_type<winrt::Windows::System::Display::DisplayRequest>::python_type;

namespace py::cpp::Windows::System::Display
{
    // ----- DisplayRequest class --------------------
    constexpr const char* const _type_name_DisplayRequest = "DisplayRequest";

    static PyObject* _new_DisplayRequest(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
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
                winrt::Windows::System::Display::DisplayRequest instance{  };
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

    static void _dealloc_DisplayRequest(py::wrapper::Windows::System::Display::DisplayRequest* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* DisplayRequest_RequestActive(py::wrapper::Windows::System::Display::DisplayRequest* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.RequestActive();
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

    static PyObject* DisplayRequest_RequestRelease(py::wrapper::Windows::System::Display::DisplayRequest* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.RequestRelease();
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

    static PyObject* _from_DisplayRequest(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::System::Display::DisplayRequest>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_DisplayRequest[] = {
        { "request_active", (PyCFunction)DisplayRequest_RequestActive, METH_VARARGS, nullptr },
        { "request_release", (PyCFunction)DisplayRequest_RequestRelease, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_DisplayRequest, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_DisplayRequest[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_DisplayRequest[] = 
    {
        { Py_tp_new, _new_DisplayRequest },
        { Py_tp_dealloc, _dealloc_DisplayRequest },
        { Py_tp_methods, _methods_DisplayRequest },
        { Py_tp_getset, _getset_DisplayRequest },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_DisplayRequest =
    {
        "_winrt_Windows_System_Display.DisplayRequest",
        sizeof(py::wrapper::Windows::System::Display::DisplayRequest),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_DisplayRequest
    };

    // ----- Windows.System.Display Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::System::Display::DisplayRequest>::python_type = py::register_python_type(module, _type_name_DisplayRequest, &_type_spec_DisplayRequest, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.System.Display");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_System_Display",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::System::Display

PyMODINIT_FUNC
PyInit__winrt_Windows_System_Display (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::System::Display::module_def);
}