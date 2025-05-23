// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.Devices.Pwm.h"

PyTypeObject* py::winrt_type<winrt::Windows::Devices::Pwm::PwmController>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Devices::Pwm::PwmPin>::python_type;

namespace py::cpp::Windows::Devices::Pwm
{
    // ----- PwmController class --------------------
    constexpr const char* const _type_name_PwmController = "PwmController";

    static PyObject* _new_PwmController(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_PwmController);
        return nullptr;
    }

    static void _dealloc_PwmController(py::wrapper::Windows::Devices::Pwm::PwmController* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* PwmController_FromIdAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);

                return py::convert(winrt::Windows::Devices::Pwm::PwmController::FromIdAsync(param0));
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

    static PyObject* PwmController_GetControllersAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Devices::Pwm::Provider::IPwmProvider>(args, 0);

                return py::convert(winrt::Windows::Devices::Pwm::PwmController::GetControllersAsync(param0));
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

    static PyObject* PwmController_GetDefaultAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::Devices::Pwm::PwmController::GetDefaultAsync());
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

    static PyObject* PwmController_GetDeviceSelector(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::Devices::Pwm::PwmController::GetDeviceSelector());
            }
            catch (...)
            {
                py::to_PyErr();
                return nullptr;
            }
        }
        else if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);

                return py::convert(winrt::Windows::Devices::Pwm::PwmController::GetDeviceSelector(param0));
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

    static PyObject* PwmController_OpenPin(py::wrapper::Windows::Devices::Pwm::PwmController* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<int32_t>(args, 0);

                return py::convert(self->obj.OpenPin(param0));
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

    static PyObject* PwmController_SetDesiredFrequency(py::wrapper::Windows::Devices::Pwm::PwmController* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<double>(args, 0);

                return py::convert(self->obj.SetDesiredFrequency(param0));
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

    static PyObject* PwmController_get_ActualFrequency(py::wrapper::Windows::Devices::Pwm::PwmController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.ActualFrequency());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PwmController_get_MaxFrequency(py::wrapper::Windows::Devices::Pwm::PwmController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.MaxFrequency());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PwmController_get_MinFrequency(py::wrapper::Windows::Devices::Pwm::PwmController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.MinFrequency());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PwmController_get_PinCount(py::wrapper::Windows::Devices::Pwm::PwmController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.PinCount());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_PwmController(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Pwm::PwmController>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_PwmController[] = {
        { "from_id_async", (PyCFunction)PwmController_FromIdAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "get_controllers_async", (PyCFunction)PwmController_GetControllersAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "get_default_async", (PyCFunction)PwmController_GetDefaultAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "get_device_selector", (PyCFunction)PwmController_GetDeviceSelector, METH_VARARGS | METH_STATIC, nullptr },
        { "open_pin", (PyCFunction)PwmController_OpenPin, METH_VARARGS, nullptr },
        { "set_desired_frequency", (PyCFunction)PwmController_SetDesiredFrequency, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_PwmController, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_PwmController[] = {
        { const_cast<char*>("actual_frequency"), (getter)PwmController_get_ActualFrequency, nullptr, nullptr, nullptr },
        { const_cast<char*>("max_frequency"), (getter)PwmController_get_MaxFrequency, nullptr, nullptr, nullptr },
        { const_cast<char*>("min_frequency"), (getter)PwmController_get_MinFrequency, nullptr, nullptr, nullptr },
        { const_cast<char*>("pin_count"), (getter)PwmController_get_PinCount, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_PwmController[] = 
    {
        { Py_tp_new, _new_PwmController },
        { Py_tp_dealloc, _dealloc_PwmController },
        { Py_tp_methods, _methods_PwmController },
        { Py_tp_getset, _getset_PwmController },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_PwmController =
    {
        "_winrt_Windows_Devices_Pwm.PwmController",
        sizeof(py::wrapper::Windows::Devices::Pwm::PwmController),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_PwmController
    };

    // ----- PwmPin class --------------------
    constexpr const char* const _type_name_PwmPin = "PwmPin";

    static PyObject* _new_PwmPin(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_PwmPin);
        return nullptr;
    }

    static void _dealloc_PwmPin(py::wrapper::Windows::Devices::Pwm::PwmPin* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* PwmPin_Close(py::wrapper::Windows::Devices::Pwm::PwmPin* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.Close();
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

    static PyObject* PwmPin_GetActiveDutyCyclePercentage(py::wrapper::Windows::Devices::Pwm::PwmPin* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(self->obj.GetActiveDutyCyclePercentage());
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

    static PyObject* PwmPin_SetActiveDutyCyclePercentage(py::wrapper::Windows::Devices::Pwm::PwmPin* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<double>(args, 0);

                self->obj.SetActiveDutyCyclePercentage(param0);
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

    static PyObject* PwmPin_Start(py::wrapper::Windows::Devices::Pwm::PwmPin* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.Start();
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

    static PyObject* PwmPin_Stop(py::wrapper::Windows::Devices::Pwm::PwmPin* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.Stop();
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

    static PyObject* PwmPin_get_Polarity(py::wrapper::Windows::Devices::Pwm::PwmPin* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Polarity());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static int PwmPin_put_Polarity(py::wrapper::Windows::Devices::Pwm::PwmPin* self, PyObject* arg, void* /*unused*/) noexcept
    {
        if (arg == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "property delete not supported");
            return -1;
        }

        try
        {
            auto param0 = py::convert_to<winrt::Windows::Devices::Pwm::PwmPulsePolarity>(arg);

            self->obj.Polarity(param0);
            return 0;
        }
        catch (...)
        {
            py::to_PyErr();
            return -1;
        }
    }

    static PyObject* PwmPin_get_Controller(py::wrapper::Windows::Devices::Pwm::PwmPin* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Controller());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* PwmPin_get_IsStarted(py::wrapper::Windows::Devices::Pwm::PwmPin* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsStarted());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_PwmPin(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Pwm::PwmPin>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _enter_PwmPin(py::wrapper::Windows::Devices::Pwm::PwmPin* self) noexcept
    {
        Py_INCREF(self);
        return (PyObject*)self;
    }

    static PyObject* _exit_PwmPin(py::wrapper::Windows::Devices::Pwm::PwmPin* self) noexcept
    {
        try
        {
            self->obj.Close();
            Py_RETURN_FALSE;
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_PwmPin[] = {
        { "close", (PyCFunction)PwmPin_Close, METH_VARARGS, nullptr },
        { "get_active_duty_cycle_percentage", (PyCFunction)PwmPin_GetActiveDutyCyclePercentage, METH_VARARGS, nullptr },
        { "set_active_duty_cycle_percentage", (PyCFunction)PwmPin_SetActiveDutyCyclePercentage, METH_VARARGS, nullptr },
        { "start", (PyCFunction)PwmPin_Start, METH_VARARGS, nullptr },
        { "stop", (PyCFunction)PwmPin_Stop, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_PwmPin, METH_O | METH_STATIC, nullptr },
        { "__enter__", (PyCFunction)_enter_PwmPin, METH_NOARGS, nullptr },
        { "__exit__",  (PyCFunction)_exit_PwmPin, METH_VARARGS, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_PwmPin[] = {
        { const_cast<char*>("polarity"), (getter)PwmPin_get_Polarity, (setter)PwmPin_put_Polarity, nullptr, nullptr },
        { const_cast<char*>("controller"), (getter)PwmPin_get_Controller, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_started"), (getter)PwmPin_get_IsStarted, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_PwmPin[] = 
    {
        { Py_tp_new, _new_PwmPin },
        { Py_tp_dealloc, _dealloc_PwmPin },
        { Py_tp_methods, _methods_PwmPin },
        { Py_tp_getset, _getset_PwmPin },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_PwmPin =
    {
        "_winrt_Windows_Devices_Pwm.PwmPin",
        sizeof(py::wrapper::Windows::Devices::Pwm::PwmPin),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_PwmPin
    };

    // ----- Windows.Devices.Pwm Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::Devices::Pwm::PwmController>::python_type = py::register_python_type(module, _type_name_PwmController, &_type_spec_PwmController, bases.get());
            py::winrt_type<winrt::Windows::Devices::Pwm::PwmPin>::python_type = py::register_python_type(module, _type_name_PwmPin, &_type_spec_PwmPin, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.Devices.Pwm");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_Devices_Pwm",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::Devices::Pwm

PyMODINIT_FUNC
PyInit__winrt_Windows_Devices_Pwm (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::Devices::Pwm::module_def);
}
