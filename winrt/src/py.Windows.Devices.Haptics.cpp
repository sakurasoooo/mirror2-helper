// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#include "pybase.h"
#include "py.Windows.Devices.Haptics.h"

PyTypeObject* py::winrt_type<winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Devices::Haptics::SimpleHapticsController>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Devices::Haptics::SimpleHapticsControllerFeedback>::python_type;
PyTypeObject* py::winrt_type<winrt::Windows::Devices::Haptics::VibrationDevice>::python_type;

namespace py::cpp::Windows::Devices::Haptics
{
    // ----- KnownSimpleHapticsControllerWaveforms class --------------------
    constexpr const char* const _type_name_KnownSimpleHapticsControllerWaveforms = "KnownSimpleHapticsControllerWaveforms";

    static PyObject* _new_KnownSimpleHapticsControllerWaveforms(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_KnownSimpleHapticsControllerWaveforms);
        return nullptr;
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_BuzzContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::BuzzContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_Click(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::Click());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_Press(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::Press());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_Release(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::Release());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_RumbleContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::RumbleContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_BrushContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::BrushContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_ChiselMarkerContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::ChiselMarkerContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_EraserContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::EraserContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_Error(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::Error());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_GalaxyPenContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::GalaxyPenContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_Hover(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::Hover());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_InkContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::InkContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_MarkerContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::MarkerContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_PencilContinuous(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::PencilContinuous());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* KnownSimpleHapticsControllerWaveforms_get_Success(PyObject* /*unused*/, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms::Success());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_KnownSimpleHapticsControllerWaveforms[] = {
        { "get_buzz_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_BuzzContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_click", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_Click, METH_NOARGS | METH_STATIC, nullptr },
        { "get_press", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_Press, METH_NOARGS | METH_STATIC, nullptr },
        { "get_release", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_Release, METH_NOARGS | METH_STATIC, nullptr },
        { "get_rumble_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_RumbleContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_brush_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_BrushContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_chisel_marker_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_ChiselMarkerContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_eraser_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_EraserContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_error", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_Error, METH_NOARGS | METH_STATIC, nullptr },
        { "get_galaxy_pen_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_GalaxyPenContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_hover", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_Hover, METH_NOARGS | METH_STATIC, nullptr },
        { "get_ink_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_InkContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_marker_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_MarkerContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_pencil_continuous", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_PencilContinuous, METH_NOARGS | METH_STATIC, nullptr },
        { "get_success", (PyCFunction)KnownSimpleHapticsControllerWaveforms_get_Success, METH_NOARGS | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_KnownSimpleHapticsControllerWaveforms[] = {
        { nullptr }
    };

    static PyType_Slot _type_slots_KnownSimpleHapticsControllerWaveforms[] = 
    {
        { Py_tp_new, _new_KnownSimpleHapticsControllerWaveforms },
        { Py_tp_methods, _methods_KnownSimpleHapticsControllerWaveforms },
        { Py_tp_getset, _getset_KnownSimpleHapticsControllerWaveforms },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_KnownSimpleHapticsControllerWaveforms =
    {
        "_winrt_Windows_Devices_Haptics.KnownSimpleHapticsControllerWaveforms",
        0,
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_KnownSimpleHapticsControllerWaveforms
    };

    // ----- SimpleHapticsController class --------------------
    constexpr const char* const _type_name_SimpleHapticsController = "SimpleHapticsController";

    static PyObject* _new_SimpleHapticsController(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SimpleHapticsController);
        return nullptr;
    }

    static void _dealloc_SimpleHapticsController(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SimpleHapticsController_SendHapticFeedback(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Devices::Haptics::SimpleHapticsControllerFeedback>(args, 0);

                self->obj.SendHapticFeedback(param0);
                Py_RETURN_NONE;
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
                auto param0 = py::convert_to<winrt::Windows::Devices::Haptics::SimpleHapticsControllerFeedback>(args, 0);
                auto param1 = py::convert_to<double>(args, 1);

                self->obj.SendHapticFeedback(param0, param1);
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

    static PyObject* SimpleHapticsController_SendHapticFeedbackForDuration(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 3)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Devices::Haptics::SimpleHapticsControllerFeedback>(args, 0);
                auto param1 = py::convert_to<double>(args, 1);
                auto param2 = py::convert_to<winrt::Windows::Foundation::TimeSpan>(args, 2);

                self->obj.SendHapticFeedbackForDuration(param0, param1, param2);
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

    static PyObject* SimpleHapticsController_SendHapticFeedbackForPlayCount(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 4)
        {
            try
            {
                auto param0 = py::convert_to<winrt::Windows::Devices::Haptics::SimpleHapticsControllerFeedback>(args, 0);
                auto param1 = py::convert_to<double>(args, 1);
                auto param2 = py::convert_to<int32_t>(args, 2);
                auto param3 = py::convert_to<winrt::Windows::Foundation::TimeSpan>(args, 3);

                self->obj.SendHapticFeedbackForPlayCount(param0, param1, param2, param3);
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

    static PyObject* SimpleHapticsController_StopFeedback(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                self->obj.StopFeedback();
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

    static PyObject* SimpleHapticsController_get_Id(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, void* /*unused*/) noexcept
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

    static PyObject* SimpleHapticsController_get_IsIntensitySupported(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsIntensitySupported());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SimpleHapticsController_get_IsPlayCountSupported(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsPlayCountSupported());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SimpleHapticsController_get_IsPlayDurationSupported(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsPlayDurationSupported());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SimpleHapticsController_get_IsReplayPauseIntervalSupported(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.IsReplayPauseIntervalSupported());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SimpleHapticsController_get_SupportedFeedback(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SupportedFeedback());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SimpleHapticsController(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Haptics::SimpleHapticsController>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SimpleHapticsController[] = {
        { "send_haptic_feedback", (PyCFunction)SimpleHapticsController_SendHapticFeedback, METH_VARARGS, nullptr },
        { "send_haptic_feedback_for_duration", (PyCFunction)SimpleHapticsController_SendHapticFeedbackForDuration, METH_VARARGS, nullptr },
        { "send_haptic_feedback_for_play_count", (PyCFunction)SimpleHapticsController_SendHapticFeedbackForPlayCount, METH_VARARGS, nullptr },
        { "stop_feedback", (PyCFunction)SimpleHapticsController_StopFeedback, METH_VARARGS, nullptr },
        { "_from", (PyCFunction)_from_SimpleHapticsController, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SimpleHapticsController[] = {
        { const_cast<char*>("id"), (getter)SimpleHapticsController_get_Id, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_intensity_supported"), (getter)SimpleHapticsController_get_IsIntensitySupported, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_play_count_supported"), (getter)SimpleHapticsController_get_IsPlayCountSupported, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_play_duration_supported"), (getter)SimpleHapticsController_get_IsPlayDurationSupported, nullptr, nullptr, nullptr },
        { const_cast<char*>("is_replay_pause_interval_supported"), (getter)SimpleHapticsController_get_IsReplayPauseIntervalSupported, nullptr, nullptr, nullptr },
        { const_cast<char*>("supported_feedback"), (getter)SimpleHapticsController_get_SupportedFeedback, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SimpleHapticsController[] = 
    {
        { Py_tp_new, _new_SimpleHapticsController },
        { Py_tp_dealloc, _dealloc_SimpleHapticsController },
        { Py_tp_methods, _methods_SimpleHapticsController },
        { Py_tp_getset, _getset_SimpleHapticsController },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SimpleHapticsController =
    {
        "_winrt_Windows_Devices_Haptics.SimpleHapticsController",
        sizeof(py::wrapper::Windows::Devices::Haptics::SimpleHapticsController),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SimpleHapticsController
    };

    // ----- SimpleHapticsControllerFeedback class --------------------
    constexpr const char* const _type_name_SimpleHapticsControllerFeedback = "SimpleHapticsControllerFeedback";

    static PyObject* _new_SimpleHapticsControllerFeedback(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_SimpleHapticsControllerFeedback);
        return nullptr;
    }

    static void _dealloc_SimpleHapticsControllerFeedback(py::wrapper::Windows::Devices::Haptics::SimpleHapticsControllerFeedback* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* SimpleHapticsControllerFeedback_get_Duration(py::wrapper::Windows::Devices::Haptics::SimpleHapticsControllerFeedback* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Duration());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* SimpleHapticsControllerFeedback_get_Waveform(py::wrapper::Windows::Devices::Haptics::SimpleHapticsControllerFeedback* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.Waveform());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_SimpleHapticsControllerFeedback(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Haptics::SimpleHapticsControllerFeedback>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_SimpleHapticsControllerFeedback[] = {
        { "_from", (PyCFunction)_from_SimpleHapticsControllerFeedback, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_SimpleHapticsControllerFeedback[] = {
        { const_cast<char*>("duration"), (getter)SimpleHapticsControllerFeedback_get_Duration, nullptr, nullptr, nullptr },
        { const_cast<char*>("waveform"), (getter)SimpleHapticsControllerFeedback_get_Waveform, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_SimpleHapticsControllerFeedback[] = 
    {
        { Py_tp_new, _new_SimpleHapticsControllerFeedback },
        { Py_tp_dealloc, _dealloc_SimpleHapticsControllerFeedback },
        { Py_tp_methods, _methods_SimpleHapticsControllerFeedback },
        { Py_tp_getset, _getset_SimpleHapticsControllerFeedback },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_SimpleHapticsControllerFeedback =
    {
        "_winrt_Windows_Devices_Haptics.SimpleHapticsControllerFeedback",
        sizeof(py::wrapper::Windows::Devices::Haptics::SimpleHapticsControllerFeedback),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_SimpleHapticsControllerFeedback
    };

    // ----- VibrationDevice class --------------------
    constexpr const char* const _type_name_VibrationDevice = "VibrationDevice";

    static PyObject* _new_VibrationDevice(PyTypeObject* type, PyObject* args, PyObject* kwds) noexcept
    {
        py::set_invalid_activation_error(_type_name_VibrationDevice);
        return nullptr;
    }

    static void _dealloc_VibrationDevice(py::wrapper::Windows::Devices::Haptics::VibrationDevice* self)
    {
        auto hash_value = std::hash<winrt::Windows::Foundation::IInspectable>{}(self->obj);
        py::wrapped_instance(hash_value, nullptr);
        self->obj = nullptr;
    }

    static PyObject* VibrationDevice_FindAllAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::Devices::Haptics::VibrationDevice::FindAllAsync());
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

    static PyObject* VibrationDevice_FromIdAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 1)
        {
            try
            {
                auto param0 = py::convert_to<winrt::hstring>(args, 0);

                return py::convert(winrt::Windows::Devices::Haptics::VibrationDevice::FromIdAsync(param0));
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

    static PyObject* VibrationDevice_GetDefaultAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::Devices::Haptics::VibrationDevice::GetDefaultAsync());
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

    static PyObject* VibrationDevice_GetDeviceSelector(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::Devices::Haptics::VibrationDevice::GetDeviceSelector());
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

    static PyObject* VibrationDevice_RequestAccessAsync(PyObject* /*unused*/, PyObject* args) noexcept
    {
        Py_ssize_t arg_count = PyTuple_Size(args);

        if (arg_count == 0)
        {
            try
            {
                return py::convert(winrt::Windows::Devices::Haptics::VibrationDevice::RequestAccessAsync());
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

    static PyObject* VibrationDevice_get_Id(py::wrapper::Windows::Devices::Haptics::VibrationDevice* self, void* /*unused*/) noexcept
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

    static PyObject* VibrationDevice_get_SimpleHapticsController(py::wrapper::Windows::Devices::Haptics::VibrationDevice* self, void* /*unused*/) noexcept
    {
        try
        {
            return py::convert(self->obj.SimpleHapticsController());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyObject* _from_VibrationDevice(PyObject* /*unused*/, PyObject* arg) noexcept
    {
        try
        {
            auto return_value = py::convert_to<winrt::Windows::Foundation::IInspectable>(arg);
            return py::convert(return_value.as<winrt::Windows::Devices::Haptics::VibrationDevice>());
        }
        catch (...)
        {
            py::to_PyErr();
            return nullptr;
        }
    }

    static PyMethodDef _methods_VibrationDevice[] = {
        { "find_all_async", (PyCFunction)VibrationDevice_FindAllAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "from_id_async", (PyCFunction)VibrationDevice_FromIdAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "get_default_async", (PyCFunction)VibrationDevice_GetDefaultAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "get_device_selector", (PyCFunction)VibrationDevice_GetDeviceSelector, METH_VARARGS | METH_STATIC, nullptr },
        { "request_access_async", (PyCFunction)VibrationDevice_RequestAccessAsync, METH_VARARGS | METH_STATIC, nullptr },
        { "_from", (PyCFunction)_from_VibrationDevice, METH_O | METH_STATIC, nullptr },
        { nullptr }
    };

    static PyGetSetDef _getset_VibrationDevice[] = {
        { const_cast<char*>("id"), (getter)VibrationDevice_get_Id, nullptr, nullptr, nullptr },
        { const_cast<char*>("simple_haptics_controller"), (getter)VibrationDevice_get_SimpleHapticsController, nullptr, nullptr, nullptr },
        { nullptr }
    };

    static PyType_Slot _type_slots_VibrationDevice[] = 
    {
        { Py_tp_new, _new_VibrationDevice },
        { Py_tp_dealloc, _dealloc_VibrationDevice },
        { Py_tp_methods, _methods_VibrationDevice },
        { Py_tp_getset, _getset_VibrationDevice },
        { 0, nullptr },
    };

    static PyType_Spec _type_spec_VibrationDevice =
    {
        "_winrt_Windows_Devices_Haptics.VibrationDevice",
        sizeof(py::wrapper::Windows::Devices::Haptics::VibrationDevice),
        0,
        Py_TPFLAGS_DEFAULT,
        _type_slots_VibrationDevice
    };

    // ----- Windows.Devices.Haptics Initialization --------------------
    static int module_exec(PyObject* module) noexcept
    {
        try
        {
            py::pyobj_handle bases { PyTuple_Pack(1, py::winrt_type<py::winrt_base>::python_type) };

            py::winrt_type<winrt::Windows::Devices::Haptics::KnownSimpleHapticsControllerWaveforms>::python_type = py::register_python_type(module, _type_name_KnownSimpleHapticsControllerWaveforms, &_type_spec_KnownSimpleHapticsControllerWaveforms, nullptr);
            py::winrt_type<winrt::Windows::Devices::Haptics::SimpleHapticsController>::python_type = py::register_python_type(module, _type_name_SimpleHapticsController, &_type_spec_SimpleHapticsController, bases.get());
            py::winrt_type<winrt::Windows::Devices::Haptics::SimpleHapticsControllerFeedback>::python_type = py::register_python_type(module, _type_name_SimpleHapticsControllerFeedback, &_type_spec_SimpleHapticsControllerFeedback, bases.get());
            py::winrt_type<winrt::Windows::Devices::Haptics::VibrationDevice>::python_type = py::register_python_type(module, _type_name_VibrationDevice, &_type_spec_VibrationDevice, bases.get());

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

    PyDoc_STRVAR(module_doc, "Windows.Devices.Haptics");

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_winrt_Windows_Devices_Haptics",
        module_doc,
        0,
        nullptr,
        module_slots,
        nullptr,
        nullptr,
        nullptr
    };
} // py::cpp::Windows::Devices::Haptics

PyMODINIT_FUNC
PyInit__winrt_Windows_Devices_Haptics (void) noexcept
{
    return PyModuleDef_Init(&py::cpp::Windows::Devices::Haptics::module_def);
}
