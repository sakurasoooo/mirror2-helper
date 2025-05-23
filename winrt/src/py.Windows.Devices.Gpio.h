// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Devices.Gpio.Provider.h")
#include "py.Windows.Devices.Gpio.Provider.h"
#endif

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#include <winrt/Windows.Devices.Gpio.h>

namespace py::proj::Windows::Devices::Gpio
{}

namespace py::impl::Windows::Devices::Gpio
{}

namespace py::wrapper::Windows::Devices::Gpio
{
    using GpioChangeCounter = py::winrt_wrapper<winrt::Windows::Devices::Gpio::GpioChangeCounter>;
    using GpioChangeReader = py::winrt_wrapper<winrt::Windows::Devices::Gpio::GpioChangeReader>;
    using GpioController = py::winrt_wrapper<winrt::Windows::Devices::Gpio::GpioController>;
    using GpioPin = py::winrt_wrapper<winrt::Windows::Devices::Gpio::GpioPin>;
    using GpioPinValueChangedEventArgs = py::winrt_wrapper<winrt::Windows::Devices::Gpio::GpioPinValueChangedEventArgs>;
    using GpioChangeCount = py::winrt_struct_wrapper<winrt::Windows::Devices::Gpio::GpioChangeCount>;
    using GpioChangeRecord = py::winrt_struct_wrapper<winrt::Windows::Devices::Gpio::GpioChangeRecord>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::GpioChangeCounter>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::GpioChangeReader>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::GpioController>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::GpioPin>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::GpioPinValueChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::GpioChangeCount>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::GpioChangeRecord>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct converter<winrt::Windows::Devices::Gpio::GpioChangeCount>
    {
        static PyObject* convert(winrt::Windows::Devices::Gpio::GpioChangeCount instance) noexcept;
        static winrt::Windows::Devices::Gpio::GpioChangeCount convert_to(PyObject* obj);
    };

    template<>
    struct converter<winrt::Windows::Devices::Gpio::GpioChangeRecord>
    {
        static PyObject* convert(winrt::Windows::Devices::Gpio::GpioChangeRecord instance) noexcept;
        static winrt::Windows::Devices::Gpio::GpioChangeRecord convert_to(PyObject* obj);
    };

}
