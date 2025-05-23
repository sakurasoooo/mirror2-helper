// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#include <winrt/Windows.Devices.Gpio.Provider.h>

namespace py::proj::Windows::Devices::Gpio::Provider
{}

namespace py::impl::Windows::Devices::Gpio::Provider
{}

namespace py::wrapper::Windows::Devices::Gpio::Provider
{
    using GpioPinProviderValueChangedEventArgs = py::winrt_wrapper<winrt::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs>;
    using IGpioControllerProvider = py::winrt_wrapper<winrt::Windows::Devices::Gpio::Provider::IGpioControllerProvider>;
    using IGpioPinProvider = py::winrt_wrapper<winrt::Windows::Devices::Gpio::Provider::IGpioPinProvider>;
    using IGpioProvider = py::winrt_wrapper<winrt::Windows::Devices::Gpio::Provider::IGpioProvider>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::Provider::GpioPinProviderValueChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioControllerProvider>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioPinProvider>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Gpio::Provider::IGpioProvider>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}
