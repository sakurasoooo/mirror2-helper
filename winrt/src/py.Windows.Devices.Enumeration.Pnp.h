// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Devices.Enumeration.h")
#include "py.Windows.Devices.Enumeration.h"
#endif

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#include <winrt/Windows.Devices.Enumeration.Pnp.h>

namespace py::proj::Windows::Devices::Enumeration::Pnp
{}

namespace py::impl::Windows::Devices::Enumeration::Pnp
{}

namespace py::wrapper::Windows::Devices::Enumeration::Pnp
{
    using PnpObject = py::winrt_wrapper<winrt::Windows::Devices::Enumeration::Pnp::PnpObject>;
    using PnpObjectCollection = py::winrt_wrapper<winrt::Windows::Devices::Enumeration::Pnp::PnpObjectCollection>;
    using PnpObjectUpdate = py::winrt_wrapper<winrt::Windows::Devices::Enumeration::Pnp::PnpObjectUpdate>;
    using PnpObjectWatcher = py::winrt_wrapper<winrt::Windows::Devices::Enumeration::Pnp::PnpObjectWatcher>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Devices::Enumeration::Pnp::PnpObject>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Enumeration::Pnp::PnpObjectCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Enumeration::Pnp::PnpObjectUpdate>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Devices::Enumeration::Pnp::PnpObjectWatcher>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}