// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.System.h")
#include "py.Windows.System.h"
#endif

#include <winrt/Windows.Security.Authorization.AppCapabilityAccess.h>

namespace py::proj::Windows::Security::Authorization::AppCapabilityAccess
{}

namespace py::impl::Windows::Security::Authorization::AppCapabilityAccess
{}

namespace py::wrapper::Windows::Security::Authorization::AppCapabilityAccess
{
    using AppCapability = py::winrt_wrapper<winrt::Windows::Security::Authorization::AppCapabilityAccess::AppCapability>;
    using AppCapabilityAccessChangedEventArgs = py::winrt_wrapper<winrt::Windows::Security::Authorization::AppCapabilityAccess::AppCapabilityAccessChangedEventArgs>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Security::Authorization::AppCapabilityAccess::AppCapability>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Security::Authorization::AppCapabilityAccess::AppCapabilityAccessChangedEventArgs>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}