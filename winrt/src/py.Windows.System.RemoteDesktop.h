// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#include <winrt/Windows.System.RemoteDesktop.h>

namespace py::proj::Windows::System::RemoteDesktop
{}

namespace py::impl::Windows::System::RemoteDesktop
{}

namespace py::wrapper::Windows::System::RemoteDesktop
{
    using InteractiveSession = py::winrt_wrapper<winrt::Windows::System::RemoteDesktop::InteractiveSession>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::System::RemoteDesktop::InteractiveSession>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}
