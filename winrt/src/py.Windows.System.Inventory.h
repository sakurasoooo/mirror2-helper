// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#include <winrt/Windows.System.Inventory.h>

namespace py::proj::Windows::System::Inventory
{}

namespace py::impl::Windows::System::Inventory
{}

namespace py::wrapper::Windows::System::Inventory
{
    using InstalledDesktopApp = py::winrt_wrapper<winrt::Windows::System::Inventory::InstalledDesktopApp>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::System::Inventory::InstalledDesktopApp>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}