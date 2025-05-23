// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.ApplicationModel.Contacts.h")
#include "py.Windows.ApplicationModel.Contacts.h"
#endif

#if __has_include("py.Windows.ApplicationModel.DataTransfer.h")
#include "py.Windows.ApplicationModel.DataTransfer.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.Storage.Streams.h")
#include "py.Windows.Storage.Streams.h"
#endif

#include <winrt/Windows.ApplicationModel.DataTransfer.ShareTarget.h>

namespace py::proj::Windows::ApplicationModel::DataTransfer::ShareTarget
{}

namespace py::impl::Windows::ApplicationModel::DataTransfer::ShareTarget
{}

namespace py::wrapper::Windows::ApplicationModel::DataTransfer::ShareTarget
{
    using QuickLink = py::winrt_wrapper<winrt::Windows::ApplicationModel::DataTransfer::ShareTarget::QuickLink>;
    using ShareOperation = py::winrt_wrapper<winrt::Windows::ApplicationModel::DataTransfer::ShareTarget::ShareOperation>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::ApplicationModel::DataTransfer::ShareTarget::QuickLink>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::ApplicationModel::DataTransfer::ShareTarget::ShareOperation>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}
