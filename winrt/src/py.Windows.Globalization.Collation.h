// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#include <winrt/Windows.Globalization.Collation.h>

namespace py::proj::Windows::Globalization::Collation
{}

namespace py::impl::Windows::Globalization::Collation
{}

namespace py::wrapper::Windows::Globalization::Collation
{
    using CharacterGrouping = py::winrt_wrapper<winrt::Windows::Globalization::Collation::CharacterGrouping>;
    using CharacterGroupings = py::winrt_wrapper<winrt::Windows::Globalization::Collation::CharacterGroupings>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Globalization::Collation::CharacterGrouping>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Globalization::Collation::CharacterGroupings>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}
