// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Devices.Geolocation.h")
#include "py.Windows.Devices.Geolocation.h"
#endif

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.Globalization.h")
#include "py.Windows.Globalization.h"
#endif

#if __has_include("py.Windows.Services.Maps.h")
#include "py.Windows.Services.Maps.h"
#endif

#include <winrt/Windows.Services.Maps.LocalSearch.h>

namespace py::proj::Windows::Services::Maps::LocalSearch
{}

namespace py::impl::Windows::Services::Maps::LocalSearch
{}

namespace py::wrapper::Windows::Services::Maps::LocalSearch
{
    using LocalCategories = py::winrt_wrapper<winrt::Windows::Services::Maps::LocalSearch::LocalCategories>;
    using LocalLocation = py::winrt_wrapper<winrt::Windows::Services::Maps::LocalSearch::LocalLocation>;
    using LocalLocationFinder = py::winrt_wrapper<winrt::Windows::Services::Maps::LocalSearch::LocalLocationFinder>;
    using LocalLocationFinderResult = py::winrt_wrapper<winrt::Windows::Services::Maps::LocalSearch::LocalLocationFinderResult>;
    using LocalLocationHoursOfOperationItem = py::winrt_wrapper<winrt::Windows::Services::Maps::LocalSearch::LocalLocationHoursOfOperationItem>;
    using LocalLocationRatingInfo = py::winrt_wrapper<winrt::Windows::Services::Maps::LocalSearch::LocalLocationRatingInfo>;
    using PlaceInfoHelper = py::winrt_wrapper<winrt::Windows::Services::Maps::LocalSearch::PlaceInfoHelper>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Services::Maps::LocalSearch::LocalCategories>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Services::Maps::LocalSearch::LocalLocation>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Services::Maps::LocalSearch::LocalLocationFinder>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Services::Maps::LocalSearch::LocalLocationFinderResult>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Services::Maps::LocalSearch::LocalLocationHoursOfOperationItem>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Services::Maps::LocalSearch::LocalLocationRatingInfo>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Services::Maps::LocalSearch::PlaceInfoHelper>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}
