// WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

#pragma once

#include "pybase.h"

#if __has_include("py.Windows.Foundation.h")
#include "py.Windows.Foundation.h"
#endif

#if __has_include("py.Windows.Foundation.Collections.h")
#include "py.Windows.Foundation.Collections.h"
#endif

#if __has_include("py.Windows.Globalization.h")
#include "py.Windows.Globalization.h"
#endif

#if __has_include("py.Windows.Networking.h")
#include "py.Windows.Networking.h"
#endif

#if __has_include("py.Windows.Storage.Streams.h")
#include "py.Windows.Storage.Streams.h"
#endif

#if __has_include("py.Windows.Web.Http.h")
#include "py.Windows.Web.Http.h"
#endif

#include <winrt/Windows.Web.Http.Headers.h>

namespace py::proj::Windows::Web::Http::Headers
{}

namespace py::impl::Windows::Web::Http::Headers
{}

namespace py::wrapper::Windows::Web::Http::Headers
{
    using HttpCacheDirectiveHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpCacheDirectiveHeaderValueCollection>;
    using HttpChallengeHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpChallengeHeaderValue>;
    using HttpChallengeHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpChallengeHeaderValueCollection>;
    using HttpConnectionOptionHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpConnectionOptionHeaderValue>;
    using HttpConnectionOptionHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpConnectionOptionHeaderValueCollection>;
    using HttpContentCodingHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpContentCodingHeaderValue>;
    using HttpContentCodingHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpContentCodingHeaderValueCollection>;
    using HttpContentCodingWithQualityHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpContentCodingWithQualityHeaderValue>;
    using HttpContentCodingWithQualityHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpContentCodingWithQualityHeaderValueCollection>;
    using HttpContentDispositionHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpContentDispositionHeaderValue>;
    using HttpContentHeaderCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpContentHeaderCollection>;
    using HttpContentRangeHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpContentRangeHeaderValue>;
    using HttpCookiePairHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpCookiePairHeaderValue>;
    using HttpCookiePairHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpCookiePairHeaderValueCollection>;
    using HttpCredentialsHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpCredentialsHeaderValue>;
    using HttpDateOrDeltaHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpDateOrDeltaHeaderValue>;
    using HttpExpectationHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpExpectationHeaderValue>;
    using HttpExpectationHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpExpectationHeaderValueCollection>;
    using HttpLanguageHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpLanguageHeaderValueCollection>;
    using HttpLanguageRangeWithQualityHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpLanguageRangeWithQualityHeaderValue>;
    using HttpLanguageRangeWithQualityHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpLanguageRangeWithQualityHeaderValueCollection>;
    using HttpMediaTypeHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpMediaTypeHeaderValue>;
    using HttpMediaTypeWithQualityHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpMediaTypeWithQualityHeaderValue>;
    using HttpMediaTypeWithQualityHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpMediaTypeWithQualityHeaderValueCollection>;
    using HttpMethodHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpMethodHeaderValueCollection>;
    using HttpNameValueHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpNameValueHeaderValue>;
    using HttpProductHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpProductHeaderValue>;
    using HttpProductInfoHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpProductInfoHeaderValue>;
    using HttpProductInfoHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpProductInfoHeaderValueCollection>;
    using HttpRequestHeaderCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpRequestHeaderCollection>;
    using HttpResponseHeaderCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpResponseHeaderCollection>;
    using HttpTransferCodingHeaderValue = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpTransferCodingHeaderValue>;
    using HttpTransferCodingHeaderValueCollection = py::winrt_wrapper<winrt::Windows::Web::Http::Headers::HttpTransferCodingHeaderValueCollection>;
}

namespace py
{
    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpCacheDirectiveHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpChallengeHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpChallengeHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpConnectionOptionHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpConnectionOptionHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpContentCodingHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpContentCodingHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpContentCodingWithQualityHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpContentCodingWithQualityHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpContentDispositionHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpContentHeaderCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpContentRangeHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpCookiePairHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpCookiePairHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpCredentialsHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpDateOrDeltaHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpExpectationHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpExpectationHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpLanguageHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpLanguageRangeWithQualityHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpLanguageRangeWithQualityHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpMediaTypeHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpMediaTypeWithQualityHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpMediaTypeWithQualityHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpMethodHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpNameValueHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpProductHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpProductInfoHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpProductInfoHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpRequestHeaderCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpResponseHeaderCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpTransferCodingHeaderValue>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

    template<>
    struct winrt_type<winrt::Windows::Web::Http::Headers::HttpTransferCodingHeaderValueCollection>
    {
        static PyTypeObject* python_type;
        static PyTypeObject* get_python_type() { return python_type; }
    };

}