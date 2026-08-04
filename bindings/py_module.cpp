/**
 * py_module.cpp — Python bindings for the Madhava L2 library (pybind11).
 * Exposes the engine, the exact-scan ceiling baseline, and the metrics.
 *
 *   import winnex_madhava
 *   eng = winnex_madhava.MadhavaL2(dim=128, stage1_dim=64, k=10, postfilter=True)
 *   eng.build(raw_bytes)          # raw_bytes: bytes/bytearray of n*dim uint8
 *   res = eng.search(query_float32)  # returns SearchResult
 *   res_exact = eng.search_exact(query_float32)
 */
#include "winnex_madhava/winnex_madhava.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace winnex_madhava;

PYBIND11_MODULE(_winnex_madhava, m) {
    m.doc() = "Madhava L2 — deterministic vector search with Cauchy-Schwarz bounds";

    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("dim", &Config::dim)
        .def_readwrite("stage1_dim", &Config::stage1_dim)
        .def_readwrite("seed", &Config::seed)
        .def_readwrite("k", &Config::k)
        .def_readwrite("k1_fraction", &Config::k1_fraction)
        .def_readwrite("k1_min", &Config::k1_min)
        .def_readwrite("postfilter", &Config::postfilter)
        .def_readwrite("n_threads", &Config::n_threads);

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("indices", &SearchResult::indices)
        .def_readonly("k1", &SearchResult::k1)
        .def_readonly("k3", &SearchResult::k3)
        .def_readonly("latency_ms", &SearchResult::latency_ms)
        .def_readonly("bound_pairs", &SearchResult::bound_pairs)
        .def_readonly("bound_violations", &SearchResult::bound_violations);

    py::class_<MadhavaL2>(m, "MadhavaL2")
        .def(py::init<const Config&>())
        .def("build",
             [](MadhavaL2& self, py::bytes raw, int n, int dim) {
                 char* buf = nullptr;
                 Py_ssize_t len = 0;
                 PyBytes_AsStringAndSize(raw.ptr(), &buf, &len);
                 if ((size_t)len < (size_t)n * dim)
                     throw std::runtime_error("raw buffer too small for n*dim");
                 self.build((const uint8_t*)buf, n);
                 // NOTE: py::bytes keeps a copy; for huge corpora prefer
                 // building from an mmap'ed numpy array (see build_numpy).
             },
             py::arg("raw"), py::arg("n"), py::arg("dim"))
        .def("build_numpy",
             [](MadhavaL2& self, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr) {
                 auto info = arr.request();
                 if (info.ndim != 2)
                     throw std::runtime_error("expected 2D uint8 array (n, dim)");
                 int n = (int)info.shape[0];
                 self.build((const uint8_t*)info.ptr, n);
                 return n;
             },
             py::arg("arr"))
        .def("search",
             [](const MadhavaL2& self, py::array_t<float, py::array::c_style | py::array::forcecast> q) {
                 auto info = q.request();
                 if (info.ndim != 1 || (int)info.shape[0] != self.dim())
                     throw std::runtime_error("query must be float32 of length dim");
                 return self.search((const float*)info.ptr);
             },
             py::arg("query"))
        .def("search_exact",
             [](const MadhavaL2& self, py::array_t<float, py::array::c_style | py::array::forcecast> q) {
                 auto info = q.request();
                 if (info.ndim != 1 || (int)info.shape[0] != self.dim())
                     throw std::runtime_error("query must be float32 of length dim");
                 return self.search_exact((const float*)info.ptr);
             },
             py::arg("query"))
        .def("num_vectors", &MadhavaL2::num_vectors)
        .def("dim", &MadhavaL2::dim)
        .def("config", &MadhavaL2::config)
        .def("build_seconds", &MadhavaL2::build_seconds)
        .def("built", &MadhavaL2::built);

    m.def("recall_at_k", &recall_at_k, py::arg("result"), py::arg("gt_set"), py::arg("k"));
    m.def("ndcg_at_k", &ndcg_at_k, py::arg("result"), py::arg("gt_set"), py::arg("k"));
    m.def("l2_sq", &l2_sq, py::arg("v_raw"), py::arg("q"), py::arg("dim"));
    m.def("read_bigann_groundtruth", &read_bigann_groundtruth, py::arg("path"), py::arg("n_queries"));
}
