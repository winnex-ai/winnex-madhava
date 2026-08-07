/**
 * py_module.cpp — Python bindings for the Winnex Madhava engine (pybind11).
 * Exposes the engine, the exact-scan ceiling baseline, and the metrics.
 *
 *   import winnex_madhava
 *   eng = winnex_madhava.MadhavaL2(dim=384, stage1_dim=64, stage2_dim=128,
 *                                  metric='cosine', quant='int8', k=10,
 *                                  modulation=True, postfilter=True)
 *   eng.build(raw_bytes)              # raw_bytes: bytes/bytearray of n*dim uint8
 *   res = eng.search(query_float32)   # returns SearchResult
 *   res_exact = eng.search_exact(query_float32)
 */
#include "winnex_madhava/winnex_madhava.hpp"
#include "winnex_madhava/speed_engine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace winnex_madhava;

PYBIND11_MODULE(_winnex_madhava, m) {
    m.doc() = "Winnex Madhava — deterministic vector search with Cauchy-Schwarz bounds";

    // Metric enum
    py::enum_<Metric>(m, "Metric")
        .value("COSINE", Metric::Cosine)
        .value("L2", Metric::L2)
        .export_values();

    // QuantMode enum
    py::enum_<QuantMode>(m, "QuantMode")
        .value("INT8", QuantMode::Int8)
        .value("NONE", QuantMode::None)
        .export_values();

    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("dim", &Config::dim)
        .def_readwrite("seed", &Config::seed)
        .def_readwrite("metric", &Config::metric)
        .def_readwrite("quant", &Config::quant)
        .def_readwrite("stage1_dim", &Config::stage1_dim)
        .def_readwrite("stage2_dim", &Config::stage2_dim)
        .def_readwrite("k", &Config::k)
        .def_readwrite("k1_fraction", &Config::k1_fraction)
        .def_readwrite("k2_fraction", &Config::k2_fraction)
        .def_readwrite("k1_min", &Config::k1_min)
        .def_readwrite("k2_min", &Config::k2_min)
        .def_readwrite("k2_max", &Config::k2_max)
        .def_readwrite("modulation", &Config::modulation)
        .def_readwrite("postfilter", &Config::postfilter)
        .def_readwrite("normalize_input", &Config::normalize_input)
        .def_readwrite("early_exit", &Config::early_exit)
        .def_readwrite("n_threads", &Config::n_threads);

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("indices", &SearchResult::indices)
        .def_readonly("k1", &SearchResult::k1)
        .def_readonly("k2", &SearchResult::k2)
        .def_readonly("k3", &SearchResult::k3)
        .def_readonly("latency_ms", &SearchResult::latency_ms)
        .def_readonly("bound_pairs", &SearchResult::bound_pairs)
        .def_readonly("bound_violations", &SearchResult::bound_violations)
        .def_readonly("modulation_gain", &SearchResult::modulation_gain);

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

    // SpeedEngine — native speed mode (QKᵀ matmul: CUDA if available, else CPU)
    py::class_<SpeedEngine>(m, "SpeedEngine")
        .def(py::init([](py::array_t<float, py::array::c_style> arr, int dim, int metric) {
                 auto info = arr.request();
                 return new SpeedEngine((const float*)info.ptr, (int)info.shape[0], dim,
                                        (Metric)metric);
             }),
             py::arg("corpus_f32"), py::arg("dim"), py::arg("metric"))
        .def("build_numpy_u8",
             [](SpeedEngine& self, py::array_t<uint8_t, py::array::c_style> arr, int dim, int metric) {
                 auto info = arr.request();
                 return new SpeedEngine((const uint8_t*)info.ptr, (int)info.shape[0], dim,
                                        (Metric)metric);
             })
        .def("search",
             [](const SpeedEngine& self, py::array_t<float, py::array::c_style> q, int k) {
                 auto info = q.request();
                 auto r = self.search((const float*)info.ptr, k);
                 py::dict d;
                 d["indices"] = r.indices;
                 d["latency_ms"] = r.latency_ms;
                 d["bound_pairs"] = r.bound_pairs;
                 d["bound_violations"] = r.bound_violations;
                 return d;
             },
             py::arg("query"), py::arg("k"))
        .def("search_batch",
             [](const SpeedEngine& self, py::array_t<float, py::array::c_style> q, int nq, int k) {
                 auto info = q.request();
                 auto r = self.search_batch((const float*)info.ptr, nq, k);
                 py::dict d;
                 d["indices"] = r.indices;
                 d["latency_ms"] = r.latency_ms;
                 return d;
             },
             py::arg("queries"), py::arg("nq"), py::arg("k"))
        .def("num_vectors", &SpeedEngine::num_vectors)
        .def("dim", &SpeedEngine::dim)
        .def("has_gpu", &SpeedEngine::has_gpu);

    m.def("recall_at_k", &recall_at_k, py::arg("result"), py::arg("gt_set"), py::arg("k"));
    m.def("ndcg_at_k", &ndcg_at_k, py::arg("result"), py::arg("gt_set"), py::arg("k"));
    m.def("l2_sq", &l2_sq, py::arg("v_raw"), py::arg("q"), py::arg("dim"));
    m.def("read_bigann_groundtruth", &read_bigann_groundtruth, py::arg("path"), py::arg("n_queries"));
}
