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

    // BasisMode enum — "UB Width" mode
    py::enum_<BasisMode>(m, "BasisMode")
        .value("RANDOM", BasisMode::Random)
        .value("PCA_CORPUS", BasisMode::PCACorpus)
        .export_values();

    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("dim", &Config::dim)
        .def_readwrite("seed", &Config::seed)
        .def_readwrite("metric", &Config::metric)
        .def_readwrite("quant", &Config::quant)
        .def_readwrite("basis", &Config::basis)
        .def_readwrite("pca_sample", &Config::pca_sample)
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
        .def_readonly("modulation_gain", &SearchResult::modulation_gain)
        .def_readonly("pruned_by_bound", &SearchResult::pruned_by_bound)
        .def_readonly("pruned_by_prefilter", &SearchResult::pruned_by_prefilter)
        .def_readonly("exact_evals", &SearchResult::exact_evals)
        .def_readonly("audit_threshold", &SearchResult::audit_threshold)
        .def_readonly("audit_ids", &SearchResult::audit_ids)
        .def_readonly("audit_ubs", &SearchResult::audit_ubs)
        .def_readonly("audit_l2_lbs", &SearchResult::audit_l2_lbs)
        .def_readonly("audit_residuals", &SearchResult::audit_residuals);

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
        .def("build_float32",
             [](MadhavaL2& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                 auto info = arr.request();
                 if (info.ndim != 2)
                     throw std::runtime_error("expected 2D float32 array (n, dim)");
                 int n = (int)info.shape[0];
                 self.build_float32((const float*)info.ptr, n);
                 return n;
             },
             py::arg("arr"))
        .def("set_basis",
             [](MadhavaL2& self, py::array_t<float, py::array::c_style | py::array::forcecast> p1,
                 py::object p2) {
                 auto i1 = p1.request();
                 int s1 = self.config().stage1_dim;
                 if (i1.ndim != 2 || i1.shape[0] != s1 || i1.shape[1] != self.dim())
                     throw std::runtime_error("set_basis: P1 must be (stage1_dim, dim)");
                 const float* p2ptr = nullptr;
                 if (!p2.is_none()) {
                     auto i2 = p2.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>().request();
                     int s2 = self.config().stage2_dim;
                     if (i2.ndim != 2 || i2.shape[0] != s2 || i2.shape[1] != self.dim())
                         throw std::runtime_error("set_basis: P2 must be (stage2_dim, dim)");
                     p2ptr = (const float*)i2.ptr;
                 }
                 self.set_basis((const float*)i1.ptr, p2ptr);
             },
             py::arg("P1"), py::arg("P2") = py::none())
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
        .def("search_audited",
             [](const MadhavaL2& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> q,
                int64_t k, int64_t max_audit_records) {
                 auto info = q.request();
                 if (info.ndim != 1 || (int)info.shape[0] != self.dim())
                     throw std::runtime_error("query must be float32 of length dim");
                 auto r = self.search_audited((const float*)info.ptr, k, max_audit_records);
                 py::dict d;
                 d["indices"] = r.base.indices;
                 d["latency_ms"] = r.base.latency_ms;
                 d["bound_violations"] = r.base.bound_violations;
                 d["bound_pairs"] = r.base.bound_pairs;
                 d["audit_candidates"] = r.audit_candidates;
                 d["audit_excluded"] = r.audit_excluded;
                 py::list audit;
                 for (const auto& rec : r.audit) {
                     py::dict rd;
                     rd["doc_id"] = (int64_t)rec.doc_id;
                     rd["true_cosine"] = rec.true_cosine;
                     rd["projected_cosine"] = rec.projected_cosine;
                     rd["residual_norm"] = rec.residual_norm;
                     rd["upper_bound"] = rec.upper_bound;
                     rd["threshold"] = rec.threshold;
                     rd["excluded"] = rec.excluded;
                     rd["stage"] = rec.stage;
                     audit.append(rd);
                 }
                 d["audit"] = audit;
                 return d;
             },
             py::arg("query"), py::arg("k") = 10, py::arg("max_audit_records") = 500)
        .def("audit_json",
             [](const MadhavaL2& self, py::array_t<float, py::array::c_style | py::array::forcecast> q,
                 int64_t k, int64_t max_audit_records) {
                 auto info = q.request();
                 if (info.ndim != 1 || (int)info.shape[0] != self.dim())
                     throw std::runtime_error("query must be float32 of length dim");
                 return self.audit_json((const float*)info.ptr, k, max_audit_records);
             },
             py::arg("query"), py::arg("k") = 10, py::arg("max_audit_records") = 500)
        .def("search_with_commitment",
             [](const MadhavaL2& self, py::array_t<float, py::array::c_style | py::array::forcecast> q,
                 int64_t k, int64_t max_sample) {
                 auto info = q.request();
                 if (info.ndim != 1 || (int)info.shape[0] != self.dim())
                     throw std::runtime_error("query must be float32 of length dim");
                 auto c = self.search_with_commitment((const float*)info.ptr, k, max_sample);
                 py::dict d;
                 d["indices"] = c.indices;
                 d["bound_pairs"] = c.bound_pairs;
                 d["bound_violations"] = c.bound_violations;
                 d["latency_ms"] = c.latency_ms;
                 d["total_excluded_count"] = c.total_excluded_count;
                 d["global_threshold"] = c.global_threshold;
                 py::list samples;
                 for (const auto& s : c.sampled_records) {
                     py::dict sd;
                     sd["doc_id"] = (int64_t)s.doc_id;
                     sd["upper_bound"] = s.upper_bound;
                     sd["excluded"] = s.excluded;
                     samples.append(sd);
                 }
                 d["sampled_records"] = samples;
                 return d;
             },
             py::arg("query"), py::arg("k") = 10, py::arg("max_sample") = 50)
        .def("search_batch",
             [](const MadhavaL2& self, py::array_t<float, py::array::c_style | py::array::forcecast> q, int nq, int k) {
                 auto info = q.request();
                 if (info.ndim != 2 || (int)info.shape[0] != nq || (int)info.shape[1] != self.dim())
                     throw std::runtime_error("queries must be (nq, dim) float32");
                 return self.search_batch((const float*)info.ptr, nq, k);
             },
             py::arg("queries"), py::arg("nq"), py::arg("k"))
        .def("save_index",
             [](const MadhavaL2& self, const std::string& path) {
                 return self.save_index(path);
             },
             py::arg("path"))
        .def("load_index",
             [](MadhavaL2& self, const std::string& path) {
                 return self.load_index(path);
             },
             py::arg("path"))
        .def("num_vectors", &MadhavaL2::num_vectors)
        .def("dim", &MadhavaL2::dim)
        .def("config", &MadhavaL2::config)
        .def("build_seconds", &MadhavaL2::build_seconds)
        .def("built", &MadhavaL2::built)
        // UB Width diagnostics
        .def("basis1", [](const MadhavaL2& self) {
            int s = self.config().stage1_dim, D = self.dim();
            py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)s, (py::ssize_t)D});
            const float* b = self.basis1();
            if (b) std::memcpy(out.mutable_data(), b, (size_t)s * D * sizeof(float));
            return out;
        })
        .def("basis2", [](const MadhavaL2& self) {
            int s = self.config().stage2_dim, D = self.dim();
            if (s <= 0 || !self.basis2()) return py::array_t<float>();
            py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)s, (py::ssize_t)D});
            std::memcpy(out.mutable_data(), self.basis2(), (size_t)s * D * sizeof(float));
            return out;
        })
        .def("residuals1", [](const MadhavaL2& self) {
            int n = self.num_vectors();
            py::array_t<float> out(n);
            float* dst = out.mutable_data();
            const float* e = self.residuals1();
            if (e) {
                std::memcpy(dst, e, (size_t)n * sizeof(float));
            } else {
                // e1_ not populated (e.g. k >= d, projection is complete):
                // e(v) = sqrt(1 - ||P v||^2) = 0 by construction. Return zeros
                // instead of an uninitialized array.
                std::memset(dst, 0, (size_t)n * sizeof(float));
            }
            return out;
        })
        .def("residuals2", [](const MadhavaL2& self) {
            int n = self.num_vectors();
            py::array_t<float> out(n);
            float* dst = out.mutable_data();
            const float* e = self.residuals2();
            if (e) {
                std::memcpy(dst, e, (size_t)n * sizeof(float));
            } else {
                std::memset(dst, 0, (size_t)n * sizeof(float));
            }
            return out;
        });

    // SpeedEngine — native speed mode (QKᵀ matmul: CUDA if available, else CPU)
    py::class_<SpeedEngine>(m, "SpeedEngine")
        .def(py::init([](py::array_t<float, py::array::c_style> arr, int dim, int metric,
                         int n_anchors, int nprobe, bool require_gpu,
                         const std::string& opencl_lib) {
                 auto info = arr.request();
                 return new SpeedEngine((const float*)info.ptr, (int)info.shape[0], dim,
                                        (Metric)metric, n_anchors, nprobe, require_gpu,
                                        opencl_lib);
             }),
             py::arg("corpus_f32"), py::arg("dim"), py::arg("metric"),
             py::arg("n_anchors") = 0, py::arg("nprobe") = 4,
             py::arg("require_gpu") = false, py::arg("opencl_lib") = "")
        // M3 (v1.8.0): construtor uint8 — evita a cópia float32 extra na camada
        // Python (build_engine(speed=True) fazia arr.astype(float32) na RAM).
        // O C++ converte internamente em corpus_f32_, mas sem a cópia Python 4×.
        // Permite ao DevAI processar corpora uint8 grandes (~10M+) com menor
        // pico de RAM e sem o passo de conversão explícito.
        .def(py::init([](py::array_t<uint8_t, py::array::c_style> arr, int dim, int metric,
                         int n_anchors, int nprobe, bool require_gpu,
                         const std::string& opencl_lib) {
                 auto info = arr.request();
                 return new SpeedEngine((const uint8_t*)info.ptr, (int)info.shape[0], dim,
                                        (Metric)metric, n_anchors, nprobe, require_gpu,
                                        opencl_lib);
             }),
             py::arg("corpus_u8"), py::arg("dim"), py::arg("metric"),
             py::arg("n_anchors") = 0, py::arg("nprobe") = 4,
             py::arg("require_gpu") = false, py::arg("opencl_lib") = "")
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
        .def("has_gpu", &SpeedEngine::has_gpu)
        .def("backend_name", &SpeedEngine::backend_name)
        .def("gpu_reason", &SpeedEngine::gpu_reason);

    m.def("recall_at_k", &recall_at_k, py::arg("result"), py::arg("gt_set"), py::arg("k"));
    m.def("ndcg_at_k", &ndcg_at_k, py::arg("result"), py::arg("gt_set"), py::arg("k"));
    m.def("l2_sq", &l2_sq, py::arg("v_raw"), py::arg("q"), py::arg("dim"));
    m.def("read_bigann_groundtruth", &read_bigann_groundtruth, py::arg("path"), py::arg("n_queries"));
}
