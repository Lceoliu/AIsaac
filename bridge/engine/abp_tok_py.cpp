// P package (2026-11-19): abp_tok_core.h's functions for Python -- the golden samples (tools/p_golden_cpp.py) and the
// tests call them on numpy arrays. Built by isaac_bridge/tok_native.py (g++, pybind11 from torch's headers; no torch).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "abp_tok_core.h"

namespace py = pybind11;
using abptok::i64;

template <class T> using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

static abptok::V3 v3_of(py::dict d) {
  abptok::V3 c;
  auto I = [&](const char *k) { return d[k].cast<i64>(); };
  auto F = [&](const char *k) { return d[k].cast<double>(); };
  c.header = I("header"); c.h_p = I("h_p"); c.fixed = I("fixed"); c.body = I("body"); c.ef = I("ef"); c.nf = I("nf");
  c.sf = I("sf"); c.frame_bytes = I("frame_bytes"); c.magic = F("magic"); c.version = F("version");
  c.version_word = F("version_word"); c.main_role = F("main_role"); c.cap_body = I("cap_body");
  c.cap_entity = I("cap_entity"); c.n_max = I("n_max"); c.s_max = I("s_max"); c.p_max = I("p_max");
  c.p0_role = I("p0_role"); c.p0_known = I("p0_known");
  c.e_buckets = d["e_buckets"].cast<std::vector<i64>>();
  return c;
}

static abptok::Fold fold_of(py::dict d) {
  abptok::Fold L;
  auto I = [&](const char *k) { return d[k].cast<i64>(); };
  L.n = I("n"); L.nh = I("nh"); L.H = I("H"); L.cw = I("cw"); L.dest_at = I("dest_at"); L.row_bytes = I("row_bytes");
  L.mem = d["mem"].cast<bool>(); L.t_on = d["t_on"].cast<bool>(); L.crn = d["crn"].cast<bool>();
  L.sizes = d["sizes"].cast<std::vector<i64>>(); L.rsizes = d["rsizes"].cast<std::vector<i64>>();
  L.off_e = I("off_e"); L.per_e = I("per_e"); L.end_e = I("end_e");
  L.off_n_ent = I("off_n_ent"); L.off_crn = I("off_crn"); L.off_t = I("off_t");
  return L;
}

static py::tuple check_batch(py::dict consts, arr<float> headers, arr<double> versions, arr<double> sizes,
                             py::object p0, py::object p_max) {
  abptok::V3 c = v3_of(consts);
  if (headers.ndim() != 2 || headers.shape(1) < 8) throw std::invalid_argument("headers: [b, >= 8] float32");
  i64 b = headers.shape(0);
  if (versions.size() < b || sizes.size() < b) throw std::invalid_argument("versions / sizes: [b]");
  arr<float> p0a;
  const float *pp = nullptr;
  if (!p0.is_none()) {
    p0a = p0.cast<arr<float>>();
    if (p0a.size() < 2*b) throw std::invalid_argument("p0: [b, 2] float32");
    pp = p0a.data();
  }
  arr<i64> counts({b, (i64)4});
  std::fill(counts.mutable_data(), counts.mutable_data()+4*b, (i64)0);
  abptok::CheckOut o = abptok::check_batch(c, headers.data(), headers.shape(1), versions.data(), sizes.data(), pp, b,
                                           !p_max.is_none(), p_max.is_none() ? 0 : p_max.cast<i64>(),
                                           counts.mutable_data());
  return py::make_tuple(o.code, o.record, counts, o.trunc_n, o.trunc_s, o.e_bucket);
}

static py::tuple fold_pack(py::dict layout, arr<uint8_t> rows, arr<i64> pending, py::object widx, py::object reset,
                           py::object dest, py::object ukey, py::object ut, i64 eb, py::array_t<uint8_t> stage,
                           py::array_t<uint8_t> ctrl, py::object prefix) {
  abptok::Fold L = fold_of(layout);
  i64 k = rows.ndim() == 2 ? rows.shape(0) : 0;
  if (k < 1 || k > L.n || rows.shape(1) != L.row_bytes) throw std::invalid_argument("rows: [k, row_bytes] uint8, 1 <= k <= n");
  if (pending.size() != k*L.nh) throw std::invalid_argument("pending: [k, nh] int64");
  auto st = stage.mutable_unchecked<2>(); auto ct = ctrl.mutable_unchecked<2>();
  if (st.shape(0) != L.n || st.shape(1) != L.stage_w() || ct.shape(0) != L.n || ct.shape(1) != L.cw ||
      !(stage.flags() & py::array::c_style) || !(ctrl.flags() & py::array::c_style))
    throw std::invalid_argument("stage [n, row_bytes + 8 nh] / ctrl [n, cw]: C-contiguous uint8");
  arr<i64> w_, d_, uk_, ut_, pf_;
  arr<uint8_t> r_;
  const i64 *wp = nullptr, *dp = nullptr, *ukp = nullptr, *utp = nullptr, *pfp = nullptr;
  const uint8_t *rp = nullptr;
  if (!widx.is_none()) { w_ = widx.cast<arr<i64>>(); if (w_.size() != k) throw std::invalid_argument("widx: [k]"); wp = w_.data(); }
  if ((L.mem || L.t_on) && !wp) throw std::invalid_argument("widx needed (mem / t_on)");
  if (!reset.is_none()) { r_ = reset.cast<arr<uint8_t>>(); if (r_.size() != k) throw std::invalid_argument("reset: [k]"); rp = r_.data(); }
  if (!dest.is_none()) {
    if (!L.t_on) throw std::invalid_argument("dest without t_on");
    d_ = dest.cast<arr<i64>>(); if (d_.size() != k) throw std::invalid_argument("dest: [k]"); dp = d_.data();
  }
  if (!ukey.is_none()) {
    uk_ = ukey.cast<arr<i64>>(); ut_ = ut.cast<arr<i64>>();
    if (uk_.size() != k || ut_.size() != k) throw std::invalid_argument("ukey / ut: [k]");
    ukp = uk_.data(); utp = ut_.data();
  }
  if (!prefix.is_none()) { pf_ = prefix.cast<arr<i64>>(); if (pf_.size() != k) throw std::invalid_argument("prefix: [k]"); pfp = pf_.data(); }
  abptok::FoldOut o = abptok::fold_pack(L, rows.data(), k, pending.data(), wp, rp, dp, ukp, utp, eb,
                                        stage.mutable_data(), ctrl.mutable_data(), pfp);
  if (o.g < 0) throw std::invalid_argument("fold_pack: no graph for these records (n_ent / eb / k past the layout)");
  return py::make_tuple(o.g, o.r, o.graph, o.crn_rows);
}

static arr<float> crn(arr<i64> keys, arr<i64> t, i64 columns) {
  i64 n = keys.size();
  if (t.size() != n) throw std::invalid_argument("crn: keys and t of one length");
  arr<float> out({n, columns});
  for (i64 i = 0; i < n; i++) abptok::crn_row(keys.data()[i], t.data()[i], columns, out.mutable_data()+i*columns);
  return out;
}

PYBIND11_MODULE(TOK_EXTENSION_NAME, m) {
  m.doc() = "P package: abp_tok_core.h (v3 frame check, CRN, fold packing) for the golden samples and tests";
  m.def("check_batch", &check_batch, py::arg("consts"), py::arg("headers"), py::arg("versions"), py::arg("sizes"),
        py::arg("p0") = py::none(), py::arg("p_max") = py::none());
  m.def("fold_pack", &fold_pack, py::arg("layout"), py::arg("rows"), py::arg("pending"), py::arg("widx"),
        py::arg("reset"), py::arg("dest"), py::arg("ukey"), py::arg("ut"), py::arg("eb"), py::arg("stage"),
        py::arg("ctrl"), py::arg("prefix") = py::none());
  m.def("crn", &crn, py::arg("keys"), py::arg("t"), py::arg("columns"));
}
