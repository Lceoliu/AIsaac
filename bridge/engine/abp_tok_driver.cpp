// P package phase 1 (2026-11-19): the thin TokPolicy driver -- milestone A, the call's middle in C++.
// TokMiddle replays the CUDA graphs a GraphActor(fold=True) captured (registered from Python, never captured here:
// P-driver-design 11.3) for one call of records: the v3 check (abp_tok_core check_batch), the fold staging with the
// frame-prefix rule (fold_pack), the cf_snap bank copies, three H2D copies (raw, v3_counts, ctrl), the replay, the D2H
// copy of the outputs, the step's K / V, s and o into the output ring (or a rollout's tensors) with index_copy_, all on
// the driver's stream, then a synchronize. call() is the test entry (tools/p_replay.py's 'cpp' backend: the same-batch
// gate) and the body the driver thread will run per call (milestone B).
#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "abp_tok_core.h"

namespace py = pybind11;
using abptok::i64;

namespace {

abptok::V3 v3_of(py::dict d) {
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

abptok::Fold fold_of(py::dict d) {
  abptok::Fold L;
  auto I = [&](const char *k) { return d[k].cast<i64>(); };
  L.n = I("n"); L.nh = I("nh"); L.H = I("H"); L.cw = I("cw"); L.dest_at = I("dest_at"); L.row_bytes = I("row_bytes");
  L.mem = d["mem"].cast<bool>(); L.t_on = d["t_on"].cast<bool>(); L.crn = d["crn"].cast<bool>();
  L.sizes = d["sizes"].cast<std::vector<i64>>(); L.rsizes = d["rsizes"].cast<std::vector<i64>>();
  L.off_e = I("off_e"); L.per_e = I("per_e"); L.end_e = I("end_e");
  L.off_n_ent = I("off_n_ent"); L.off_crn = I("off_crn"); L.off_t = I("off_t");
  return L;
}

struct Entry {   // one captured graph (GraphActor.graphs[i]) and its outputs
  std::shared_ptr<at::cuda::CUDAGraph> graph;
  at::Tensor out;        // [n, cols] float32 (GraphActor.outs[i])
  at::Tensor tkv, o;     // [n, L2D + D] state dtype, [n, D] float32 (GraphActor.t_outs[i]; undefined without t_on)
};

}  // namespace

struct TokMiddle {
  abptok::V3 v3;
  abptok::Fold L;
  i64 cols = 0, L2D = 0, D = 0;
  i64 off_schema = 0, off_bytes = 0, off_frame = 0, off_first = 0;
  // GraphActor's static inputs (device) and pinned staging (host): written here, read by every graph
  at::Tensor raw, ctrl, counts, stage_h, ctrl_h, counts_h, out_h;
  std::vector<Entry> graphs;
  int64_t stream_id = 0; int64_t device_index = 0; bool have_stream = false;
  // the output ring (or, in the tests, a rollout's t_kv / t_s / t_o): kv [K, L, 2, D], s [K, D], o [K + obase, D]
  at::Tensor ring_kv, ring_s, ring_o; i64 ring_obase = 0;
  // cf_snap (B34): the actor's TemporalState and the bank's, their six tensors and slot axes (kv: 2, the rest: 0)
  std::vector<at::Tensor> ts_t, bank_t; std::vector<i64> ts_ax;
  // the last call
  i64 last_e_bucket = -1, last_graph = -1, last_rows = 0, calls = 0;

  TokMiddle(py::dict fold, py::dict v3c, py::dict fields, i64 cols_, i64 L2D_, i64 D_)
      : v3(v3_of(v3c)), L(fold_of(fold)), cols(cols_), L2D(L2D_), D(D_) {
    off_schema = fields["visible_schema"].cast<i64>(); off_bytes = fields["visible_bytes"].cast<i64>();
    off_frame = fields["visible_frame"].cast<i64>(); off_first = fields["first"].cast<i64>();
  }

  void set_buffers(at::Tensor raw_, at::Tensor ctrl_, at::Tensor counts_, at::Tensor stage_h_, at::Tensor ctrl_h_,
                   at::Tensor counts_h_, at::Tensor out_h_) {
    TORCH_CHECK(raw_.is_cuda() && ctrl_.is_cuda() && counts_.is_cuda(), "set_buffers: device inputs");
    TORCH_CHECK(stage_h_.is_pinned() && ctrl_h_.is_pinned() && counts_h_.is_pinned() && out_h_.is_pinned(),
                "set_buffers: pinned host buffers");
    TORCH_CHECK(stage_h_.size(0) == L.n && stage_h_.size(1) == L.stage_w() && ctrl_h_.size(1) == L.cw &&
                counts_h_.size(1) == 4 && out_h_.size(1) == cols, "set_buffers: shapes");
    raw = raw_; ctrl = ctrl_; counts = counts_; stage_h = stage_h_; ctrl_h = ctrl_h_; counts_h = counts_h_; out_h = out_h_;
  }
  void set_stream(int64_t sid, int64_t dev) { stream_id = sid; device_index = dev; have_stream = true; }
  void register_graph(i64 idx, std::shared_ptr<at::cuda::CUDAGraph> g, at::Tensor out, c10::optional<at::Tensor> tkv,
                      c10::optional<at::Tensor> o) {
    if ((i64)graphs.size() <= idx) graphs.resize(idx+1);
    TORCH_CHECK(out.size(1) == cols, "register_graph: output columns");
    graphs[idx] = Entry{g, out, tkv ? *tkv : at::Tensor(), o ? *o : at::Tensor()};
  }
  void set_ring(at::Tensor kv, at::Tensor s, at::Tensor o, i64 obase) { ring_kv = kv; ring_s = s; ring_o = o; ring_obase = obase; }
  void set_cf(std::vector<at::Tensor> ts, std::vector<at::Tensor> bank) {
    TORCH_CHECK(ts.size() == 6 && bank.size() == 6, "set_cf: TemporalState's six tensors");
    ts_t = ts; bank_t = bank; ts_ax = {2, 0, 0, 0, 0, 0};
  }
  i64 n_graphs() const { i64 k = 0; for (auto &e : graphs) k += e.graph != nullptr; return k; }

  c10::cuda::CUDAStream stream() const {
    TORCH_CHECK(have_stream, "TokMiddle: no stream set");
    return c10::cuda::CUDAStream::unpack3(stream_id, (c10::DeviceIndex)device_index, c10::DeviceType::CUDA);
  }

  // One call (milestone A: synchronous). rows [k, row_bytes] uint8 (each record's ROW bytes; only its prefix is read),
  // pending [k, nh] int64, widx [k] int64 (the temporal slots), ring_rows [k] int64 (-1 in none: no K / V kept),
  // ukey / ut [k] (none: the graph's own uniforms or the records' crn), cf_src / cf_dst (none: no bank copy).
  // Returns (code, record, out [k, cols] float32 numpy, e_bucket, graph).
  py::tuple call(py::array_t<uint8_t, py::array::c_style> rows, py::array_t<i64, py::array::c_style | py::array::forcecast> pending,
                 py::array_t<i64, py::array::c_style | py::array::forcecast> widx, py::object ring_rows, py::object ukey,
                 py::object ut, py::object cf_src, py::object cf_dst) {
    const i64 k = rows.shape(0);
    TORCH_CHECK(rows.ndim() == 2 && rows.shape(1) == L.row_bytes && k >= 1 && k <= L.n, "call: rows [k, row_bytes], 1 <= k <= n");
    TORCH_CHECK(pending.size() == k*L.nh && widx.size() == k, "call: pending [k, nh], widx [k]");
    const uint8_t *rb = rows.data();
    // the v3 check: header words, visible_schema, visible_bytes, P row 0's role / known (P0_WORDS)
    std::vector<float> head((size_t)(8*k)), p0((size_t)(2*k));
    std::vector<double> vs((size_t)k), nb((size_t)k);
    std::vector<i64> pf((size_t)k), cnt((size_t)(4*k), 0);
    const i64 cap = L.row_bytes-off_frame;
    for (i64 i = 0; i < k; i++) {
      const uint8_t *r = rb+i*L.row_bytes;
      std::memcpy(&head[8*i], r+off_frame, 32);
      int32_t s, b; std::memcpy(&s, r+off_schema, 4); std::memcpy(&b, r+off_bytes, 4);
      vs[i] = (double)s; nb[i] = (double)b;
      std::memcpy(&p0[2*i], r+off_frame+4*v3.p0_role, 4);
      std::memcpy(&p0[2*i+1], r+off_frame+4*v3.p0_known, 4);
      // (A-4's take_prefix: up to the frame's end in whole floats; outside 0 .. the field whole)
      pf[i] = (b >= 0 && b <= cap) ? off_frame-L.end_e+(((i64)b+3) & ~(i64)3) : L.row_bytes-L.end_e;
    }
    abptok::CheckOut chk = abptok::check_batch(v3, head.data(), 8, vs.data(), nb.data(), p0.data(), k, true, v3.p_max,
                                               cnt.data());
    if (chk.code) return py::make_tuple(chk.code, chk.record, py::none(), (i64)-1, (i64)-1);
    // the staging (pinned, GraphActor's), the counts, the control block
    std::vector<i64> uk, utv;
    const i64 *ukp = nullptr, *utp = nullptr;
    if (!ukey.is_none()) {
      auto a = ukey.cast<py::array_t<i64, py::array::c_style | py::array::forcecast>>();
      auto b = ut.cast<py::array_t<i64, py::array::c_style | py::array::forcecast>>();
      TORCH_CHECK(a.size() == k && b.size() == k, "call: ukey / ut [k]");
      uk.assign(a.data(), a.data()+k); utv.assign(b.data(), b.data()+k); ukp = uk.data(); utp = utv.data();
    }
    std::vector<i64> dest;
    bool keep = !ring_rows.is_none() && L.t_on;
    if (keep) {
      auto a = ring_rows.cast<py::array_t<i64, py::array::c_style | py::array::forcecast>>();
      TORCH_CHECK(a.size() == k, "call: ring_rows [k]");
      dest.assign(a.data(), a.data()+k);
    }
    uint8_t *st = stage_h.data_ptr<uint8_t>(), *ct = ctrl_h.data_ptr<uint8_t>();
    abptok::FoldOut fo = abptok::fold_pack(L, rb, k, pending.data(), widx.data(), nullptr, keep ? dest.data() : nullptr,
                                           ukp, utp, chk.e_bucket, st, ct, pf.data());
    TORCH_CHECK(fo.g >= 0, "call: no graph for this call (bucket ", chk.e_bucket, ", ", k, " rows)");
    TORCH_CHECK(fo.graph < (i64)graphs.size() && graphs[fo.graph].graph, "call: graph ", fo.graph, " not registered");
    i64 *ch = counts_h.data_ptr<i64>();
    std::memcpy(ch, cnt.data(), (size_t)(32*k));
    std::memset(ch+4*k, 0, (size_t)(32*(L.n-k)));
    Entry &e = graphs[fo.graph];
    std::vector<i64> cs, cd;   // (cf_snap: the actor slots and the bank rows they are copied to)
    if (!cf_src.is_none()) {
      auto a = cf_src.cast<py::array_t<i64, py::array::c_style | py::array::forcecast>>();
      auto b = cf_dst.cast<py::array_t<i64, py::array::c_style | py::array::forcecast>>();
      TORCH_CHECK(!ts_t.empty() && a.size() == b.size(), "call: cf_snap needs set_cf and equal lengths");
      cs.assign(a.data(), a.data()+a.size()); cd.assign(b.data(), b.data()+b.size());
    }
    {
      py::gil_scoped_release nogil;
      c10::InferenceMode im;   // (the graphs' RNG state and inputs are inference tensors)
      auto s = stream();
      c10::cuda::CUDAStreamGuard guard(s);
      // cf_snap (B34): the bank rows from the actor's slots, before this step (on this stream, ahead of the replay)
      if (!cs.empty()) {
        auto dev = ts_t[0].device();
        at::Tensor si = at::tensor(cs, at::kLong).to(dev), di = at::tensor(cd, at::kLong).to(dev);
        for (size_t j = 0; j < 6; j++) bank_t[j].index_copy_(ts_ax[j], di, ts_t[j].index_select(ts_ax[j], si));
      }
      raw.narrow(0, 0, k).copy_(stage_h.narrow(0, 0, k), true);
      counts.copy_(counts_h, true);
      ctrl.narrow(0, 0, fo.r).copy_(ctrl_h.narrow(0, 0, fo.r), true);
      e.graph->replay();
      out_h.narrow(0, 0, k).copy_(e.out.narrow(0, 0, k), true);
      if (keep) {
        TORCH_CHECK(ring_kv.defined() && e.tkv.defined(), "call: ring_rows without a ring / temporal outputs");
        at::Tensor rr = ctrl.narrow(0, 0, k).narrow(1, L.dest_at, 8).view(at::kLong).reshape({k}).contiguous();
        ring_kv.index_copy_(0, rr, e.tkv.narrow(0, 0, k).narrow(1, 0, L2D).reshape(
            [&] { auto sz = ring_kv.sizes().vec(); sz[0] = k; return sz; }()).to(ring_kv.scalar_type()));
        ring_s.index_copy_(0, rr, e.tkv.narrow(0, 0, k).narrow(1, L2D, D).to(ring_s.scalar_type()));
        ring_o.index_copy_(0, rr+ring_obase, e.o.narrow(0, 0, k).to(ring_o.scalar_type()));
      }
      s.synchronize();
    }
    last_e_bucket = chk.e_bucket; last_graph = fo.graph; last_rows = k; calls++;
    py::array_t<float> out({k, cols});
    std::memcpy(out.mutable_data(), out_h.data_ptr<float>(), (size_t)(4*k*cols));
    return py::make_tuple(chk.code, chk.record, out, chk.e_bucket, fo.graph);
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "P package: the thin TokPolicy driver (milestone A: TokMiddle, one call's middle in C++)";
  py::class_<TokMiddle>(m, "TokMiddle")
      .def(py::init<py::dict, py::dict, py::dict, i64, i64, i64>(), py::arg("fold"), py::arg("v3"), py::arg("fields"),
           py::arg("cols"), py::arg("L2D"), py::arg("D"))
      .def("set_buffers", &TokMiddle::set_buffers)
      .def("set_stream", &TokMiddle::set_stream)
      .def("register_graph", &TokMiddle::register_graph, py::arg("idx"), py::arg("graph"), py::arg("out"),
           py::arg("tkv") = py::none(), py::arg("o") = py::none())
      .def("set_ring", &TokMiddle::set_ring)
      .def("set_cf", &TokMiddle::set_cf)
      .def("n_graphs", &TokMiddle::n_graphs)
      .def("call", &TokMiddle::call, py::arg("rows"), py::arg("pending"), py::arg("widx"), py::arg("ring_rows") = py::none(),
           py::arg("ukey") = py::none(), py::arg("ut") = py::none(), py::arg("cf_src") = py::none(),
           py::arg("cf_dst") = py::none())
      .def_readonly("last_e_bucket", &TokMiddle::last_e_bucket).def_readonly("last_graph", &TokMiddle::last_graph)
      .def_readonly("calls", &TokMiddle::calls);
}
