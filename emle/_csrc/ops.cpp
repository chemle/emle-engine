// SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
// SPDX-License-Identifier: MIT

// Fused operators for EMLE: operator definitions, CPU kernels and autograd.
//
// emle::electrostatics(xq, xm, q, s, q_core, q_val, mask)
//     -> (E_static (B,), fields (B, N, 3), F0 (B, N, 3))
// for QM positions xq (B, N, 3) and MM positions xm (B, M, 3) in Bohr, MM
// charges q (B, M), valence widths s, core and valence charges (B, N) and the
// mask of real QM atoms (B, N):
//     E_static = sum_nm q_m (q_core_n R_nm + q_val_n T0(r_nm, s_n))
//     fields_n = sum_m q_m f1(r'_nm, 2 s_n) T1_nm
//     F0_n     = sum_m q_m T1_nm
// with d = x_m - x_n, r = |d|, R = 1 / (r + 1e-10), r' = 1 / R, T1 = -d R^3,
// T0(r, s) = (1 - (1 + r / (2 s)) exp(-r / s)) / r and
// f1(r, S) = 1 - (1 + x + x^2 / 2) exp(-x), x = r / S
// (the same expressions as EMLEBase._get_mesh_data, get_static_energy and
// _get_mu_ind). The (N, M) pair terms are never stored.
//
// emle::aev(species, xyz, ...) -> (B, N, 1008) TorchANI-style AEVs (CUDA).
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/autograd.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

namespace {

constexpr double EPS = 1e-10;

template <typename scalar_t>
void electrostatics_forward_cpu_impl(const Tensor& xq, const Tensor& xm, const Tensor& q, const Tensor& s,
                                     const Tensor& qc, const Tensor& qv, const Tensor& mask, Tensor& E,
                                     Tensor& fields, Tensor& F0) {
  const int64_t B = xq.size(0), N = xq.size(1), M = xm.size(1);
  auto XQ = xq.accessor<scalar_t, 3>(); auto XM = xm.accessor<scalar_t, 3>();
  auto Q = q.accessor<scalar_t, 2>(); auto S = s.accessor<scalar_t, 2>();
  auto QC = qc.accessor<scalar_t, 2>(); auto QV = qv.accessor<scalar_t, 2>();
  auto MK = mask.accessor<bool, 2>();
  auto FA = fields.accessor<scalar_t, 3>(); auto F0A = F0.accessor<scalar_t, 3>();
  std::vector<double> e_rows(B * N, 0.0);
  at::parallel_for(0, B * N, 1, [&](int64_t begin, int64_t end) {
    for (int64_t bn = begin; bn < end; ++bn) {
      const int64_t b = bn / N, n = bn % N;
      double es = 0, f[3] = {0, 0, 0}, f0[3] = {0, 0, 0};
      if (MK[b][n]) {
        const double sn = S[b][n], qcn = QC[b][n], qvn = QV[b][n];
        const double x0 = XQ[b][n][0], x1 = XQ[b][n][1], x2 = XQ[b][n][2];
        for (int64_t m = 0; m < M; ++m) {
          const double d0 = XM[b][m][0] - x0, d1 = XM[b][m][1] - x1, d2 = XM[b][m][2] - x2;
          const double r = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
          const double qm = Q[b][m];
          const double R = 1.0 / (r + EPS);
          const double T0 = (1.0 - (1.0 + r / (sn * 2.0)) * std::exp(-r / sn)) / r;
          es += qm * (qcn * R + qvn * T0);
          const double R3 = R * R * R;
          const double x = (1.0 / R) / (2.0 * sn);
          const double f1 = 1.0 - (1.0 + x * (1.0 + 0.5 * x)) * std::exp(-x);
          const double w0 = -qm * R3, w1 = w0 * f1;
          f0[0] += w0 * d0; f0[1] += w0 * d1; f0[2] += w0 * d2;
          f[0] += w1 * d0; f[1] += w1 * d1; f[2] += w1 * d2;
        }
      }
      e_rows[bn] = es;
      for (int k = 0; k < 3; ++k) { FA[b][n][k] = f[k]; F0A[b][n][k] = f0[k]; }
    }
  });
  auto EA = E.accessor<scalar_t, 1>();
  for (int64_t b = 0; b < B; ++b) {
    double e = 0;
    for (int64_t n = 0; n < N; ++n) e += e_rows[b * N + n];
    EA[b] = e;
  }
}

// Gradient contributions of one (QM, MM) pair; see the CUDA version.
inline void pair_grad(double d0, double d1, double d2, double qm, double sn, double qcn, double qvn, double ge,
                      const double* gf, const double* gf0, double* dd, double& dq, double& ds, double& dqc,
                      double& dqv) {
  const double r = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
  if (r == 0.0) { dd[0] = dd[1] = dd[2] = dq = ds = dqc = dqv = 0.0; return; }
  const double R = 1.0 / (r + EPS), R3 = R * R * R;
  const double xs = r / sn, ex = std::exp(-xs);
  const double T0 = (1.0 - (1.0 + r / (sn * 2.0)) * ex) / r;
  const double dT0_dr = ex * (1.0 + xs) / (2.0 * sn * r) - T0 / r;
  const double dT0_ds = -ex * (1.0 + xs) / (2.0 * sn * sn);
  dqc = ge * qm * R;
  dqv = ge * qm * T0;
  ds = ge * qm * qvn * dT0_ds;
  double dL_dr = ge * qm * (-qcn * R * R + qvn * dT0_dr);
  dq = ge * (qcn * R + qvn * T0);
  const double x = (1.0 / R) / (2.0 * sn), e2 = std::exp(-x);
  const double f1 = 1.0 - (1.0 + x * (1.0 + 0.5 * x)) * e2;
  const double df1_dx = 0.5 * x * x * e2;
  const double d[3] = {d0, d1, d2};
  double G[3], gfT1 = 0, gT1 = 0, dG = 0;
  for (int k = 0; k < 3; ++k) {
    const double T1 = -d[k] * R3;
    G[k] = qm * (f1 * gf[k] + gf0[k]);
    gfT1 += gf[k] * T1;
    gT1 += gf0[k] * T1;
    dG += d[k] * G[k];
  }
  const double dL_df1 = qm * gfT1;
  dq += f1 * gfT1 + gT1;
  dL_dr += dL_df1 * df1_dx / (2.0 * sn);
  ds += dL_df1 * df1_dx * (-x / sn);
  const double c = dL_dr / r + 3.0 * R3 * R * dG / r;
  for (int k = 0; k < 3; ++k) dd[k] = c * d[k] - R3 * G[k];
}

template <typename scalar_t>
void electrostatics_backward_cpu_impl(const Tensor& xq, const Tensor& xm, const Tensor& q, const Tensor& s,
                                      const Tensor& qc, const Tensor& qv, const Tensor& mask, const Tensor& gE,
                                      const Tensor& gF, const Tensor& gF0, Tensor& g_xq, Tensor& g_xm, Tensor& g_q,
                                      Tensor& g_s, Tensor& g_qc, Tensor& g_qv) {
  const int64_t B = xq.size(0), N = xq.size(1), M = xm.size(1);
  auto XQ = xq.accessor<scalar_t, 3>(); auto XM = xm.accessor<scalar_t, 3>();
  auto Q = q.accessor<scalar_t, 2>(); auto S = s.accessor<scalar_t, 2>();
  auto QC = qc.accessor<scalar_t, 2>(); auto QV = qv.accessor<scalar_t, 2>();
  auto MK = mask.accessor<bool, 2>();
  auto GE = gE.accessor<scalar_t, 1>(); auto GF = gF.accessor<scalar_t, 3>(); auto GF0 = gF0.accessor<scalar_t, 3>();
  auto GXQ = g_xq.accessor<scalar_t, 3>(); auto GS = g_s.accessor<scalar_t, 2>();
  auto GQC = g_qc.accessor<scalar_t, 2>(); auto GQV = g_qv.accessor<scalar_t, 2>();
  auto GXM = g_xm.accessor<scalar_t, 3>(); auto GQ = g_q.accessor<scalar_t, 2>();
  // Per-QM-atom gradients: reduce over MM atoms.
  at::parallel_for(0, B * N, 1, [&](int64_t begin, int64_t end) {
    for (int64_t bn = begin; bn < end; ++bn) {
      const int64_t b = bn / N, n = bn % N;
      double gx[3] = {0, 0, 0}, gs = 0, gqc = 0, gqv = 0;
      if (MK[b][n]) {
        const double gf[3] = {GF[b][n][0], GF[b][n][1], GF[b][n][2]};
        const double gf0[3] = {GF0[b][n][0], GF0[b][n][1], GF0[b][n][2]};
        for (int64_t m = 0; m < M; ++m) {
          double dd[3], dq, ds, dqc, dqv;
          pair_grad(XM[b][m][0] - XQ[b][n][0], XM[b][m][1] - XQ[b][n][1], XM[b][m][2] - XQ[b][n][2], Q[b][m],
                    S[b][n], QC[b][n], QV[b][n], GE[b], gf, gf0, dd, dq, ds, dqc, dqv);
          for (int k = 0; k < 3; ++k) gx[k] -= dd[k];
          gs += ds; gqc += dqc; gqv += dqv;
        }
      }
      for (int k = 0; k < 3; ++k) GXQ[b][n][k] = gx[k];
      GS[b][n] = gs; GQC[b][n] = gqc; GQV[b][n] = gqv;
    }
  });
  // Per-MM-atom gradients: reduce over QM atoms.
  at::parallel_for(0, B * M, 64, [&](int64_t begin, int64_t end) {
    for (int64_t bm = begin; bm < end; ++bm) {
      const int64_t b = bm / M, m = bm % M;
      double a[4] = {0, 0, 0, 0};
      for (int64_t n = 0; n < N; ++n) {
        if (!MK[b][n]) continue;
        const double gf[3] = {GF[b][n][0], GF[b][n][1], GF[b][n][2]};
        const double gf0[3] = {GF0[b][n][0], GF0[b][n][1], GF0[b][n][2]};
        double dd[3], dq, ds, dqc, dqv;
        pair_grad(XM[b][m][0] - XQ[b][n][0], XM[b][m][1] - XQ[b][n][1], XM[b][m][2] - XQ[b][n][2], Q[b][m],
                  S[b][n], QC[b][n], QV[b][n], GE[b], gf, gf0, dd, dq, ds, dqc, dqv);
        for (int k = 0; k < 3; ++k) a[k] += dd[k];
        a[3] += dq;
      }
      for (int k = 0; k < 3; ++k) GXM[b][m][k] = a[k];
      GQ[b][m] = a[3];
    }
  });
}

std::tuple<Tensor, Tensor, Tensor> electrostatics_fwd_cpu(const Tensor& xq_, const Tensor& xm_, const Tensor& q_,
                                                          const Tensor& s_, const Tensor& qc_, const Tensor& qv_,
                                                          const Tensor& mask_) {
  auto xq = xq_.contiguous(), xm = xm_.contiguous(), q = q_.contiguous(), s = s_.contiguous();
  auto qc = qc_.contiguous(), qv = qv_.contiguous(), mask = mask_.contiguous();
  auto E = torch::zeros({xq.size(0)}, xq.options());
  auto fields = torch::zeros_like(xq), F0 = torch::zeros_like(xq);
  AT_DISPATCH_FLOATING_TYPES(xq.scalar_type(), "electrostatics_fwd_cpu", [&] {
    electrostatics_forward_cpu_impl<scalar_t>(xq, xm, q, s, qc, qv, mask, E, fields, F0);
  });
  return {E, fields, F0};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> electrostatics_bwd_cpu(
    const Tensor& xq_, const Tensor& xm_, const Tensor& q_, const Tensor& s_, const Tensor& qc_, const Tensor& qv_,
    const Tensor& mask_, const Tensor& gE_, const Tensor& gF_, const Tensor& gF0_) {
  auto xq = xq_.contiguous(), xm = xm_.contiguous(), q = q_.contiguous(), s = s_.contiguous();
  auto qc = qc_.contiguous(), qv = qv_.contiguous(), mask = mask_.contiguous();
  auto gE = gE_.contiguous(), gF = gF_.contiguous(), gF0 = gF0_.contiguous();
  auto g_xq = torch::zeros_like(xq), g_xm = torch::zeros_like(xm), g_q = torch::zeros_like(q);
  auto g_s = torch::zeros_like(s), g_qc = torch::zeros_like(qc), g_qv = torch::zeros_like(qv);
  AT_DISPATCH_FLOATING_TYPES(xq.scalar_type(), "electrostatics_bwd_cpu", [&] {
    electrostatics_backward_cpu_impl<scalar_t>(xq, xm, q, s, qc, qv, mask, gE, gF, gF0, g_xq, g_xm, g_q, g_s,
                                               g_qc, g_qv);
  });
  return {g_xq, g_xm, g_q, g_s, g_qc, g_qv};
}

// Autograd wrappers: the forward and backward kernels are separate operators
// (dispatched to the CPU or CUDA implementation).
class ElectrostaticsFunction : public torch::autograd::Function<ElectrostaticsFunction> {
 public:
  static tensor_list forward(AutogradContext* ctx, Tensor xq, Tensor xm, Tensor q, Tensor s, Tensor qc, Tensor qv,
                             Tensor mask) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("emle::electrostatics_fwd", "")
                         .typed<std::tuple<Tensor, Tensor, Tensor>(const Tensor&, const Tensor&, const Tensor&,
                                                                   const Tensor&, const Tensor&, const Tensor&,
                                                                   const Tensor&)>();
    at::AutoDispatchBelowADInplaceOrView guard;
    auto out = op.call(xq, xm, q, s, qc, qv, mask);
    ctx->save_for_backward({xq, xm, q, s, qc, qv, mask});
    return {std::get<0>(out), std::get<1>(out), std::get<2>(out)};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list g) {
    TORCH_CHECK(!torch::GradMode::is_enabled(), "The fused EMLE operators do not support double backward");
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("emle::electrostatics_bwd", "")
                         .typed<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(
                             const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&)>();
    auto v = ctx->get_saved_variables();
    auto gE = g[0].defined() ? g[0] : torch::zeros({v[0].size(0)}, v[0].options());
    auto gF = g[1].defined() ? g[1] : torch::zeros_like(v[0]);
    auto gF0 = g[2].defined() ? g[2] : torch::zeros_like(v[0]);
    auto r = op.call(v[0], v[1], v[2], v[3], v[4], v[5], v[6], gE, gF, gF0);
    return {std::get<0>(r), std::get<1>(r), std::get<2>(r), std::get<3>(r), std::get<4>(r), std::get<5>(r),
            Tensor()};
  }
};

std::tuple<Tensor, Tensor, Tensor> electrostatics_autograd(const Tensor& xq, const Tensor& xm, const Tensor& q,
                                                           const Tensor& s, const Tensor& qc, const Tensor& qv,
                                                           const Tensor& mask) {
  auto r = ElectrostaticsFunction::apply(xq, xm, q, s, qc, qv, mask);
  return {r[0], r[1], r[2]};
}

std::tuple<Tensor, Tensor, Tensor> electrostatics_no_autograd(const Tensor& xq, const Tensor& xm, const Tensor& q,
                                                              const Tensor& s, const Tensor& qc, const Tensor& qv,
                                                              const Tensor& mask) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("emle::electrostatics_fwd", "")
                       .typed<std::tuple<Tensor, Tensor, Tensor>(const Tensor&, const Tensor&, const Tensor&,
                                                                 const Tensor&, const Tensor&, const Tensor&,
                                                                 const Tensor&)>();
  return op.call(xq, xm, q, s, qc, qv, mask);
}

class AEVFunction : public torch::autograd::Function<AEVFunction> {
 public:
  static tensor_list forward(AutogradContext* ctx, Tensor species, Tensor xyz, std::vector<double> shf_r,
                             std::vector<double> shf_a, std::vector<double> shf_z, double eta_r, double rc_r,
                             double eta_a, double zeta, double rc_a) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("emle::aev_fwd", "")
                         .typed<Tensor(const Tensor&, const Tensor&, c10::ArrayRef<double>, c10::ArrayRef<double>,
                                       c10::ArrayRef<double>, double, double, double, double, double)>();
    at::AutoDispatchBelowADInplaceOrView guard;
    auto out = op.call(species, xyz, shf_r, shf_a, shf_z, eta_r, rc_r, eta_a, zeta, rc_a);
    ctx->save_for_backward({species, xyz});
    ctx->saved_data["shf_r"] = shf_r; ctx->saved_data["shf_a"] = shf_a; ctx->saved_data["shf_z"] = shf_z;
    ctx->saved_data["eta_r"] = eta_r; ctx->saved_data["rc_r"] = rc_r; ctx->saved_data["eta_a"] = eta_a;
    ctx->saved_data["zeta"] = zeta; ctx->saved_data["rc_a"] = rc_a;
    return {out};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list g) {
    TORCH_CHECK(!torch::GradMode::is_enabled(), "The fused EMLE operators do not support double backward");
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("emle::aev_bwd", "")
                         .typed<Tensor(const Tensor&, const Tensor&, const Tensor&, c10::ArrayRef<double>,
                                       c10::ArrayRef<double>, c10::ArrayRef<double>, double, double, double, double,
                                       double)>();
    auto v = ctx->get_saved_variables();
    auto& d = ctx->saved_data;
    auto shf_r = d["shf_r"].toDoubleVector(), shf_a = d["shf_a"].toDoubleVector(), shf_z = d["shf_z"].toDoubleVector();
    auto gx = op.call(v[0], v[1], g[0], shf_r, shf_a, shf_z, d["eta_r"].toDouble(), d["rc_r"].toDouble(),
                      d["eta_a"].toDouble(), d["zeta"].toDouble(), d["rc_a"].toDouble());
    return {Tensor(), gx, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};

Tensor aev_autograd(const Tensor& species, const Tensor& xyz, c10::ArrayRef<double> shf_r,
                    c10::ArrayRef<double> shf_a, c10::ArrayRef<double> shf_z, double eta_r, double rc_r,
                    double eta_a, double zeta, double rc_a) {
  return AEVFunction::apply(species, xyz, shf_r.vec(), shf_a.vec(), shf_z.vec(), eta_r, rc_r, eta_a, zeta,
                            rc_a)[0];
}

Tensor aev_no_autograd(const Tensor& species, const Tensor& xyz, c10::ArrayRef<double> shf_r,
                       c10::ArrayRef<double> shf_a, c10::ArrayRef<double> shf_z, double eta_r, double rc_r,
                       double eta_a, double zeta, double rc_a) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("emle::aev_fwd", "")
                       .typed<Tensor(const Tensor&, const Tensor&, c10::ArrayRef<double>, c10::ArrayRef<double>,
                                     c10::ArrayRef<double>, double, double, double, double, double)>();
  return op.call(species, xyz, shf_r, shf_a, shf_z, eta_r, rc_r, eta_a, zeta, rc_a);
}

}  // namespace

TORCH_LIBRARY(emle, m) {
  m.def("electrostatics_fwd(Tensor xq, Tensor xm, Tensor q, Tensor s, Tensor q_core, Tensor q_val, Tensor mask)"
        " -> (Tensor, Tensor, Tensor)");
  m.def("electrostatics_bwd(Tensor xq, Tensor xm, Tensor q, Tensor s, Tensor q_core, Tensor q_val, Tensor mask,"
        " Tensor grad_E, Tensor grad_fields, Tensor grad_F0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("electrostatics(Tensor xq, Tensor xm, Tensor q, Tensor s, Tensor q_core, Tensor q_val, Tensor mask)"
        " -> (Tensor, Tensor, Tensor)");
  m.def("aev_fwd(Tensor species, Tensor xyz, float[] shf_r, float[] shf_a, float[] shf_z, float eta_r,"
        " float rc_r, float eta_a, float zeta, float rc_a) -> Tensor");
  m.def("aev_bwd(Tensor species, Tensor xyz, Tensor grad, float[] shf_r, float[] shf_a, float[] shf_z,"
        " float eta_r, float rc_r, float eta_a, float zeta, float rc_a) -> Tensor");
  m.def("aev(Tensor species, Tensor xyz, float[] shf_r, float[] shf_a, float[] shf_z, float eta_r, float rc_r,"
        " float eta_a, float zeta, float rc_a) -> Tensor");
}

TORCH_LIBRARY_IMPL(emle, CPU, m) {
  m.impl("electrostatics_fwd", &electrostatics_fwd_cpu);
  m.impl("electrostatics_bwd", &electrostatics_bwd_cpu);
  m.impl("electrostatics", &electrostatics_no_autograd);
}

TORCH_LIBRARY_IMPL(emle, Autograd, m) {
  m.impl("electrostatics", &electrostatics_autograd);
  m.impl("aev", &aev_autograd);
}


// The CUDA dispatch of the composite ops (when autograd is not involved).
TORCH_LIBRARY_IMPL(emle, CUDA, m) {
  m.impl("electrostatics", &electrostatics_no_autograd);
  m.impl("aev", &aev_no_autograd);
}

// Python module, so that an ahead-of-time build (setup.py) can be imported as
// emle._C, which registers the operators above.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Fused EMLE operators (registered as torch.ops.emle.*).";
#ifdef EMLE_WITH_CUDA
  m.attr("with_cuda") = true;
#else
  m.attr("with_cuda") = false;
#endif
}
