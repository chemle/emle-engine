// SPDX-FileCopyrightText: 2023-2026 The EMLE-Engine developers
// SPDX-License-Identifier: MIT

// Fused operators for EMLE: CUDA kernels (see ops.cpp for the definitions).
#include <torch/extension.h>
#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include <vector>

namespace {

// ---- QM/MM electrostatics ----

#define EPS 1e-10

template <typename T> __device__ __forceinline__ T dexp(T x);
template <> __device__ __forceinline__ float dexp<float>(float x) { return expf(x); }
template <> __device__ __forceinline__ double dexp<double>(double x) { return exp(x); }

template <typename T, int NV>
__device__ void block_reduce(T (&v)[NV], T* sh) {
  // sh: blockDim.x * NV
  const int t = threadIdx.x;
  for (int k = 0; k < NV; ++k) sh[k * blockDim.x + t] = v[k];
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s)
      for (int k = 0; k < NV; ++k) sh[k * blockDim.x + t] += sh[k * blockDim.x + t + s];
    __syncthreads();
  }
  for (int k = 0; k < NV; ++k) v[k] = sh[k * blockDim.x];
}

// One block per (b, n).
template <typename T>
__global__ void fwd_kernel(const T* __restrict__ xq, const T* __restrict__ xm, const T* __restrict__ q,
                           const T* __restrict__ s, const T* __restrict__ qc, const T* __restrict__ qv,
                           const bool* __restrict__ mask, int N, int M,
                           T* __restrict__ e_rows, T* __restrict__ fields, T* __restrict__ F0) {
  extern __shared__ unsigned char smem[];
  T* sh = reinterpret_cast<T*>(smem);
  const int bn = blockIdx.x, b = bn / N;
  T v[7] = {0, 0, 0, 0, 0, 0, 0};  // es, f0..2, f00..02
  if (mask[bn]) {
    const T sn = s[bn], qcn = qc[bn], qvn = qv[bn];
    const T x0 = xq[bn * 3], x1 = xq[bn * 3 + 1], x2 = xq[bn * 3 + 2];
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
      const int bm = b * M + m;
      const T d0 = xm[bm * 3] - x0, d1 = xm[bm * 3 + 1] - x1, d2 = xm[bm * 3 + 2] - x2;
      const T r = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
      const T qm = q[bm];
      const T R = T(1) / (r + T(EPS));
      const T T0 = (T(1) - (T(1) + r / (sn * T(2))) * dexp<T>(-r / sn)) / r;
      v[0] += qm * (qcn * R + qvn * T0);
      const T R3 = R * R * R;
      const T x = (T(1) / R) / (T(2) * sn);
      const T f1 = T(1) - (T(1) + x * (T(1) + T(0.5) * x)) * dexp<T>(-x);
      const T w0 = -qm * R3, w1 = w0 * f1;
      v[1] += w1 * d0; v[2] += w1 * d1; v[3] += w1 * d2;
      v[4] += w0 * d0; v[5] += w0 * d1; v[6] += w0 * d2;
    }
  }
  block_reduce<T, 7>(v, sh);
  if (threadIdx.x == 0) {
    e_rows[bn] = v[0];
    for (int k = 0; k < 3; ++k) { fields[bn * 3 + k] = v[1 + k]; F0[bn * 3 + k] = v[4 + k]; }
  }
}

// Shared per-pair backward terms. Returns dL/dd (3), dL/dq_m, and per-n terms.
template <typename T>
__device__ __forceinline__ void pair_grad(T d0, T d1, T d2, T qm, T sn, T qcn, T qvn, T ge,
                                          const T* gf, const T* gf0, T* dd, T& dq, T& ds, T& dqc, T& dqv) {
  const T r = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
  if (r == T(0)) { dd[0] = dd[1] = dd[2] = dq = ds = dqc = dqv = T(0); return; }
  const T R = T(1) / (r + T(EPS)), R3 = R * R * R;
  const T xs = r / sn, ex = dexp<T>(-xs);
  const T T0 = (T(1) - (T(1) + r / (sn * T(2))) * ex) / r;
  const T dT0_dr = ex * (T(1) + xs) / (T(2) * sn * r) - T0 / r;
  const T dT0_ds = -ex * (T(1) + xs) / (T(2) * sn * sn);
  dqc = ge * qm * R;
  dqv = ge * qm * T0;
  ds = ge * qm * qvn * dT0_ds;
  T dL_dr = ge * qm * (-qcn * R * R + qvn * dT0_dr);
  dq = ge * (qcn * R + qvn * T0);
  const T x = (T(1) / R) / (T(2) * sn), e2 = dexp<T>(-x);
  const T f1 = T(1) - (T(1) + x * (T(1) + T(0.5) * x)) * e2;
  const T df1_dx = T(0.5) * x * x * e2;
  const T d[3] = {d0, d1, d2};
  T G[3], gfT1 = 0, gT1 = 0, dG = 0;
  for (int k = 0; k < 3; ++k) {
    const T T1 = -d[k] * R3;
    G[k] = qm * (f1 * gf[k] + gf0[k]);
    gfT1 += gf[k] * T1;
    gT1 += gf0[k] * T1;
    dG += d[k] * G[k];
  }
  const T dL_df1 = qm * gfT1;
  dq += f1 * gfT1 + gT1;
  dL_dr += dL_df1 * df1_dx / (T(2) * sn);
  ds += dL_df1 * df1_dx * (-x / sn);
  const T c = dL_dr / r + T(3) * R3 * R * dG / r;
  for (int k = 0; k < 3; ++k) dd[k] = c * d[k] - R3 * G[k];
}

// Per-QM-atom gradients: one block per (b, n), reduce over m.
template <typename T>
__global__ void bwd_qm_kernel(const T* __restrict__ xq, const T* __restrict__ xm, const T* __restrict__ q,
                              const T* __restrict__ s, const T* __restrict__ qc, const T* __restrict__ qv,
                              const bool* __restrict__ mask, const T* __restrict__ gE, const T* __restrict__ gF,
                              const T* __restrict__ gF0, int N, int M, T* g_xq, T* g_s, T* g_qc, T* g_qv) {
  extern __shared__ unsigned char smem[];
  T* sh = reinterpret_cast<T*>(smem);
  const int bn = blockIdx.x, b = bn / N;
  T v[6] = {0, 0, 0, 0, 0, 0};  // gx0..2, gs, gqc, gqv
  if (mask[bn]) {
    const T sn = s[bn], qcn = qc[bn], qvn = qv[bn], ge = gE[b];
    const T gf[3] = {gF[bn * 3], gF[bn * 3 + 1], gF[bn * 3 + 2]};
    const T gf0[3] = {gF0[bn * 3], gF0[bn * 3 + 1], gF0[bn * 3 + 2]};
    const T x0 = xq[bn * 3], x1 = xq[bn * 3 + 1], x2 = xq[bn * 3 + 2];
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
      const int bm = b * M + m;
      T dd[3], dq, ds, dqc, dqv;
      pair_grad<T>(xm[bm * 3] - x0, xm[bm * 3 + 1] - x1, xm[bm * 3 + 2] - x2, q[bm], sn, qcn, qvn, ge, gf, gf0,
                   dd, dq, ds, dqc, dqv);
      v[0] -= dd[0]; v[1] -= dd[1]; v[2] -= dd[2];
      v[3] += ds; v[4] += dqc; v[5] += dqv;
    }
  }
  block_reduce<T, 6>(v, sh);
  if (threadIdx.x == 0) {
    for (int k = 0; k < 3; ++k) g_xq[bn * 3 + k] = v[k];
    g_s[bn] = v[3]; g_qc[bn] = v[4]; g_qv[bn] = v[5];
  }
}

// Per-MM-atom gradients: one thread per (b, m), loop over n.
template <typename T>
__global__ void bwd_mm_kernel(const T* __restrict__ xq, const T* __restrict__ xm, const T* __restrict__ q,
                              const T* __restrict__ s, const T* __restrict__ qc, const T* __restrict__ qv,
                              const bool* __restrict__ mask, const T* __restrict__ gE, const T* __restrict__ gF,
                              const T* __restrict__ gF0, int B, int N, int M, T* g_xm, T* g_q) {
  const int bm = blockIdx.x * blockDim.x + threadIdx.x;
  if (bm >= B * M) return;
  const int b = bm / M;
  const T y0 = xm[bm * 3], y1 = xm[bm * 3 + 1], y2 = xm[bm * 3 + 2], qm = q[bm], ge = gE[b];
  T a[4] = {0, 0, 0, 0};
  for (int n = 0; n < N; ++n) {
    const int bn = b * N + n;
    if (!mask[bn]) continue;
    const T gf[3] = {gF[bn * 3], gF[bn * 3 + 1], gF[bn * 3 + 2]};
    const T gf0[3] = {gF0[bn * 3], gF0[bn * 3 + 1], gF0[bn * 3 + 2]};
    T dd[3], dq, ds, dqc, dqv;
    pair_grad<T>(y0 - xq[bn * 3], y1 - xq[bn * 3 + 1], y2 - xq[bn * 3 + 2], qm, s[bn], qc[bn], qv[bn], ge, gf, gf0,
                 dd, dq, ds, dqc, dqv);
    a[0] += dd[0]; a[1] += dd[1]; a[2] += dd[2]; a[3] += dq;
  }
  for (int k = 0; k < 3; ++k) g_xm[bm * 3 + k] = a[k];
  g_q[bm] = a[3];
}

static const int THREADS = 256;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> electrostatics_fwd_cuda(const torch::Tensor& xq_in, const torch::Tensor& xm_in, const torch::Tensor& q_in, const torch::Tensor& s_in,
                                         const torch::Tensor& qc_in, const torch::Tensor& qv_in, const torch::Tensor& mask_in) {
  torch::Tensor xq = xq_in;
  torch::Tensor xm = xm_in;
  torch::Tensor q = q_in;
  torch::Tensor s = s_in;
  torch::Tensor qc = qc_in;
  torch::Tensor qv = qv_in;
  torch::Tensor mask = mask_in;
  const c10::cuda::CUDAGuard guard(xq.device());
  xq = xq.contiguous(); xm = xm.contiguous(); q = q.contiguous(); s = s.contiguous();
  qc = qc.contiguous(); qv = qv.contiguous(); mask = mask.contiguous();
  const int B = xq.size(0), N = xq.size(1), M = xm.size(1);
  auto e_rows = torch::empty({B, N}, xq.options());
  auto fields = torch::empty_like(xq), F0 = torch::empty_like(xq);
  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(xq.scalar_type(), "fused_forward_cuda", [&] {
    fwd_kernel<scalar_t><<<B * N, THREADS, THREADS * 7 * sizeof(scalar_t), stream>>>(
        xq.data_ptr<scalar_t>(), xm.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), s.data_ptr<scalar_t>(),
        qc.data_ptr<scalar_t>(), qv.data_ptr<scalar_t>(), mask.data_ptr<bool>(), N, M,
        e_rows.data_ptr<scalar_t>(), fields.data_ptr<scalar_t>(), F0.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {e_rows.sum(1), fields, F0};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> electrostatics_bwd_cuda(const torch::Tensor& xq_in, const torch::Tensor& xm_in, const torch::Tensor& q_in, const torch::Tensor& s_in,
                                          const torch::Tensor& qc_in, const torch::Tensor& qv_in, const torch::Tensor& mask_in,
                                          const torch::Tensor& gE_in, const torch::Tensor& gF_in, const torch::Tensor& gF0_in) {
  torch::Tensor xq = xq_in;
  torch::Tensor xm = xm_in;
  torch::Tensor q = q_in;
  torch::Tensor s = s_in;
  torch::Tensor qc = qc_in;
  torch::Tensor qv = qv_in;
  torch::Tensor mask = mask_in;
  torch::Tensor gE = gE_in;
  torch::Tensor gF = gF_in;
  torch::Tensor gF0 = gF0_in;
  const c10::cuda::CUDAGuard guard(xq.device());
  xq = xq.contiguous(); xm = xm.contiguous(); q = q.contiguous(); s = s.contiguous();
  qc = qc.contiguous(); qv = qv.contiguous(); mask = mask.contiguous();
  gE = gE.contiguous(); gF = gF.contiguous(); gF0 = gF0.contiguous();
  const int B = xq.size(0), N = xq.size(1), M = xm.size(1);
  auto g_xq = torch::empty_like(xq), g_xm = torch::empty_like(xm), g_q = torch::empty_like(q);
  auto g_s = torch::empty_like(s), g_qc = torch::empty_like(qc), g_qv = torch::empty_like(qv);
  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(xq.scalar_type(), "fused_backward_cuda", [&] {
    bwd_qm_kernel<scalar_t><<<B * N, THREADS, THREADS * 6 * sizeof(scalar_t), stream>>>(
        xq.data_ptr<scalar_t>(), xm.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), s.data_ptr<scalar_t>(),
        qc.data_ptr<scalar_t>(), qv.data_ptr<scalar_t>(), mask.data_ptr<bool>(), gE.data_ptr<scalar_t>(),
        gF.data_ptr<scalar_t>(), gF0.data_ptr<scalar_t>(), N, M, g_xq.data_ptr<scalar_t>(),
        g_s.data_ptr<scalar_t>(), g_qc.data_ptr<scalar_t>(), g_qv.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int blocks = (B * M + THREADS - 1) / THREADS;
    bwd_mm_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
        xq.data_ptr<scalar_t>(), xm.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), s.data_ptr<scalar_t>(),
        qc.data_ptr<scalar_t>(), qv.data_ptr<scalar_t>(), mask.data_ptr<bool>(), gE.data_ptr<scalar_t>(),
        gF.data_ptr<scalar_t>(), gF0.data_ptr<scalar_t>(), B, N, M, g_xm.data_ptr<scalar_t>(), g_q.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {g_xq, g_xm, g_q, g_s, g_qc, g_qv};
}


// ---- AEVs ----

#define AEV_PI 3.14159265358979323846
#define NS 7          // number of species
#define NR 16         // radial shifts
#define NA 8          // angular radial shifts
#define NZ 4          // angle sections
#define RADIAL_LEN (NS * NR)            // 112
#define NPAIRS (NS * (NS + 1) / 2)      // 28
#define AEV_LEN (RADIAL_LEN + NPAIRS * NA * NZ)  // 1008

struct Params {
  double eta_r, rc_r, eta_a, zeta, rc_a;
  double shf_r[NR], shf_a[NA], shf_z[NZ];
};

__device__ __forceinline__ int pair_index(int a, int c) {
  // Row-major upper triangle (incl. diagonal) of NS x NS, as torch.triu_indices.
  if (a > c) { int t = a; a = c; c = t; }
  return a * NS - a * (a - 1) / 2 + (c - a);
}

template <typename T>
__device__ __forceinline__ T fc(T r, T rc) { return T(0.5) * cos(T(AEV_PI) * r / rc) + T(0.5); }
template <typename T>
__device__ __forceinline__ T dfc(T r, T rc) { return -T(0.5) * T(AEV_PI) / rc * sin(T(AEV_PI) * r / rc); }

template <typename T>
__global__ void aev_fwd(const int* __restrict__ species, const T* __restrict__ xyz, int N, Params p, T* __restrict__ out) {
  extern __shared__ unsigned char smem[];
  T* acc = reinterpret_cast<T*>(smem);
  const int bi = blockIdx.x, b = bi / N, i = bi % N;
  for (int f = threadIdx.x; f < AEV_LEN; f += blockDim.x) acc[f] = 0;
  __syncthreads();
  if (species[bi] >= 0) {
    const T xi = xyz[bi * 3], yi = xyz[bi * 3 + 1], zi = xyz[bi * 3 + 2];
    const int* sp = species + b * N;
    const T* X = xyz + b * N * 3;
    // Radial.
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      if (j == i || sp[j] < 0) continue;
      const T dx = X[j * 3] - xi, dy = X[j * 3 + 1] - yi, dz = X[j * 3 + 2] - zi;
      const T r = sqrt(dx * dx + dy * dy + dz * dz);
      if (r > T(p.rc_r)) continue;
      const T f = fc<T>(r, T(p.rc_r));
      for (int k = 0; k < NR; ++k) {
        const T u = r - T(p.shf_r[k]);
        atomicAdd(&acc[sp[j] * NR + k], T(0.25) * exp(-T(p.eta_r) * u * u) * f);
      }
    }
    // Angular: unordered pairs j < k.
    const int npairs = N * (N - 1) / 2;
    for (int t = threadIdx.x; t < npairs; t += blockDim.x) {
      // Map t -> (j, k), j < k.
      int j = int((2 * N - 1 - sqrt(T((2 * N - 1) * (2 * N - 1) - 8 * t))) / 2);
      while (j > 0 && t < j * (2 * N - j - 1) / 2) --j;
      while (t >= (j + 1) * (2 * N - j - 2) / 2) ++j;
      const int k = t - j * (2 * N - j - 1) / 2 + j + 1;
      if (j == i || k == i || sp[j] < 0 || sp[k] < 0) continue;
      const T ax = X[j * 3] - xi, ay = X[j * 3 + 1] - yi, az = X[j * 3 + 2] - zi;
      const T bx = X[k * 3] - xi, by = X[k * 3 + 1] - yi, bz = X[k * 3 + 2] - zi;
      const T r1 = sqrt(ax * ax + ay * ay + az * az), r2 = sqrt(bx * bx + by * by + bz * bz);
      if (r1 > T(p.rc_a) || r2 > T(p.rc_a)) continue;
      const T c = (ax * bx + ay * by + az * bz) / max(r1 * r2, T(1e-10));
      const T theta = acos(T(0.95) * c);
      const T F = fc<T>(r1, T(p.rc_a)) * fc<T>(r2, T(p.rc_a));
      const T mean = T(0.5) * (r1 + r2);
      T Az[NZ];
      for (int z = 0; z < NZ; ++z) Az[z] = T(2) * pow((T(1) + cos(theta - T(p.shf_z[z]))) / T(2), T(p.zeta));
      const int base = RADIAL_LEN + pair_index(sp[j], sp[k]) * NA * NZ;
      for (int a = 0; a < NA; ++a) {
        const T u = mean - T(p.shf_a[a]);
        const T Ra = exp(-T(p.eta_a) * u * u) * F;
        for (int z = 0; z < NZ; ++z) atomicAdd(&acc[base + a * NZ + z], Ra * Az[z]);
      }
    }
  }
  __syncthreads();
  for (int f = threadIdx.x; f < AEV_LEN; f += blockDim.x) out[bi * AEV_LEN + f] = acc[f];
}

template <typename T>
__global__ void aev_bwd(const int* __restrict__ species, const T* __restrict__ xyz, const T* __restrict__ grad,
                        int N, Params p, T* __restrict__ gx) {
  extern __shared__ unsigned char smem[];
  T* g = reinterpret_cast<T*>(smem);
  const int bi = blockIdx.x, b = bi / N, i = bi % N;
  for (int f = threadIdx.x; f < AEV_LEN; f += blockDim.x) g[f] = grad[bi * AEV_LEN + f];
  __syncthreads();
  if (species[bi] < 0) return;
  const T xi = xyz[bi * 3], yi = xyz[bi * 3 + 1], zi = xyz[bi * 3 + 2];
  const int* sp = species + b * N;
  const T* X = xyz + b * N * 3;
  T* GX = gx + b * N * 3;
  // Radial.
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    if (j == i || sp[j] < 0) continue;
    const T d[3] = {X[j * 3] - xi, X[j * 3 + 1] - yi, X[j * 3 + 2] - zi};
    const T r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (r > T(p.rc_r)) continue;
    const T f = fc<T>(r, T(p.rc_r)), df = dfc<T>(r, T(p.rc_r));
    T gr = 0;
    for (int k = 0; k < NR; ++k) {
      const T u = r - T(p.shf_r[k]), e = exp(-T(p.eta_r) * u * u);
      gr += g[sp[j] * NR + k] * T(0.25) * e * (-T(2) * T(p.eta_r) * u * f + df);
    }
    for (int c = 0; c < 3; ++c) {
      const T v = gr * d[c] / r;
      atomicAdd(&GX[j * 3 + c], v);
      atomicAdd(&GX[i * 3 + c], -v);
    }
  }
  // Angular.
  const int npairs = N * (N - 1) / 2;
  for (int t = threadIdx.x; t < npairs; t += blockDim.x) {
    int j = int((2 * N - 1 - sqrt(T((2 * N - 1) * (2 * N - 1) - 8 * t))) / 2);
    while (j > 0 && t < j * (2 * N - j - 1) / 2) --j;
    while (t >= (j + 1) * (2 * N - j - 2) / 2) ++j;
    const int k = t - j * (2 * N - j - 1) / 2 + j + 1;
    if (j == i || k == i || sp[j] < 0 || sp[k] < 0) continue;
    const T a[3] = {X[j * 3] - xi, X[j * 3 + 1] - yi, X[j * 3 + 2] - zi};
    const T bv[3] = {X[k * 3] - xi, X[k * 3 + 1] - yi, X[k * 3 + 2] - zi};
    const T r1 = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    const T r2 = sqrt(bv[0] * bv[0] + bv[1] * bv[1] + bv[2] * bv[2]);
    if (r1 > T(p.rc_a) || r2 > T(p.rc_a)) continue;
    const T dot = a[0] * bv[0] + a[1] * bv[1] + a[2] * bv[2];
    const T rr = max(r1 * r2, T(1e-10));
    const T c = dot / rr;
    const T theta = acos(T(0.95) * c);
    const T dtheta_dc = -T(0.95) / sqrt(T(1) - T(0.9025) * c * c);
    const T f1 = fc<T>(r1, T(p.rc_a)), f2 = fc<T>(r2, T(p.rc_a));
    const T df1 = dfc<T>(r1, T(p.rc_a)), df2 = dfc<T>(r2, T(p.rc_a));
    const T F = f1 * f2, mean = T(0.5) * (r1 + r2);
    T Az[NZ], dAz[NZ];
    for (int z = 0; z < NZ; ++z) {
      const T dt = theta - T(p.shf_z[z]);
      const T h = (T(1) + cos(dt)) / T(2);
      Az[z] = T(2) * pow(h, T(p.zeta));
      dAz[z] = -T(p.zeta) * pow(h, T(p.zeta) - T(1)) * sin(dt);
    }
    const int base = RADIAL_LEN + pair_index(sp[j], sp[k]) * NA * NZ;
    T dL_dc = 0, dL_dm = 0, dL_dF = 0;
    for (int ia = 0; ia < NA; ++ia) {
      const T u = mean - T(p.shf_a[ia]);
      const T R = exp(-T(p.eta_a) * u * u), dR = -T(2) * T(p.eta_a) * u * R;
      for (int z = 0; z < NZ; ++z) {
        const T G = g[base + ia * NZ + z];
        dL_dc += G * R * F * dAz[z] * dtheta_dc;
        dL_dm += G * Az[z] * F * dR;
        dL_dF += G * Az[z] * R;
      }
    }
    const T dL_dr1 = T(0.5) * dL_dm + dL_dF * df1 * f2;
    const T dL_dr2 = T(0.5) * dL_dm + dL_dF * f1 * df2;
    for (int cc = 0; cc < 3; ++cc) {
      const T va = dL_dr1 * a[cc] / r1 + dL_dc * (bv[cc] / rr - c * a[cc] / (r1 * r1));
      const T vb = dL_dr2 * bv[cc] / r2 + dL_dc * (a[cc] / rr - c * bv[cc] / (r2 * r2));
      atomicAdd(&GX[j * 3 + cc], va);
      atomicAdd(&GX[k * 3 + cc], vb);
      atomicAdd(&GX[i * 3 + cc], -(va + vb));
    }
  }
}

static Params make_params(c10::ArrayRef<double> shf_r, c10::ArrayRef<double> shf_a,
                          c10::ArrayRef<double> shf_z, double eta_r, double rc_r, double eta_a,
                          double zeta, double rc_a) {
  TORCH_CHECK(shf_r.size() == NR && shf_a.size() == NA && shf_z.size() == NZ,
              "emle::aev supports ", NR, " radial shifts, ", NA, " angular shifts and ", NZ,
              " angle sections");
  Params p;
  p.eta_r = eta_r; p.rc_r = rc_r; p.eta_a = eta_a; p.zeta = zeta; p.rc_a = rc_a;
  for (int k = 0; k < NR; ++k) p.shf_r[k] = shf_r[k];
  for (int k = 0; k < NA; ++k) p.shf_a[k] = shf_a[k];
  for (int k = 0; k < NZ; ++k) p.shf_z[k] = shf_z[k];
  return p;
}

torch::Tensor aev_fwd_cuda(const torch::Tensor& species_in, const torch::Tensor& xyz_in, c10::ArrayRef<double> shf_r, c10::ArrayRef<double> shf_a,
                          c10::ArrayRef<double> shf_z, double eta_r, double rc_r, double eta_a, double zeta, double rc_a) {
  torch::Tensor species = species_in;
  torch::Tensor xyz = xyz_in;
  const c10::cuda::CUDAGuard guard(xyz.device());
  species = species.to(torch::kInt).contiguous(); xyz = xyz.contiguous();
  const int B = xyz.size(0), N = xyz.size(1);
  auto out = torch::empty({B, N, AEV_LEN}, xyz.options());
  Params p = make_params(shf_r, shf_a, shf_z, eta_r, rc_r, eta_a, zeta, rc_a);
  AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "aev_forward", [&] {
    aev_fwd<scalar_t><<<B * N, 128, AEV_LEN * sizeof(scalar_t), c10::cuda::getCurrentCUDAStream()>>>(
        species.data_ptr<int>(), xyz.data_ptr<scalar_t>(), N, p, out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return out;
}

torch::Tensor aev_bwd_cuda(const torch::Tensor& species_in, const torch::Tensor& xyz_in, const torch::Tensor& grad_in, c10::ArrayRef<double> shf_r,
                           c10::ArrayRef<double> shf_a, c10::ArrayRef<double> shf_z, double eta_r, double rc_r, double eta_a,
                           double zeta, double rc_a) {
  torch::Tensor species = species_in;
  torch::Tensor xyz = xyz_in;
  torch::Tensor grad = grad_in;
  const c10::cuda::CUDAGuard guard(xyz.device());
  species = species.to(torch::kInt).contiguous(); xyz = xyz.contiguous(); grad = grad.contiguous();
  const int B = xyz.size(0), N = xyz.size(1);
  auto gx = torch::zeros_like(xyz);
  Params p = make_params(shf_r, shf_a, shf_z, eta_r, rc_r, eta_a, zeta, rc_a);
  AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "aev_backward", [&] {
    aev_bwd<scalar_t><<<B * N, 128, AEV_LEN * sizeof(scalar_t), c10::cuda::getCurrentCUDAStream()>>>(
        species.data_ptr<int>(), xyz.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), N, p, gx.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return gx;
}


}  // namespace

TORCH_LIBRARY_IMPL(emle, CUDA, m) {
  m.impl("electrostatics_fwd", &electrostatics_fwd_cuda);
  m.impl("electrostatics_bwd", &electrostatics_bwd_cuda);
  m.impl("aev_fwd", &aev_fwd_cuda);
  m.impl("aev_bwd", &aev_bwd_cuda);
}
