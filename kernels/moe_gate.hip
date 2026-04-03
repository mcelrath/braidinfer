#include <hip/hip_runtime.h>

// Standalone MoE gate kernel: top-k selection + weight computation from pre-computed scores.
// Replaces CPU round-trip (GPU→host copy + CPU softmax/topk + host→GPU copy).
//
// Gate modes:
//   0 = softmax: exp(s)/sum(exp(s)) over top-k, then optionally scale
//   1 = norm_topk: softmax over top-k, then multiply by routed_scaling_factor
//   2 = sigmoid: sigmoid(s) for weights, sigmoid(s)+bias for selection, then renormalize+scale
//
// correction_bias: if non-NULL, added to scores BEFORE top-k selection (not to weights).
//   For sigmoid mode: selection = sigmoid(scores) + bias, weights = sigmoid(scores) renormalized.
//   For softmax mode: scores += bias before softmax computation.
//
// Outputs: expert_ids[k] (int), expert_weights[k] (float)
// Launch: 1 block, 256 threads. Only thread 0 does the serial top-k (k is small: 2-22).

extern "C" __global__ void moe_gate_topk(
    const float* __restrict__ scores,        // [num_experts] — output of gate linear_proj
    int* __restrict__ expert_ids,            // [k] — output
    float* __restrict__ expert_weights,      // [k] — output
    const float* __restrict__ correction_bias, // [num_experts] or NULL
    const int num_experts,
    const int k,
    const int gate_mode,                     // 0=softmax, 1=norm_topk, 2=sigmoid
    const float routed_scaling_factor
) {
    int tid = threadIdx.x;

    extern __shared__ float shmem[];
    float* sh_sel = shmem;        // [num_experts] — selection scores (biased)
    float* sh_raw = shmem + 512;  // [num_experts] — raw scores (for weight computation)

    // Load scores into shared memory (all threads cooperate)
    for (int i = tid; i < num_experts; i += blockDim.x) {
        float s = scores[i];
        sh_raw[i] = s;
        if (gate_mode == 2) {
            // Sigmoid mode: selection uses sigmoid + bias, raw stores sigmoid for weights
            float sig = 1.0f / (1.0f + expf(-s));
            sh_raw[i] = sig;
            sh_sel[i] = correction_bias ? sig + correction_bias[i] : sig;
        } else {
            // Softmax/norm_topk: add correction bias to selection scores
            sh_sel[i] = correction_bias ? s + correction_bias[i] : s;
        }
    }
    __syncthreads();

    if (tid != 0) return;

    // Top-k selection from sh_sel (serial — k is small)
    int ids[32];
    for (int j = 0; j < k; j++) {
        float best = -1e30f;
        int best_idx = 0;
        for (int i = 0; i < num_experts; i++) {
            if (sh_sel[i] > best) {
                best = sh_sel[i];
                best_idx = i;
            }
        }
        ids[j] = best_idx;
        expert_ids[j] = best_idx;
        sh_sel[best_idx] = -1e30f;
    }

    // Compute weights from raw scores of selected experts
    if (gate_mode == 2) {
        // Sigmoid: weights = renormalized sigmoid(raw_scores[selected])
        float sum = 0.0f;
        for (int j = 0; j < k; j++) sum += sh_raw[ids[j]];
        for (int j = 0; j < k; j++) {
            expert_weights[j] = (sh_raw[ids[j]] / sum) * routed_scaling_factor;
        }
    } else {
        // Softmax over selected experts' raw scores (not biased)
        float selected[32];
        float max_s = -1e30f;
        for (int j = 0; j < k; j++) {
            selected[j] = sh_raw[ids[j]];
            if (selected[j] > max_s) max_s = selected[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            selected[j] = expf(selected[j] - max_s);
            sum += selected[j];
        }
        float scale = (gate_mode == 1) ? routed_scaling_factor : 1.0f;
        for (int j = 0; j < k; j++) {
            expert_weights[j] = (selected[j] / sum) * scale;
        }
    }
}
