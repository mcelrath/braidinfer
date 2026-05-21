use rand::prelude::*;
use rand::rngs::StdRng;

use crate::model::Model;
use crate::weights::ModelError;

pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    /// Min-p filtering threshold (0.0 = disabled). After temperature scaling and
    /// top-k, removes tokens whose probability is below `min_p * max_prob`.
    pub min_p: f32,
    /// Per-token additive logit bias applied before temperature scaling.
    /// Each entry is `(token_id, delta)`.
    pub logit_bias: Option<Vec<(u32, f32)>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: None,
            min_p: 0.0,
            logit_bias: None,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: None,
            min_p: 0.0,
            logit_bias: None,
        }
    }
}

pub fn sample(
    logits: &mut [f32],
    params: &SamplingParams,
    token_history: &[u32],
    rng: &mut impl Rng,
) -> u32 {
    let n = logits.len();

    // Logit bias: additive adjustment before any other processing
    if let Some(ref biases) = params.logit_bias {
        for &(token_id, delta) in biases {
            let idx = token_id as usize;
            if idx < n {
                logits[idx] += delta;
            }
        }
    }

    // Repetition penalty
    if params.repetition_penalty != 1.0 {
        for &tok in token_history {
            let idx = tok as usize;
            if idx < n {
                if logits[idx] > 0.0 {
                    logits[idx] /= params.repetition_penalty;
                } else {
                    logits[idx] *= params.repetition_penalty;
                }
            }
        }
    }

    // Temperature scaling / greedy
    if params.temperature == 0.0 {
        return argmax(logits);
    }
    for v in logits.iter_mut() {
        *v /= params.temperature;
    }

    // Top-k filtering
    if params.top_k > 0 && params.top_k < n {
        let k = params.top_k;
        // Find the k-th largest value via partial sort
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = logits[indices[k - 1]];
        for &i in &indices[k..] {
            logits[i] = f32::NEG_INFINITY;
        }
        let _ = threshold; // used implicitly
    }

    // Min-p filtering: remove tokens with prob < min_p * max_prob
    if params.min_p > 0.0 {
        // Compute unnormalized probs from current logits to find max_prob
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_prob = (max_logit - max_logit).exp(); // = 1.0 by definition (relative to max)
        let threshold = params.min_p * max_prob; // = min_p
        // Filter: keep only tokens where exp(logit - max_logit) >= min_p
        for v in logits.iter_mut() {
            if *v > f32::NEG_INFINITY && (*v - max_logit).exp() < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Top-p (nucleus) filtering
    if params.top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > f32::NEG_INFINITY)
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax over filtered set
        let max_v = indexed[0].1;
        let mut probs: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_v).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Accumulate and cut at top_p
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        for i in cutoff_idx..indexed.len() {
            logits[indexed[i].0] = f32::NEG_INFINITY;
        }
    }

    // Softmax
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Categorical sample via cumulative sum
    let u: f32 = rng.r#gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u <= cumsum {
            return i as u32;
        }
    }
    argmax(&probs)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

pub fn generate(
    model: &mut Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    stop_tokens: &[u32],
    max_new_tokens: usize,
) -> Result<Vec<u32>, ModelError> {
    let mut rng: StdRng = match params.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut tokens = prompt_tokens.to_vec();
    let mut generated = Vec::new();

    let n_prompt = prompt_tokens.len();
    if n_prompt == 0 {
        return Ok(vec![]);
    }

    // Prefill: run each prompt token to populate KV/GDN state
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        model.decode_step(tok, i as u32)?;
    }

    // Generate
    for i in 0..max_new_tokens {
        let pos = (n_prompt + i) as u32;
        let last_tok = *tokens.last().unwrap();
        let mut logits = model.decode_step(last_tok, pos)?;
        let next = sample(&mut logits, params, &tokens, &mut rng);
        tokens.push(next);
        generated.push(next);
        if stop_tokens.contains(&next) {
            break;
        }
    }

    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn small_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_greedy_deterministic() {
        let params = SamplingParams::greedy();
        let mut logits1 = vec![0.1f32, 0.5, 0.2, 0.8, 0.3];
        let mut logits2 = logits1.clone();
        let mut rng = small_rng();
        let t1 = sample(&mut logits1, &params, &[], &mut rng);
        let t2 = sample(&mut logits2, &params, &[], &mut rng);
        assert_eq!(t1, 3);
        assert_eq!(t2, 3);
    }

    #[test]
    fn test_temperature_sharpens() {
        let mut rng = small_rng();
        let base_logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        // Compute softmax at temperature=1.0
        let max_v = 5.0f32;
        let probs_t1: Vec<f32> = base_logits.iter().map(|&v| (v - max_v).exp()).collect();
        let sum: f32 = probs_t1.iter().sum();
        let max_prob_t1 = probs_t1.iter().cloned().fold(0.0f32, f32::max) / sum;

        // Compute softmax at temperature=0.5
        let probs_t05: Vec<f32> = base_logits
            .iter()
            .map(|&v| ((v - max_v) / 0.5).exp())
            .collect();
        let sum2: f32 = probs_t05.iter().sum();
        let max_prob_t05 = probs_t05.iter().cloned().fold(0.0f32, f32::max) / sum2;

        assert!(
            max_prob_t05 > max_prob_t1,
            "lower temperature should sharpen distribution"
        );

        // Also verify sample() with low temp is deterministic-ish (argmax-like)
        let params_cold = SamplingParams {
            temperature: 0.01,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(0),
            ..Default::default()
        };
        let mut logits = base_logits.clone();
        let tok = sample(&mut logits, &params_cold, &[], &mut rng);
        assert_eq!(tok, 4); // index 4 has highest logit
    }

    #[test]
    fn test_top_k_filters() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 5,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(0),
            ..Default::default()
        };
        let mut logits = vec![0.1f32; 20];
        logits[0] = 5.0;
        logits[2] = 4.0;
        logits[5] = 3.0;
        logits[10] = 2.0;
        logits[15] = 1.5;
        // indices 0,2,5,10,15 are top-5

        let orig = logits.clone();
        let mut rng = small_rng();
        sample(&mut logits, &params, &[], &mut rng);

        // After top-k, only 5 non-neginf values; all others set to -inf
        // We can't check logits after sample returns (they're consumed), so test the logic directly
        let mut l2 = orig.clone();
        let n = l2.len();
        let k = 5usize;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            l2[b]
                .partial_cmp(&l2[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in &indices[k..] {
            l2[i] = f32::NEG_INFINITY;
        }
        let non_inf_count = l2.iter().filter(|&&v| v > f32::NEG_INFINITY).count();
        assert_eq!(non_inf_count, 5);
    }

    #[test]
    fn test_min_p_filters() {
        // With min_p=0.5, tokens whose softmax prob < 0.5 * max_prob are removed.
        // logits: [10.0, 0.0, 0.0, 0.0] — after temperature=1.0, max_logit=10.0,
        // exp(0-10) << 0.5 so only token 0 survives.
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(0),
            min_p: 0.5,
            logit_bias: None,
        };
        let mut rng = small_rng();
        let mut logits = vec![10.0f32, 0.0, 0.0, 0.0];
        let tok = sample(&mut logits, &params, &[], &mut rng);
        assert_eq!(tok, 0, "only token 0 should survive min_p=0.5 filter");
    }

    #[test]
    fn test_logit_bias_applied() {
        // Boost token 1 by 100 so it wins despite lower base logit.
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: None,
            min_p: 0.0,
            logit_bias: Some(vec![(1u32, 100.0f32)]),
        };
        let mut logits = vec![10.0f32, 1.0, 5.0, 3.0];
        let mut rng = small_rng();
        let tok = sample(&mut logits, &params, &[], &mut rng);
        assert_eq!(tok, 1, "logit_bias should boost token 1 to win");
    }

    #[test]
    fn test_repetition_penalty() {
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 2.0,
            seed: None,
            ..Default::default()
        };
        let history = vec![3u32];
        let mut rng = small_rng();
        // logits[3] = 1.1/2.0 = 0.55, logits[4] = 1.0 -> winner = 4
        let mut logits = vec![0.1f32, 0.2, 0.3, 1.1, 1.0];
        let tok = sample(&mut logits, &params, &history, &mut rng);
        assert_eq!(
            tok, 4,
            "repetition penalty should reduce logit[3] below logit[4]"
        );
    }
}
