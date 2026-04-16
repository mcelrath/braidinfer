use braidinfer_hip::memory::DeviceBuffer;

use crate::paged_kv::{self, RecurrentCheckpointPool};

use super::Model;
use super::ModelError;
use crate::config::*;
use crate::weights::*;

impl Model {
    /// Save the current GDN recurrent states into a checkpoint pool slot.
    /// Lazy-initializes the pool on first call. Returns the slot index.
    pub fn save_recurrent_checkpoint(&mut self) -> Result<u32, ModelError> {
        if self.checkpoint_pool.is_none() {
            // Pool capacity 1: prefill uses ring-buffer overwrite (only most-recent needed).
            // Speculative decode (future) may increase this.
            self.checkpoint_pool =
                Some(RecurrentCheckpointPool::new(self.device, &self.config, 1)?);
        }
        // Free previous slot before allocating new one (ring buffer with capacity 1)
        if let Some(prev) = self.last_checkpoint_slot.take() {
            self.checkpoint_pool.as_mut().unwrap().free(prev);
        }
        let recurrent_bufs: Vec<&DeviceBuffer<f32>> =
            self.gdn_states.iter().map(|s| &s.recurrent).collect();
        let pool = self.checkpoint_pool.as_mut().unwrap();
        let slot = paged_kv::save_checkpoint(pool, &recurrent_bufs, self.stream.raw())?;
        self.last_checkpoint_slot = Some(slot);
        Ok(slot)
    }

    /// Process a sequence of tokens (prefill). Returns logits for the last token.
    /// Saves GDN checkpoints at each 64-token chunk boundary.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::MissingWeight("empty token sequence".into()));
        }
        let mut logits = vec![];
        for (i, &tok) in tokens.iter().enumerate() {
            logits = self.decode_step(tok, i as u32)?;
        }
        Ok(logits)
    }

    /// Read all GDN recurrent state to host (for testing).
    pub fn read_gdn_state(&self) -> Result<Vec<Vec<f32>>, ModelError> {
        self.stream.synchronize()?;
        let mut result = Vec::with_capacity(self.gdn_states.len());
        for state in &self.gdn_states {
            let n = state.recurrent.len();
            let mut buf = vec![0.0f32; n];
            state.recurrent.copy_to_host(&mut buf)?;
            result.push(buf);
        }
        Ok(result)
    }

    /// Restore GDN recurrent states from a previously saved checkpoint slot.
    pub fn restore_recurrent_checkpoint(&mut self, slot: u32) -> Result<(), ModelError> {
        let pool = self
            .checkpoint_pool
            .as_ref()
            .ok_or_else(|| ModelError::MissingWeight("checkpoint_pool not initialized".into()))?;
        let mut recurrent_bufs: Vec<&mut DeviceBuffer<f32>> = self
            .gdn_states
            .iter_mut()
            .map(|s| &mut s.recurrent)
            .collect();
        let stream_raw = self.stream.raw();
        paged_kv::restore_checkpoint(pool, slot, &mut recurrent_bufs, stream_raw)?;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn read_hidden(&self) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut buf = vec![0.0f32; self.config.hidden_size];
        self.activations.hidden.copy_to_host(&mut buf)?;
        Ok(buf)
    }

    pub fn decode_step_traced(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<(Vec<f32>, Vec<(String, Vec<f32>)>), ModelError> {
        let hs = self.config.hidden_size as u32;
        let vs = self.config.vocab_size as u32;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();
        if self
            .config
            .layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::Attention)
        {
            self.append_paged_decode_token(position)?;
        }

        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for i in 0..self.config.num_layers {
            if self.config.layers[i].layer_type == LayerType::Attention {
                self.attention_forward(i, kv_idx, position)?;
                kv_idx += 1;
            } else {
                self.gdn_forward(i, gdn_idx)?;
                gdn_idx += 1;
            }
            traces.push((format!("layer_{i}"), self.read_hidden()?));
        }

        unsafe {
            d2d_copy_f32(
                &mut self.activations.normed,
                0,
                &self.activations.hidden,
                0,
                hs as usize,
                &self.stream,
            )?;
        }
        self.kernels.rmsnorm.forward(
            &mut self.activations.hidden,
            &self.activations.normed,
            &self.final_norm_weight,
            1,
            hs,
            self.config.rms_norm_eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;
        traces.push(("final_norm".into(), self.read_hidden()?));

        self.kernels.lm_head.forward(
            &mut self.activations.logits,
            &self.embed_weight,
            &self.activations.hidden,
            vs,
            hs,
            &self.stream,
        )?;
        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        self.seq_len = position + 1;
        Ok((logits, traces))
    }

    fn read_buf(&self, buf: &DeviceBuffer<f32>) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut v = vec![0.0f32; buf.len()];
        buf.copy_to_host(&mut v)?;
        Ok(v)
    }

    pub fn gdn_layer0_trace(
        &mut self,
        token_id: u32,
    ) -> Result<Vec<(String, Vec<f32>)>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let nh = self.config.linear_num_heads as u32;
        let kd = self.config.linear_key_head_dim as u32;
        let vd = self.config.linear_value_head_dim as u32;
        let ck = self.config.linear_conv_kernel_dim as u32;
        let eps = self.config.rms_norm_eps;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let weights = match &self.layers[0] {
            LayerWeights::Gdn(w) => w as *const GdnLayerWeights,
            _ => panic!("layer 0 not GDN"),
        };
        let w = unsafe { &*weights };

        // RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &w.input_norm,
            1,
            hs,
            eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;
        traces.push(("normed".into(), self.read_buf(&self.activations.normed)?));

        let nvh_traced = self.config.linear_num_value_heads as u32;
        let gqa_traced = nvh_traced / nh;

        // QKV projection
        w.w_qkv.forward(
            &self.kernels.linear_proj,
            &mut self.activations.qkv,
            &self.activations.normed,
            nh * kd * 2 + nvh_traced * vd,
            hs,
            &self.stream,
        )?;
        traces.push(("qkv_pre_conv".into(), self.read_buf(&self.activations.qkv)?));

        // a, b, z projections
        w.w_a.forward(
            &self.kernels.linear_proj,
            &mut self.activations.a_proj,
            &self.activations.normed,
            nvh_traced,
            hs,
            &self.stream,
        )?;
        w.w_b.forward(
            &self.kernels.linear_proj,
            &mut self.activations.b_proj,
            &self.activations.normed,
            nvh_traced,
            hs,
            &self.stream,
        )?;
        w.w_z.forward(
            &self.kernels.linear_proj,
            &mut self.activations.z_proj,
            &self.activations.normed,
            nvh_traced * vd,
            hs,
            &self.stream,
        )?;
        traces.push(("a_proj".into(), self.read_buf(&self.activations.a_proj)?));
        traces.push(("b_proj".into(), self.read_buf(&self.activations.b_proj)?));
        traces.push(("z_proj".into(), self.read_buf(&self.activations.z_proj)?));

        // Conv1d: split qkv, run 3 separate convs, reassemble
        let conv_q_len = (nh * kd) as usize;
        let conv_k_len = (nh * kd) as usize;
        let conv_v_len = (nvh_traced * vd) as usize;
        let ck_usize = ck as usize;

        unsafe {
            d2d_copy_f32(
                &mut self.activations.q_gdn,
                0,
                &self.activations.qkv,
                0,
                conv_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.k_gdn,
                0,
                &self.activations.qkv,
                conv_q_len,
                conv_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.v_gdn,
                0,
                &self.activations.qkv,
                conv_q_len + conv_k_len,
                conv_v_len,
                &self.stream,
            )?;
        }

        let mut conv_w_q = DeviceBuffer::<u16>::alloc(self.device, conv_q_len * ck_usize)?;
        let mut conv_w_k = DeviceBuffer::<u16>::alloc(self.device, conv_k_len * ck_usize)?;
        let mut conv_w_v = DeviceBuffer::<u16>::alloc(self.device, conv_v_len * ck_usize)?;
        unsafe {
            let conv1d = w.conv1d_weight.as_ref().expect("conv1d_weight required for trace (set TRACE env var)");
            d2d_copy_u16(
                &mut conv_w_q,
                0,
                conv1d,
                0,
                conv_q_len * ck_usize,
                &self.stream,
            )?;
            d2d_copy_u16(
                &mut conv_w_k,
                0,
                conv1d,
                conv_q_len * ck_usize,
                conv_k_len * ck_usize,
                &self.stream,
            )?;
            d2d_copy_u16(
                &mut conv_w_v,
                0,
                conv1d,
                (conv_q_len + conv_k_len) * ck_usize,
                conv_v_len * ck_usize,
                &self.stream,
            )?;
        }

        let conv_state_q_len = conv_q_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_len * (ck_usize - 1);

        let mut cs_q = DeviceBuffer::<f32>::alloc(self.device, conv_state_q_len)?;
        let mut cs_k = DeviceBuffer::<f32>::alloc(self.device, conv_state_k_len)?;
        let mut cs_v = DeviceBuffer::<f32>::alloc(self.device, conv_state_v_len)?;
        unsafe {
            d2d_copy_f32(
                &mut cs_q,
                0,
                &self.gdn_conv_states[0],
                0,
                conv_state_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut cs_k,
                0,
                &self.gdn_conv_states[0],
                conv_state_q_len,
                conv_state_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut cs_v,
                0,
                &self.gdn_conv_states[0],
                conv_state_q_len + conv_state_k_len,
                conv_state_v_len,
                &self.stream,
            )?;
        }

        let mut conv_out_q = DeviceBuffer::<f32>::alloc(self.device, conv_q_len)?;
        let mut conv_out_k = DeviceBuffer::<f32>::alloc(self.device, conv_k_len)?;
        let mut conv_out_v = DeviceBuffer::<f32>::alloc(self.device, conv_v_len)?;

        self.kernels.causal_conv1d.forward(
            &mut cs_q,
            &self.activations.q_gdn,
            &conv_w_q,
            &mut conv_out_q,
            conv_q_len as u32,
            ck,
            &self.stream,
        )?;
        self.kernels.causal_conv1d.forward(
            &mut cs_k,
            &self.activations.k_gdn,
            &conv_w_k,
            &mut conv_out_k,
            conv_k_len as u32,
            ck,
            &self.stream,
        )?;
        self.kernels.causal_conv1d.forward(
            &mut cs_v,
            &self.activations.v_gdn,
            &conv_w_v,
            &mut conv_out_v,
            conv_v_len as u32,
            ck,
            &self.stream,
        )?;

        traces.push(("conv_out_q".into(), self.read_buf(&conv_out_q)?));
        traces.push(("conv_out_k".into(), self.read_buf(&conv_out_k)?));
        traces.push(("conv_out_v".into(), self.read_buf(&conv_out_v)?));

        // Copy conv outputs to q/k/v
        unsafe {
            d2d_copy_f32(
                &mut self.activations.q_gdn,
                0,
                &conv_out_q,
                0,
                conv_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.k_gdn,
                0,
                &conv_out_k,
                0,
                conv_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.v_gdn,
                0,
                &conv_out_v,
                0,
                conv_v_len,
                &self.stream,
            )?;
        }

        // Gate
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn,
            &w.a_log,
            &self.activations.a_proj,
            &w.dt_bias,
            nh,
            &self.stream,
        )?;
        traces.push(("gate".into(), self.read_buf(&self.activations.gate_gdn)?));

        // Recurrent
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn,
            &self.activations.k_gdn,
            &self.activations.v_gdn,
            &self.activations.gate_gdn,
            &self.activations.b_proj,
            &mut self.gdn_states[0].recurrent,
            &mut self.activations.recurrent_out,
            nvh_traced,
            kd,
            vd,
            gqa_traced,
            &self.stream,
        )?;
        traces.push((
            "recurrent_out".into(),
            self.read_buf(&self.activations.recurrent_out)?,
        ));

        // RMSNormGated
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated,
            &self.activations.recurrent_out,
            &self.activations.z_proj,
            &w.output_norm,
            nvh_traced,
            vd,
            eps,
            &self.stream,
        )?;
        traces.push((
            "normed_gated".into(),
            self.read_buf(&self.activations.normed_gated)?,
        ));

        // out_proj
        w.w_out.forward(
            &self.kernels.linear_proj,
            &mut self.activations.out_proj,
            &self.activations.normed_gated,
            hs,
            nvh_traced * vd,
            &self.stream,
        )?;
        traces.push((
            "out_proj".into(),
            self.read_buf(&self.activations.out_proj)?,
        ));

        // Residual
        unsafe {
            d2d_copy_f32(
                &mut self.activations.residual,
                0,
                &self.activations.hidden,
                0,
                hs as usize,
                &self.stream,
            )?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs,
            &self.stream,
        )?;
        traces.push(("after_residual".into(), self.read_hidden()?));

        Ok(traces)
    }

    pub fn reset_state(&mut self) -> Result<(), ModelError> {
        let nh = self.config.linear_num_heads;
        let kd = self.config.linear_key_head_dim;
        let vd = self.config.linear_value_head_dim;
        let ck = self.config.linear_conv_kernel_dim;
        let nvh_r = self.config.linear_num_value_heads;
        let qkv_out = nh * kd * 2 + nvh_r * vd;

        for state in &mut self.gdn_states {
            let zeros = vec![0.0f32; nvh_r * kd * vd];
            state.recurrent.copy_from_host(&zeros)?;
        }
        for conv_state in &mut self.gdn_conv_states {
            let zeros = vec![0.0f32; qkv_out * (ck - 1)];
            conv_state.copy_from_host(&zeros)?;
        }
        if let Some(caches) = self.legacy_kv_caches.as_mut() {
            let kv_size = self.config.max_seq_len * self.config.num_kv_heads * self.config.head_dim;
            let zeros_kv = vec![0.0f32; kv_size];
            for cache in caches {
                cache.k.copy_from_host(&zeros_kv)?;
                cache.v.copy_from_host(&zeros_kv)?;
            }
        }
        self.seq_len = 0;
        if let Some(seq) = self.paged_seq.as_mut() {
            if let Some(q_alloc) = self.quant_allocator.as_mut() {
                seq.free_quant_slots(q_alloc);
            }
            if let Some(alloc) = self.page_allocator.as_mut() {
                seq.reset(alloc);
            }
        }
        Ok(())
    }
}
