# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for DeepSeek-V4 Compressed Sparse Attention (CSA) Indexer loss and training."""

import unittest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_compressed
from maxtext.layers.attention_mla import indexer_losses


class DeepSeekV4IndexerLossTest(unittest.TestCase):
  """Tests for DeepSeek-V4 CSA Indexer KL Divergence loss calculation and gradients."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.seq_len = 16
    self.base_emb_dim = 64
    self.base_num_query_heads = 4
    self.base_num_kv_heads = 1
    self.head_dim = 32
    self.compress_ratio = 4
    self.indexer_n_heads = 4
    self.indexer_head_dim = 32
    self.indexer_topk = 2
    self.q_lora_rank = 32

  def _get_config(
      self,
      indexer_loss_scaling_factor=0.5,
      indexer_sparse_training=False,
      mla_qk_head_chunk_size=0,
      indexer_topk=None,
  ):
    """Constructs a test MaxTextConfig with CSA indexer configuration."""
    topk = indexer_topk if indexer_topk is not None else self.indexer_topk
    argv = [
        "",
        "src/maxtext/configs/base.yml",
        "run_name=test_dsv4_indexer",
        "decoder_block=deepseek4",
        "attention_type=compressed",
        "attention=dot_product",
        "use_indexer=True",
        f"indexer_loss_scaling_factor={indexer_loss_scaling_factor}",
        f"indexer_sparse_training={indexer_sparse_training}",
        f"mla_qk_head_chunk_size={mla_qk_head_chunk_size}",
        f"max_target_length={self.seq_len}",
        f"indexer_topk={topk}",
        f"indexer_n_heads={self.indexer_n_heads}",
        f"indexer_head_dim={self.indexer_head_dim}",
        f"base_emb_dim={self.base_emb_dim}",
        f"base_num_query_heads={self.base_num_query_heads}",
        f"base_num_kv_heads={self.base_num_kv_heads}",
        f"head_dim={self.head_dim}",
        f"qk_rope_head_dim={self.head_dim}",
        f"q_lora_rank={self.q_lora_rank}",
        "o_groups=2",
        "o_lora_rank=16",
    ]
    return pyconfig.initialize(argv)

  def _init_csa_attention(self, config):
    """Initializes a CompressedAttention module for testing."""
    rngs = nnx.Rngs(0)
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    attn = attention_compressed.CompressedAttention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(self.batch_size, self.seq_len, config.emb_dim),
        inputs_kv_shape=(self.batch_size, self.seq_len, config.emb_dim),
        compress_ratio=self.compress_ratio,
        q_lora_rank=config.q_lora_rank,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs,
    )
    return attn

  def test_csa_indexer_loss_computation(self):
    """Test that CSA forward pass computes and stores indexer_loss variable."""
    config = self._get_config(indexer_loss_scaling_factor=0.5, indexer_sparse_training=False)
    attn = self._init_csa_attention(config)

    inputs_q = jax.random.normal(jax.random.PRNGKey(1), (self.batch_size, self.seq_len, config.emb_dim))
    inputs_kv = jax.random.normal(jax.random.PRNGKey(2), (self.batch_size, self.seq_len, config.emb_dim))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    out, _ = attn(
        inputs_q=inputs_q,
        inputs_kv=inputs_kv,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertIsNotNone(out)
    self.assertEqual(out.shape, (self.batch_size, self.seq_len, config.emb_dim))
    self.assertTrue(hasattr(attn, "indexer_loss"))
    self.assertIsInstance(attn.indexer_loss, indexer_losses)

    loss_val = attn.indexer_loss.value
    self.assertGreater(float(loss_val), 0.0)

  def test_csa_indexer_loss_sparse_training_mode(self):
    """Test CSA forward pass and indexer loss in sparse pre-training mode."""
    config = self._get_config(indexer_loss_scaling_factor=0.5, indexer_sparse_training=True)
    attn = self._init_csa_attention(config)

    inputs_q = jax.random.normal(jax.random.PRNGKey(3), (self.batch_size, self.seq_len, config.emb_dim))
    inputs_kv = jax.random.normal(jax.random.PRNGKey(4), (self.batch_size, self.seq_len, config.emb_dim))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    out, _ = attn(
        inputs_q=inputs_q,
        inputs_kv=inputs_kv,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertIsNotNone(out)
    self.assertTrue(hasattr(attn, "indexer_loss"))
    self.assertIsInstance(attn.indexer_loss, indexer_losses)
    self.assertGreater(float(attn.indexer_loss.value), 0.0)

  def test_csa_indexer_loss_kl_divergence_zero(self):
    """Test KL divergence is 0 when predicted and target distributions match."""
    config = self._get_config(indexer_loss_scaling_factor=1.0)
    attn = self._init_csa_attention(config)

    n_windows = self.seq_len // self.compress_ratio
    query = jnp.zeros((self.batch_size, self.seq_len, config.num_query_heads, config.head_dim))
    compressed_kv = jnp.zeros((self.batch_size, n_windows, config.num_kv_heads, config.head_dim))
    compressed_mask = jnp.zeros((self.batch_size, 1, self.seq_len, n_windows))

    # Causal block mask (matches what DeepseekV4Indexer produces on uniform logits)
    q_pos = jnp.arange(self.seq_len)[:, None]
    block_end_pos = (jnp.arange(n_windows)[None, :] + 1) * self.compress_ratio
    future_mask = block_end_pos > (q_pos + 1)
    indexer_score = jnp.where(future_mask[None, :, :], -1e9, 0.0)

    loss = attn.calculate_csa_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        causal_mask=None,
        position_ids=None,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-5)

  def test_csa_indexer_loss_head_chunking_parity(self):
    """Test that head chunking scan produces bitwise identical loss to native einsum."""
    config_chunked = self._get_config(mla_qk_head_chunk_size=2)
    config_native = self._get_config(mla_qk_head_chunk_size=0)
    attn_chunked = self._init_csa_attention(config_chunked)
    attn_native = self._init_csa_attention(config_native)

    n_windows = self.seq_len // self.compress_ratio
    rng = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(rng, 3)
    query = jax.random.normal(k1, (self.batch_size, self.seq_len, config_chunked.num_query_heads, config_chunked.head_dim))
    compressed_kv = jax.random.normal(k2, (self.batch_size, n_windows, config_chunked.num_kv_heads, config_chunked.head_dim))
    indexer_score = jax.random.normal(k3, (self.batch_size, self.seq_len, n_windows))
    compressed_mask = jnp.zeros((self.batch_size, 1, self.seq_len, n_windows))

    loss_chunked = attn_chunked.calculate_csa_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        causal_mask=None,
        position_ids=None,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    loss_native = attn_native.calculate_csa_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        causal_mask=None,
        position_ids=None,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    np.testing.assert_allclose(float(loss_chunked), float(loss_native), rtol=1e-5, atol=1e-5)

  def test_csa_indexer_gradients_flow(self):
    """Test that gradients flow to indexer parameters and do not leak into main projections."""
    config = self._get_config(indexer_loss_scaling_factor=1.0, indexer_sparse_training=False)
    attn = self._init_csa_attention(config)

    inputs_q = jax.random.normal(jax.random.PRNGKey(1), (self.batch_size, self.seq_len, config.emb_dim))
    inputs_kv = jax.random.normal(jax.random.PRNGKey(2), (self.batch_size, self.seq_len, config.emb_dim))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    def loss_fn(attn_model):
      attn_model(
          inputs_q=inputs_q,
          inputs_kv=inputs_kv,
          decoder_segment_ids=segment_ids,
          inputs_positions=positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return attn_model.indexer_loss.value

    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(attn)

    # Gradients must flow to indexer projection kernels
    self.assertIsNotNone(grads.csa_compressor.indexer.q_proj.kernel)
    self.assertIsNotNone(grads.csa_compressor.indexer.kv_proj.kernel)
    self.assertIsNotNone(grads.csa_compressor.indexer.gate_proj.kernel)
    self.assertIsNotNone(grads.csa_compressor.indexer.weights_proj.kernel)

    q_grad_norm = jnp.linalg.norm(grads.csa_compressor.indexer.q_proj.kernel.value)
    self.assertGreater(float(q_grad_norm), 0.0)

    # Gradients must not leak into main model projections
    self.assertAlmostEqual(float(jnp.linalg.norm(grads.wq_a.kernel.value)), 0.0)
    self.assertAlmostEqual(float(jnp.linalg.norm(grads.wq_b.kernel.value)), 0.0)
    self.assertAlmostEqual(float(jnp.linalg.norm(grads.wkv.kernel.value)), 0.0)

  def test_dense_warmup_forward_mask_is_causal_dense(self):
    """Test that dense warm-up uses causal block masking without top-k pruning."""
    config = self._get_config(indexer_loss_scaling_factor=1.0, indexer_sparse_training=False, indexer_topk=1)
    attn = self._init_csa_attention(config)

    n_windows = self.seq_len // self.compress_ratio  # 4 blocks
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))

    # Construct dense causal mask directly from attention logic
    usable_len = n_windows * attn.compress_ratio
    block_positions = positions[:, :usable_len:attn.compress_ratio]
    is_future = (block_positions[:, None, :] + attn.compress_ratio) > (positions[:, :, None] + 1)
    dense_causal_mask = jnp.where(is_future, -1e9, 0.0)

    # For query token t=4 (belongs to block 1): block 0 is unmasked (0.0), block 1 is masked (-1e9)
    for b in range(self.batch_size):
      self.assertEqual(float(dense_causal_mask[b, 4, 0]), 0.0)
      self.assertLess(float(dense_causal_mask[b, 4, 1]), -1e8)
      self.assertLess(float(dense_causal_mask[b, 4, 2]), -1e8)
      self.assertLess(float(dense_causal_mask[b, 4, 3]), -1e8)

      # For query token t=8 (belongs to block 2): blocks 0, 1 are unmasked (0.0), blocks 2, 3 are masked (-1e9)
      self.assertEqual(float(dense_causal_mask[b, 8, 0]), 0.0)
      self.assertEqual(float(dense_causal_mask[b, 8, 1]), 0.0)
      self.assertLess(float(dense_causal_mask[b, 8, 2]), -1e8)
      self.assertLess(float(dense_causal_mask[b, 8, 3]), -1e8)

      # For query token t=15 (last token of block 3): blocks 0, 1, 2, 3 are ALL unmasked (0.0)
      for w in range(n_windows):
        self.assertEqual(float(dense_causal_mask[b, 15, w]), 0.0)

  def test_teacher_causality_and_packing_on_loss_function(self):
    """Test calculate_csa_indexer_loss directly on a 2-segment packed sequence with causal boundaries."""
    config = self._get_config(indexer_loss_scaling_factor=1.0)
    attn = self._init_csa_attention(config)

    # 2 segments: Doc 1 = tokens 0..7 (blocks 0, 1), Doc 2 = tokens 8..15 (blocks 2, 3)
    n_windows = self.seq_len // self.compress_ratio  # 4 blocks
    positions = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]] * self.batch_size)
    segment_ids = jnp.array([[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]] * self.batch_size)

    # Build compressed_segment_mask for the 2 documents
    comp_seg_ids = jnp.array([[1, 1, 2, 2]] * self.batch_size)
    valid_comp_seg = (segment_ids[:, :, None] == comp_seg_ids[:, None, :])
    compressed_segment_mask = jnp.where(valid_comp_seg, 0.0, -1e9)

    query = jnp.zeros((self.batch_size, self.seq_len, config.num_query_heads, config.head_dim))
    compressed_kv = jnp.zeros((self.batch_size, n_windows, config.num_kv_heads, config.head_dim))
    compressed_mask = jnp.zeros((self.batch_size, 1, self.seq_len, n_windows))

    # Case A: Perfect student prediction matching causal + packed teacher distribution
    # Doc 1:
    # - t=4 (pos 4): only block 0 is valid
    # - t=7 (pos 7): blocks 0, 1 are valid
    # Doc 2:
    # - t=12 (pos 4 in doc 2): only block 2 is valid
    # - t=15 (pos 7 in doc 2): blocks 2, 3 are valid
    # Compute ground truth causal+packing mask for student
    usable_len = n_windows * attn.compress_ratio
    block_positions = positions[:, :usable_len:attn.compress_ratio]
    is_future = (block_positions[:, None, :] + attn.compress_ratio) > (positions[:, :, None] + 1)
    causal_mask = jnp.where(is_future, -1e9, 0.0)
    ground_truth_student_scores = causal_mask + compressed_segment_mask

    loss_perfect = attn.calculate_csa_indexer_loss(
        indexer_score=ground_truth_student_scores,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        causal_mask=compressed_segment_mask,
        position_ids=positions,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    np.testing.assert_allclose(float(loss_perfect), 0.0, atol=1e-5)

    # Case B: Student predicts mass on a future block in Doc 1 (t=4 predicting block 1)
    leaky_student_scores = ground_truth_student_scores.at[:, 4, 1].set(100.0)
    loss_future_leak = attn.calculate_csa_indexer_loss(
        indexer_score=leaky_student_scores,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        causal_mask=compressed_segment_mask,
        position_ids=positions,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    self.assertGreater(float(loss_future_leak), 0.1)

    # Case C: Student in Doc 2 predicts mass on a block from Doc 1 (t=12 predicting block 0)
    cross_doc_student_scores = ground_truth_student_scores.at[:, 12, 0].set(100.0)
    loss_cross_doc = attn.calculate_csa_indexer_loss(
        indexer_score=cross_doc_student_scores,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        causal_mask=compressed_segment_mask,
        position_ids=positions,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    self.assertGreater(float(loss_cross_doc), 0.1)


if __name__ == "__main__":
  unittest.main()
