
## Computation


## Memory consumption

### Decoder block in summary

```text
1. Q, K, V projections: batch × seq_len × d_m each
   - For multi-head attention, these get reshaped to: batch × seq_len × num_heads × d_k
   - Where d_m = num_heads × d_k

2. Attention scores: Q @ K^T 
   → batch × num_heads × seq_len × seq_len
   (Note: num_heads and seq_len order corrected)
   
3. Attention weights: softmax(scores)
   → batch × num_heads × seq_len × seq_len
   
4. Attention output: weights @ V
   → batch × num_heads × seq_len × d_k
   - Each head's output has d_k dimension
   
5. Concatenate heads: 
   → batch × seq_len × d_m (after concatenation)

6. Output projection: concat @ W_o
   → batch × seq_len × d_m (after output projection)
      
7. Residual, LayerNorm:
   → batch × seq_len × d_m

8. FFN-1:
   → batch × seq_len × d_h

9. FFN-2:
   → batch × seq_len × d_m

10. Residual, LayerNorm:
   → batch × seq_len × d_m
```

### Pseudocode
```python
def transformer_decoder_block(input_tensor, W_q, W_k, W_v, W_o, W_1, W_2):
   # QKV Projections
   Q = input_tensor @ W_q  # (batch, seq_len, d_m)
   K = input_tensor @ W_k  # (batch, seq_len, d_m) 
   V = input_tensor @ W_v  # (batch, seq_len, d_m)
   
   # Reshape for multi-head: (batch, seq_len, num_heads, d_k)
   Q = Q.reshape(batch, seq_len, num_heads, d_k)
   K = K.reshape(batch, seq_len, num_heads, d_k) 
   V = V.reshape(batch, seq_len, num_heads, d_k)
   # Attention computation per head
   for each head_i in all_heads:
     attention_scores = (Q_i @ K_i.transpose) / sqrt(d_k) # Q_i freed after here. K_i,V_i can either be freed or stay in memory after here
     attention_weights = softmax(attention_scores)  # attention_scores freed after here # *** PEAK 1: input_tensor + attention_scores + attention_weights ***
     attention_output = attention_weights @ V_i # attention_weights freed after here
                                         
   concat_attention_output = concat(all_attention_outputs)  # all_attention_outputs freed after here
   W_o_output = concat_attention_output @ W_o # concat_attention_output freed after here
   residual_sum = input_tensor + W_o_output  # W_o_output freed after here
   ffn_input = LayerNorm(residual_sum)  # residual_sum freed after here
   ffn_intermediate = activation(ffn_input @ W_1)  # ffn_input STAYS ALIVE for residual connection
   ffn_output = ffn_intermediate @ W_2  # ffn_intermediate freed after here # *** PEAK 2: input_tensor + ffn_input + ffn_intermediate + ffn_output ***
   final_residual = ffn_input + ffn_output  # ffn_output freed after here
   final_output = LayerNorm(final_residual)  # ffn_input and final_residual freed after here
   return final_output
```

Memory peaks occur at:
- PEAK 1: input_tensor + attention_scores + attention_weights
- PEAK 2: input_tensor + ffn_input + ffn_intermediate + ffn_output
  -  K, V will be alive until the end of the inference if KV are cached

**timeline**
```
time flows to the right ->
| ----------------------------------------------------------------------------------------------- |
| ------------------------------------- decoder block ------------------------------------------- |
| --------------- | ----------------------------------------- | --------------------------------- |
| QKV projection  |                 Attention                 |              FFN                  |
| --------------- | ----------------------------------------- | --------------------------------- |
| ----------------------------------------------------------------------------------------------- |

Input tensor=======================================================================================
--Q,K,V===========
     --attn_scores================
                  --attn_weights============
                     ^          --attn_output================
                  (peak 1)                   --concat_outputs=========
                                                             --W_o_out=======
                                                                      --ffn_inp====================
                                                                               --ffn_inter=========
                                                                                          --ffn_out
                                                                                               ^
                                                                                           (peak 2)
```

#### Exact calculation

**Variables:**

- seq_len: sequence_length
- batch: batch_size
- d_m: hidden_dim
- d_h: intermediate_dim (typically 4×d_m)
- num_heads: number_of_heads
- d_k: dimension per head (d_m = num_heads × d_k)

**Model weights (per layer):**
- `model_weights_per_layer = 4 × d_m² + 2 × d_m × d_h`
  - 4 × d_m²: Four attention matrices (W_q, W_k, W_v, W_o)
  - 2 × d_m × d_h: Two FFN matrices (W_1, W_2)

**Attention computation (the bottleneck):**
- `attention_peak = batch × num_heads × seq_len² × 2`
  - Factor of 2: both attention_scores and attention_weights briefly coexist
  - Dimensions: batch × num_heads × seq_len × seq_len for each matrix
  - lifespan: during attention layer computation

**Layer input/output activations:**
- `layer_activations = batch × seq_len × d_m × 2`
  - Factor of 2: input and output activations (for residual connections)
  - lifespan: during the entire inference

**FFN intermediate:**
- `ffn_intermediate = batch × seq_len × d_h`
  - Peak during FFN computation (W_1 output before W_2)
  - lifespan: during FFN layer computation

**KV cache (per layer):**
- `kv_cache_per_layer = batch × 2 × seq_len × d_m`
  - Factor of 2: separate K and V caches for this layer
  - lifespan: the entire inference

**Total memory per layer:**
`total_memory_per_layer = model_weights_per_layer + max(attention_peak, ffn_intermediate) +           layer_activations + kv_cache_per_layer`

Note: The attention_peak term (∝ seq_len²) typically dominates for long sequences, making it the primary memory bottleneck during inference.