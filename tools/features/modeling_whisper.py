def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        quantized_token_ids=None
):
    r"""
    Args:
        input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
            `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        attention_mask (`torch.Tensor`)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
            but it is not used. By default the silence in the input log mel spectrogram are ignored.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """

    # expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
    # if input_features.shape[-1] != expected_seq_length:
    #     raise ValueError(
    #         f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
    #     )

    batch_size, feature_size, seq_length = input_features.shape
    seq_length = seq_length // (self.conv1.stride[0] * self.conv2.stride[0])

    attention_mask = attention_mask[:, :: self.conv1.stride[0] * self.conv2.stride[0]]
    if self.config.quantize_causal_block_size is not None:
        extended_attention_mask = self.get_block_causal_attention_mask(attention_mask,
                                                                        block_size=self.config.quantize_causal_block_size)
    else:
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, (batch_size, seq_length))
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    embed_pos = self.embed_positions.weight

    hidden_states = inputs_embeds + embed_pos[:seq_length]
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    assert attention_mask.shape[-1] == hidden_states.shape[1]
    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        if to_drop:
            layer_outputs = (None, None)
        else:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    extended_attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)
        if idx + 1 == self.config.pooling_position and self.config.pooling_kernel_size is not None:
            hidden_states = hidden_states.permute(0, 2, 1)
            if hidden_states.shape[-1] % self.config.pooling_kernel_size != 0:
                hidden_states = torch.nn.functional.pad(hidden_states, (
                0, self.config.pooling_kernel_size - hidden_states.shape[-1] % self.config.pooling_kernel_size))

            whisper_hidden_states_50hz = hidden_states.clone()

            hidden_states = self.pooling_layer(hidden_states).permute(0, 2, 1)
            attention_mask = attention_mask[:, ::self.config.pooling_kernel_size]
            if self.config.quantize_causal_block_size is not None:
                extended_attention_mask = self.get_block_causal_attention_mask(attention_mask, block_size=self.config.quantize_causal_block_size // self.config.pooling_kernel_size)
            else:
                extended_attention_mask = self.get_extended_attention_mask(attention_mask, (
                batch_size, seq_length // self.config.pooling_kernel_size))

        if idx + 1 == self.config.quantize_position and self.config.quantize_vocab_size is not None:
            if quantized_token_ids is not None:
                hidden_states = self.codebook(quantized_token_ids)
            else:
                hidden_quantized, indices_flat, distances = vector_quantize(hidden_states, self.codebook.weight)
                quantized_token_ids = indices_flat.reshape(batch_size, hidden_quantized.shape[1])
                if self.training:
                    encodings = torch.nn.functional.one_hot(indices_flat, self.config.quantize_vocab_size).float()
                    encodings = encodings * attention_mask.reshape(-1, 1)
                    n = torch.sum(encodings, dim=0)
                    torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
                    self.num_active_codes = n.nonzero().shape[0]
                    if self.config.quantize_ema_decay:
                        hidden_flat = hidden_states.detach().float().reshape(-1, hidden_states.shape[-1])
                        with torch.autocast(device_type='cuda', dtype=torch.float32):
                            dw = torch.matmul(encodings.t(), hidden_flat)
                        torch.distributed.all_reduce(dw, op=torch.distributed.ReduceOp.SUM)
                        self.ema_count = self.ema_count * self.config.quantize_ema_decay + (
                                1 - self.config.quantize_ema_decay) * n
                        total_count = torch.sum(self.ema_count)
                        self.ema_count = (self.ema_count + 1e-5) / (
                                total_count + self.config.quantize_vocab_size * 1e-5) * total_count
                        self.ema_weight = self.ema_weight * self.config.quantize_ema_decay + (
                                1 - self.config.quantize_ema_decay) * dw
                        self.codebook.weight.data = self.ema_weight / self.ema_count.unsqueeze(1)
                        self.quantize_loss = self.config.quantize_loss_scale * self.config.quantize_commit_coefficient * mse_loss_with_mask(
                            hidden_states, hidden_quantized.detach(), attention_mask)
                        self.quantize_ema_count += 1
                        if self.config.quantize_restart_interval is not None and self.quantize_ema_count % self.config.quantize_restart_interval == 0:
                            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
                            segment_vocab_size = self.config.quantize_vocab_size // world_size
                            start_idx = segment_vocab_size * rank
                            ema_count_segment = self.ema_count[start_idx: start_idx + segment_vocab_size]
                            threshold = 1 * (
                                        self.config.quantize_ema_decay ** self.config.quantize_restart_interval)
                            update_indices = (ema_count_segment < threshold).nonzero()[:, 0] + start_idx
                            num_update = update_indices.shape[0]
                            mask_flat = attention_mask.reshape(-1) > 0
                            hidden_selected = hidden_flat[mask_flat]
                            hidden_update = hidden_selected[random.sample(range(len(hidden_selected)), num_update)]
                            num_update = torch.as_tensor([num_update], dtype=torch.long,
                                                            device=hidden_states.device)
                            num_update_list = [torch.as_tensor([0], dtype=torch.long, device=hidden_states.device)
                                                for _
                                                in range(world_size)]
                            torch.distributed.all_gather(num_update_list, num_update)
                            update_indices_list = [
                                torch.zeros(num.item(), dtype=torch.long, device=hidden_states.device) for num in
                                num_update_list]
                            torch.distributed.all_gather(update_indices_list, update_indices)
                            update_indices = torch.cat(update_indices_list)
                            hidden_update_list = [
                                torch.zeros(num.item(), hidden_flat.shape[-1], dtype=hidden_update.dtype,
                                            device=hidden_states.device) for num in num_update_list]
                            torch.distributed.all_gather(hidden_update_list, hidden_update)
                            hidden_update = torch.cat(hidden_update_list)
                            self.codebook.weight.data[update_indices] = hidden_update
                            self.ema_count[update_indices] = 1
                            self.ema_weight[update_indices] = hidden_update
                            if torch.distributed.get_rank() == 0:
                                print(f"restart {len(update_indices)} tokens")
                    else:
                        loss = self.config.quantize_loss_scale * (
                                self.config.quantize_commit_coefficient * mse_loss_with_mask(hidden_states,
                                                                                                hidden_quantized.detach(),
                                                                                                attention_mask) + mse_loss_with_mask(
                            hidden_quantized, hidden_states.detach(), attention_mask))
                        self.quantize_loss = loss
                    hidden_states = hidden_states + (hidden_quantized - hidden_states).detach()
                else:
                    hidden_states = hidden_quantized
            hidden_states = hidden_states + self.embed_positions2.weight[:hidden_states.shape[1]]

        if idx + 1 == self.save_hidden_position:
            import numpy as np
            import uuid
            to_save = []
            for batch_idx, hidden_state in enumerate(hidden_states):
                for seq_idx, hidden in enumerate(hidden_state):
                    if attention_mask[batch_idx, seq_idx]:
                        to_save.append(hidden.detach().cpu().numpy())
            np.save(os.path.join(self.save_hidden_dir, f"{str(uuid.uuid4())}.npy"), to_save)
    if not self.config.quantize_encoder_only:
        hidden_states = self.layer_norm(hidden_states)
    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    return {
        "last_hidden_state": hidden_states,
        "hidden_states": encoder_states,
        "attentions": all_attentions,
        "quantized_token_ids": quantized_token_ids,
        "whisper_hidden_states_50hz": whisper_hidden_states_50hz if self.config.pooling_kernel_size is not None else None,
    }