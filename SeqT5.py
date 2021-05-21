import copy
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Gumbel
from torch.nn import CrossEntropyLoss
from torch.utils import checkpoint
from transformers import T5ForConditionalGeneration, T5PreTrainedModel, top_k_top_p_filtering

from transformers.activations import ACT2FN
from transformers.file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.t5.modeling_t5 import T5Stack, T5_INPUTS_DOCSTRING
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config


logger = logging.get_logger(__name__)

s__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@dataclass
class AdversarialSeq2SeqLMOutput(Seq2SeqLMOutput):
    input_onehot: Optional[Tuple[torch.FloatTensor]] = None
    output_onehot: Optional[Tuple[torch.FloatTensor]] = None
    target_onehot: Optional[Tuple[torch.FloatTensor]] = None
    output_tokens: Optional[Tuple[torch.LongTensor]] = None
    modified_logits: Optional[Tuple[torch.FloatTensor]] = None


class T5SequentialDecoder(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super(T5SequentialDecoder, self).__init__(config, embed_tokens=embed_tokens)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        encoder_head_mask = self.get_head_mask(encoder_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        # hidden_states = self.dropout(inputs_embeds)  # TODO need this?
        hidden_states = inputs_embeds

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            encoder_layer_head_mask = encoder_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if encoder_layer_head_mask is not None:
                    encoder_layer_head_mask = encoder_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # def create_closure(use_cache, output_attentions, extended_attention_mask, encoder_extended_attention_mask):
            #     def compute_layer(*inputs):
            #         hidden_states, position_bias, encoder_hidden_states, \
            #         encoder_decoder_position_bias, layer_head_mask, encoder_layer_head_mask, past_key_value = inputs
            #         layer_outputs = layer_module(
            #             hidden_states,
            #             attention_mask=extended_attention_mask,
            #             position_bias=position_bias,
            #             encoder_hidden_states=encoder_hidden_states,
            #             encoder_attention_mask=encoder_extended_attention_mask,
            #             encoder_decoder_position_bias=encoder_decoder_position_bias,
            #             layer_head_mask=layer_head_mask,
            #             encoder_layer_head_mask=encoder_layer_head_mask,
            #             past_key_value=past_key_value,
            #             use_cache=use_cache,
            #             output_attentions=output_attentions,
            #         )
            #         lo = (layer_outputs[0],) + layer_outputs[1] +  (layer_outputs[2],) + (layer_outputs[3],)
            #         return lo
            #     return compute_layer
            # lo = checkpoint.checkpoint(create_closure(use_cache, output_attentions, extended_attention_mask, encoder_extended_attention_mask), hidden_states, position_bias, encoder_hidden_states,
            #         encoder_decoder_position_bias, layer_head_mask, encoder_layer_head_mask, past_key_value)
            # layer_outputs = (lo[0], lo[1:-2], lo[-2], lo[-1])

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                encoder_layer_head_mask=encoder_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        # hidden_states = self.dropout(hidden_states)  # TODO keep it or not?

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,  # TODO should be sliced??
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    # def forward(
    #         self,
    #         input_ids=None,
    #         attention_mask=None,
    #         encoder_hidden_states=None,
    #         encoder_attention_mask=None,
    #         inputs_embeds=None,
    #         head_mask=None,
    #         encoder_head_mask=None,
    #         past_key_values=None,
    #         use_cache=None,
    #         output_attentions=None,
    #         output_hidden_states=None,
    #         return_dict=None,
    #         logits_func=None
    # ):
    #     if isinstance(input_ids, tuple):
    #         input_ids = torch.zeros(input_ids, dtype=torch.long)
    #         one_hot_softmax = torch.zeros((input_ids.shape[0], self.embed_tokens.num_embeddings))
    #         one_hot_softmax[:, 0] = 1.
    #     else:
    #         one_hot_softmax = None
    #
    #     # Model parallel
    #     if self.model_parallel:
    #         torch.cuda.set_device(self.first_device)
    #         self.embed_tokens = self.embed_tokens.to(self.first_device)
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     if input_ids is not None and inputs_embeds is not None:
    #         err_msg_prefix = "decoder_" if self.is_decoder else ""
    #         raise ValueError(
    #             f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
    #         )
    #     elif input_ids is not None:
    #         input_shape = input_ids.size()
    #         input_ids = input_ids.view(-1, input_shape[-1])
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #     else:
    #         err_msg_prefix = "decoder_" if self.is_decoder else ""
    #         raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")
    #
    #     # if inputs_embeds is None:
    #     #     assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
    #     #     inputs_embeds = self.embed_tokens(input_ids)
    #
    #     batch_size, seq_length = input_shape
    #
    #     # required mask seq length can be calculated via length of past
    #     mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
    #
    #     if use_cache is True:
    #         assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
    #             self
    #         )
    #
    #     device = input_ids.device
    #
    #     if attention_mask is None:
    #         attention_mask = torch.ones(batch_size, mask_seq_length).to(device)
    #     if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
    #         encoder_seq_length = encoder_hidden_states.shape[1]
    #         encoder_attention_mask = torch.ones(
    #             batch_size, encoder_seq_length, device=device, dtype=torch.long
    #         )
    #
    #     # initialize past_key_values with `None` if past does not exist
    #     if past_key_values is None:
    #         past_key_values = [None] * len(self.block)
    #
    #     # ourselves in which case we just need to make it broadcastable to all heads.
    #     extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
    #
    #     if self.is_decoder and encoder_attention_mask is not None:
    #         encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    #     else:
    #         encoder_extended_attention_mask = None
    #
    #     # Prepare head mask if needed
    #     head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    #     encoder_head_mask = self.get_head_mask(encoder_head_mask, self.config.num_layers)
    #     present_key_value_states = () if use_cache else None
    #     all_hidden_states = () if output_hidden_states else None
    #     all_attentions = () if output_attentions else None
    #     all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    #     position_bias = None
    #     encoder_decoder_position_bias = None
    #
    #     # hidden_states = self.dropout(inputs_embeds)  # TODO need this?
    #     inputs_embeds = None
    #
    #     for tok_ind in range(input_ids.shape[1]):
    #
    #         # if inputs_embeds is None:
    #         #     assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
    #         #     inputs_embeds = self.embed_tokens(input_ids)
    #         if one_hot_softmax is None:
    #             inputs_embeds = self.embed_tokens(input_ids)
    #         else:
    #             if tok_ind == 0:
    #                 inputs_embeds = (one_hot_softmax @ self.embed_tokens.weight).unsqueeze(1)
    #             else:
    #                 inputs_embeds = torch.cat([inputs_embeds, (one_hot_softmax @ self.embed_tokens.weight).unsqueeze(1)], dim=1)
    #
    #         hidden_states = inputs_embeds
    #
    #         for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
    #             layer_head_mask = head_mask[i]
    #             encoder_layer_head_mask = encoder_head_mask[i]
    #             # Model parallel
    #             if self.model_parallel:
    #                 torch.cuda.set_device(hidden_states.device)
    #                 # Ensure that attention_mask is always on the same device as hidden_states
    #                 if attention_mask is not None:
    #                     attention_mask = attention_mask.to(hidden_states.device)
    #                 if position_bias is not None:
    #                     position_bias = position_bias.to(hidden_states.device)
    #                 if encoder_hidden_states is not None:
    #                     encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
    #                 if encoder_extended_attention_mask is not None:
    #                     encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
    #                 if encoder_decoder_position_bias is not None:
    #                     encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
    #                 if layer_head_mask is not None:
    #                     layer_head_mask = layer_head_mask.to(hidden_states.device)
    #                 if encoder_layer_head_mask is not None:
    #                     encoder_layer_head_mask = encoder_layer_head_mask.to(hidden_states.device)
    #             if output_hidden_states:
    #                 all_hidden_states = all_hidden_states + (hidden_states,)
    #
    #             layer_outputs = layer_module(
    #                 hidden_states,
    #                 attention_mask=extended_attention_mask[:,:,:tok_ind+1,:tok_ind+1],
    #                 position_bias=position_bias,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 encoder_attention_mask=encoder_extended_attention_mask,
    #                 encoder_decoder_position_bias=encoder_decoder_position_bias,
    #                 layer_head_mask=layer_head_mask,
    #                 encoder_layer_head_mask=encoder_layer_head_mask,
    #                 past_key_value=past_key_value,
    #                 use_cache=use_cache,
    #                 output_attentions=output_attentions,
    #             )
    #             # layer_outputs is a tuple with:
    #             # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
    #             hidden_states, present_key_value_state = layer_outputs[:2]
    #
    #             # We share the position biases between the layers - the first layer store them
    #             # layer_outputs = hidden-states, key-value-states (self-attention weights),
    #             # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
    #             position_bias = layer_outputs[2]
    #             if self.is_decoder and encoder_hidden_states is not None:
    #                 encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
    #             # append next layer key value states
    #             if use_cache:
    #                 present_key_value_states = present_key_value_states + (present_key_value_state,)
    #
    #             if output_attentions:
    #                 all_attentions = all_attentions + (layer_outputs[3],)
    #                 if self.is_decoder:
    #                     all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
    #
    #             # Model Parallel: If it's the last layer for that device, put things on the next device
    #             if self.model_parallel:
    #                 for k, v in self.device_map.items():
    #                     if i == v[-1] and "cuda:" + str(k) != self.last_device:
    #                         hidden_states = hidden_states.to("cuda:" + str(k + 1))
    #
    #         hidden_states = self.final_layer_norm(hidden_states)
    #         hidden_states = self.dropout(hidden_states)  # TODO keep it or not?
    #
    #         # Add last layer
    #         if output_hidden_states:
    #             all_hidden_states = all_hidden_states + (hidden_states,)
    #
    #         if not return_dict:
    #             return tuple(
    #                 v
    #                 for v in [
    #                     hidden_states,
    #                     present_key_value_states,
    #                     all_hidden_states,
    #                     all_attentions,
    #                     all_cross_attentions,
    #                 ]
    #                 if v is not None
    #             )
    #
    #         decoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
    #             last_hidden_state=hidden_states,
    #             past_key_values=present_key_value_states,  # TODO should be sliced??
    #             hidden_states=all_hidden_states,
    #             attentions=all_attentions,
    #             cross_attentions=all_cross_attentions,
    #         )
    #
    #         logits = logits_func(decoder_outputs)
    #
    #         one_hot_softmax = nn.functional.gumbel_softmax(logits[:, -1, :], hard=True)
    #
    #     return decoder_outputs



class SeqT5(T5ForConditionalGeneration):
    # https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L1362
    def __init__(self, config):
        T5PreTrainedModel.__init__(self, config)

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5SequentialDecoder(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def compute_logits(self, decoder_output):
        sequence_output = decoder_output[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        return lm_logits

    def create_gumbel_distribution(self):
        gumbel_loc = 0.
        gumbel_scale = 1.
        logger.info(f"Creating Gumbel distribution with parameters loc={gumbel_loc}, scale={gumbel_scale}")
        self.gumbel_dist = Gumbel(torch.tensor([gumbel_loc]), torch.tensor([gumbel_scale]))

    def seq_make_step(self, return_dict, use_cache):
        def custom_forward(*inputs):
            decoder_attention_mask, decoder_inputs_embeds, past_key_values, hidden_states, attention_mask,\
            decoder_head_mask, head_mask, output_attentions, output_hidden_states, dummy_tensor = inputs
            decoder_outputs = self.decoder(
                input_ids=None,  # decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            lm_logits = self.compute_logits(decoder_outputs)
            return lm_logits
        return custom_forward

    def gumbel_decode(
            self, decoder_input_ids, decoder_attention_mask, decoder_inputs_embeds, past_key_values,
            hidden_states, attention_mask, decoder_head_mask, head_mask, use_cache, output_attentions,
            output_hidden_states, return_dict, temperature=1., top_k=0, top_p=1., epsilon=0.
    ):
        """
        Decode with Gumbel softmax sampling with specified temperature
        """
        if not hasattr(self, "gumbel_dist"):
            self.create_gumbel_distribution()

        output_logits = []
        modified_logits = []
        decoder_inputs_embeds = None
        output_onehot = []
        for tok_ind in range(decoder_input_ids.shape[1]):
            if tok_ind == 0:
                one_hot_softmax = torch.zeros((decoder_input_ids.shape[0], self.decoder.embed_tokens.num_embeddings)).to(decoder_input_ids.device)
                one_hot_softmax[:, 0] = 1.
                decoder_inputs_embeds = (one_hot_softmax @ self.decoder.embed_tokens.weight).unsqueeze(1)
            else:
                decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, (one_hot_softmax @ self.decoder.embed_tokens.weight).unsqueeze(1)],
                                          dim=1)

            # decoder_outputs = self.decoder(
            #     input_ids=None,  # decoder_input_ids,
            #     attention_mask=decoder_attention_mask,
            #     inputs_embeds=decoder_inputs_embeds,
            #     past_key_values=past_key_values,
            #     encoder_hidden_states=hidden_states,
            #     encoder_attention_mask=attention_mask,
            #     head_mask=decoder_head_mask,
            #     encoder_head_mask=head_mask,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            #
            # lm_logits = self.compute_logits(decoder_outputs)

            dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

            lm_logits = checkpoint.checkpoint(self.seq_make_step(return_dict=True, use_cache=use_cache), decoder_attention_mask, decoder_inputs_embeds, past_key_values, hidden_states, attention_mask,
                         decoder_head_mask, head_mask, output_attentions, output_hidden_states, dummy_tensor)

            last_token_logits = lm_logits[:, -1, :]
            last_token_logits += self.gumbel_dist.sample(last_token_logits.shape).squeeze(-1).to(decoder_input_ids.device)
            last_token_logits_filtered = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)

            last_token_logits = torch.log(torch.nn.functional.softmax(last_token_logits) * epsilon + torch.nn.functional.softmax(last_token_logits_filtered) * (1. - epsilon))

            output_logits.append(lm_logits[:, -1, :].unsqueeze(1))
            output_logits.append(last_token_logits.unsqueeze(1))

            one_hot_softmax = nn.functional.gumbel_softmax(
                last_token_logits, tau=temperature, hard=True
            )

            output_onehot.append(one_hot_softmax.unsqueeze(1))

        with torch.no_grad():
            decoder_outputs = self.decoder(
                input_ids=None,  # decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        output_logits = torch.cat(output_logits, dim=1)
        modified_logits = torch.cat(modified_logits, dim=1)
        output_onehot = torch.cat(output_onehot, dim=1)

        return decoder_outputs, output_logits, output_onehot, modified_logits

    def teacher_forcing_decode(
            self, decoder_input_ids, decoder_attention_mask, decoder_inputs_embeds, past_key_values,
            hidden_states, attention_mask, decoder_head_mask, head_mask, use_cache, output_attentions,
            output_hidden_states, return_dict, top_k=0, top_p=1.
    ):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.compute_logits(decoder_outputs)
        # lm_logits = top_k_top_p_filtering(lm_logits.permute(0,2,1), top_k=top_k, top_p=top_p).permute(0,2,1)  # this is implemented in method generate

        return decoder_outputs, lm_logits

    def top_p_decode(
            self, decoder_input_ids, decoder_attention_mask, decoder_inputs_embeds, past_key_values,
            hidden_states, attention_mask, decoder_head_mask, head_mask, use_cache, output_attentions,
            output_hidden_states, return_dict, temperature=1., top_k=0, top_p=1., epsilon=0.
    ):
        """
        Decode with top p gradual sampling for RL objective
        """
        decoder_inputs_embeds = None
        modified_logits = []
        output_logits = []
        output_tokens = []
        batch_size, seq_len = decoder_input_ids.shape[:2]
        for tok_ind in range(seq_len):
            if tok_ind == 0:
                decoder_inputs_embeds = self.decoder.embed_tokens(torch.LongTensor([[0]]).repeat(batch_size, 1).to(decoder_input_ids.device))
            else:
                decoder_inputs_embeds = torch.cat([
                    decoder_inputs_embeds, self.decoder.embed_tokens(next_tokens)
                ], dim=1)

            # decoder_outputs = self.decoder(
            #     input_ids=None,  # decoder_input_ids,
            #     attention_mask=decoder_attention_mask,
            #     inputs_embeds=decoder_inputs_embeds,
            #     past_key_values=past_key_values,
            #     encoder_hidden_states=hidden_states,
            #     encoder_attention_mask=attention_mask,
            #     head_mask=decoder_head_mask,
            #     encoder_head_mask=head_mask,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            #
            # lm_logits = self.compute_logits(decoder_outputs)

            dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
            lm_logits = checkpoint.checkpoint(self.seq_make_step(return_dict=True, use_cache=use_cache),
                                              decoder_attention_mask, decoder_inputs_embeds, past_key_values,
                                              hidden_states, attention_mask,
                                              decoder_head_mask, head_mask, output_attentions, output_hidden_states,
                                              dummy_tensor)

            last_token_logits = lm_logits[:, -1, :] / temperature
            last_token_logits_filtered = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)

            last_token_logits = torch.log(torch.nn.functional.softmax(last_token_logits, dim=-1) * epsilon + torch.nn.functional.softmax(last_token_logits_filtered, dim=-1) * (1. - epsilon))

            probs = nn.functional.softmax(
                last_token_logits, dim=1
            )

            next_tokens = torch.multinomial(probs, num_samples=1)#.squeeze(1)
            output_logits.append(lm_logits[:, -1, :].unsqueeze(1))
            modified_logits.append(last_token_logits.unsqueeze(1))
            output_tokens.append(next_tokens)

        with torch.no_grad():
            decoder_outputs = self.decoder(
                input_ids=None,  # decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        output_logits = torch.cat(output_logits, dim=1)
        modified_logits = torch.cat(modified_logits, dim=1)
        output_tokens = torch.cat(output_tokens, dim=1)

        return decoder_outputs, output_logits, output_tokens, modified_logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoding_style="tf",  # options: teacher forcing (tf), gumbel (gumbel), top p (rl)
        temperature=1.,
        top_k=0,
        top_p=1.,
        epsilon=0.
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(s__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            decoder_input_ids[:, 0] = 0

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode

        decode_args =  (
                decoder_input_ids, decoder_attention_mask, decoder_inputs_embeds, past_key_values,
                hidden_states, attention_mask, decoder_head_mask, head_mask, use_cache, output_attentions,
                output_hidden_states, return_dict
            )

        if decoding_style == "tf":
            decoder_outputs, lm_logits = self.teacher_forcing_decode(*decode_args, top_k=top_k, top_p=top_p)
            input_onehot = output_onehot = target_onehot = modified_logits = output_tokens = None
        elif decoding_style == "gumbel":
            input_onehot = nn.functional.one_hot(input_ids, num_classes=self.decoder.embed_tokens.num_embeddings).float()
            target_onehot = nn.functional.one_hot(decoder_input_ids, num_classes=self.decoder.embed_tokens.num_embeddings).float()
            decoder_outputs, lm_logits, output_onehot, modified_logits = self.gumbel_decode(*decode_args, temperature=temperature, top_k=top_k, top_p=top_p, epsilon=epsilon)
            output_tokens = None
        elif decoding_style == "rl":
            decoder_outputs, lm_logits, output_tokens, modified_logits = self.top_p_decode(*decode_args, temperature=temperature, top_k=top_k, top_p=top_p, epsilon=epsilon)
            input_onehot = output_onehot = target_onehot = None
        else:
            raise ValueError(f"`decoding_style` is {decoding_style} but supported values are: tf|gumbel|rl")

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))  # TODO need to set temperature here?
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return AdversarialSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            input_onehot=input_onehot,
            output_onehot=output_onehot,
            target_onehot=target_onehot,
            output_tokens=output_tokens,
            modified_logits=modified_logits
        )


class SeqT5_Discriminator(SeqT5):
    def __init__(self, config):
        super(SeqT5_Discriminator, self).__init__(config) # todo check
    #     self.fc_layer = nn.Linear(config.d_model, 1)
    #
    # def compute_logits(self, decoder_output):
    #     sequence_output = decoder_output[0]
    #
    #     # Set device for model parallelism
    #     if self.model_parallel:
    #         torch.cuda.set_device(self.encoder.first_device)
    #         self.lm_head = self.lm_head.to(self.encoder.first_device)
    #         sequence_output = sequence_output.to(self.lm_head.weight.device)
    #
    #     prob = self.fc_layer(sequence_output)
    #     return prob

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            decoding_style="tf",  # options: teacher forcing (tf), gumbel (gumbel), top p (rl)
            temperature=1.,
            top_k=0,
            top_p=1.,
            epsilon=0.
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(s__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            decoder_input_ids[:, 0] = 0

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode

        # decode_args =  (
        #     decoder_input_ids, decoder_attention_mask, decoder_inputs_embeds, past_key_values,
        #     hidden_states, attention_mask, decoder_head_mask, head_mask, use_cache, output_attentions,
        #     output_hidden_states, return_dict
        # )

        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     encoder_head_mask=head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        #
        # lm_logits = self.compute_logits(decoder_outputs)

        return hidden_states


def test_T5():
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = SeqT5.from_pretrained('t5-small')

    input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
    labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
    outputs_tf = model(input_ids=input_ids, labels=labels, decoding_style="tf")
    outputs_gumbel = model(input_ids=input_ids, labels=labels, decoding_style="gumbel", top_k=1)
    outputs_rl = model(input_ids=input_ids, labels=labels, decoding_style="rl", top_k=1)

    input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you, but cats lead to heart attack ",
                               return_tensors="pt").input_ids  # Batch size 1
    outputs = model.generate(input_ids, temperature=1., top_p=0.9, do_sample=True, max_length=100, repetition_penalty=1.)
    # references for repetition penalty https://huggingface.co/blog/how-to-generate
    print()


if __name__ == "__main__":
    test_T5()