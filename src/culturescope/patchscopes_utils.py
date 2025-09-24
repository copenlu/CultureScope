# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import tqdm
from general_utils import decode_tokens
from general_utils import make_inputs


# ##############
#
# Hooks
#
# ##############


def set_hs_patch_hooks_llama(
    model,
    hs_patch_config,
    module="hs",      # one of "hs", "mlp", or "attn"
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
    """
    Install hooks on a Llama-family model to patch hidden states at specified layers.
    Also normalizes pad_token_id so generate() uses a scalar integer.
    """

    # ─── pad_token_id normalization ───
    # Apply if model has tokenizer attribute
    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        pad = tok.pad_token_id
        if isinstance(pad, (list, tuple)) and pad:
            pad = pad[0]
        if pad is None:
            pad = tok.eos_token_id
        # assign back
        tok.pad_token_id = pad
        # underlying model config
        mdl = getattr(model, "model", model)
        if hasattr(mdl, "config") and hasattr(mdl.config, "pad_token_id"):
            mdl.config.pad_token_id = pad
        if hasattr(mdl, "generation_config") and hasattr(mdl.generation_config, "pad_token_id"):
            try:
                setattr(mdl.generation_config, "pad_token_id", pad)
            except Exception:
                pass
    # ─── end pad_token_id normalization ───

    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, inputs):
            seq_len = inputs[0].shape[1]
            if generation_mode and seq_len == 1:
                return
            for pos, hs_val in position_hs:
                inputs[0][0, pos] = hs_val

        def post_hook(module, inputs, outputs):
            # outputs[0] has shape (batch, seq_len, hidden_size)
            if "mlp" in name or "skip_ln" in name:
                seq_len = outputs[0].shape[0]
            else:
                seq_len = outputs[0].shape[1]
            if generation_mode and seq_len == 1:
                return
            for pos, hs_val in position_hs:
                if "mlp" in name or "skip_ln" in name:
                    outputs[0][pos] = hs_val
                else:
                    outputs[0][0, pos] = hs_val

        return pre_hook if patch_input else post_hook

    hooks = []
    for layer_idx, positions in hs_patch_config.items():
        hook_fn = patch_hs(f"patch_{module}_{layer_idx}", positions, patch_input, generation_mode)
        if patch_input:
            if module == "hs":
                hooks.append(model.model.layers[layer_idx].register_forward_pre_hook(hook_fn))
            elif module == "mlp":
                hooks.append(model.model.layers[layer_idx].mlp.register_forward_pre_hook(hook_fn))
            elif module == "attn":
                hooks.append(model.model.layers[layer_idx].self_attn.register_forward_pre_hook(hook_fn))
            else:
                raise ValueError(f"Unsupported module: {module}")
        else:
            is_final = (layer_idx == len(model.model.layers) - 1)
            if skip_final_ln and module == "hs" and is_final:
                hooks.append(
                    model.model.norm.register_forward_hook(
                        patch_hs(f"patch_hs_{layer_idx}_skip_ln", positions, patch_input, generation_mode)
                    )
                )
            else:
                if module == "hs":
                    hooks.append(model.model.layers[layer_idx].register_forward_hook(hook_fn))
                elif module == "mlp":
                    hooks.append(model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn))
                elif module == "attn":
                    hooks.append(model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn))
                else:
                    raise ValueError(f"Unsupported module: {module}")
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


# ##############
#
# Inspection
#
# ##############


def inspect(
    mt,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    generation_mode=False,
    max_gen_len=20,
    verbose=False,
    temperature=None, args=None
):
    """Inspection via patching."""
    # prepare inputs
    inp_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
    if position_target < 0:
        position_target = len(inp_target["input_ids"][0]) + position_target
    inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)

    # run to collect hs_cache
    hs_cache = []
    store_hooks = []
    if module == "mlp":
        def store_mlp(module, inp, out):
            hs_cache.append(out[0])
        for lyr in mt.model.model.layers:
            store_hooks.append(lyr.mlp.register_forward_hook(store_mlp))
    elif module == "attn":
        def store_attn(module, inp, out):
            hs_cache.append(out[0].squeeze())
        for lyr in mt.model.model.layers:
            store_hooks.append(lyr.self_attn.register_forward_hook(store_attn))
    out0 = mt.model(**inp_source, output_hidden_states=True)
    if module == "hs":
        # hs_cache = []
        # for l in range(mt.num_layers):
        #     breakpoint()
        #     hs_cache.append(out0["hidden_states"][l+1][0])
        hs_cache = [out0["hidden_states"][l+1][0] for l in range(mt.num_layers)]
    remove_hooks(store_hooks)
    
    # setup patch
    hs_patch_config = {}
    # if layer_target > 24:
    #     layer_to_patch = [layer_target, layer_target-1, layer_target-2, 0, 1, 2]
    # elif layer_target > 19:
    #     layer_to_patch = [layer_target, layer_target-1, layer_target-2, 0, 1, 2]
    # elif layer_target > 14:
    #     layer_to_patch = [layer_target, layer_target-1, layer_target-2, 0, 1, 2]
    if layer_target > 1:
        layer_to_patch = [layer_target, layer_target-1, layer_target-2]
    elif layer_target > 0:
        layer_to_patch = [layer_target, layer_target-1]
    else:
        layer_to_patch = [layer_target]

    for l in range(mt.num_layers):
        if l in layer_to_patch:
            # weighted sum with N / V
            if args.collapse == "avg":
                patch_vec = torch.mean(torch.stack([hs_cache[l][p] for p in position_source]), dim=0)
            elif args.collapse == "sum":
                patch_vec = torch.sum(torch.stack([hs_cache[l][p] for p in position_source]), dim=0)
            # multi-token patching: averaging
            # avg_vec = torch.mean(hs_cache[l][-position_source:], dim=0)

            hs_patch_config[l] = [(position_target, patch_vec)]

            # single token patching
            # hs_patch_config[l] = [(position_target, hs_cache[l][position_source])]      

    # if layer_target > 20:
    #     hs_patch_config = { layer_target: , 0:  [(position_target, hs_cache[0][position_source])],
    #                         15:  [(position_target, hs_cache[15][position_source])]}
    # elif layer_target > 5:
    #     hs_patch_config = { layer_target: [(position_target, hs_cache[layer_source][position_source])], 0:  [(position_target, hs_cache[0][position_source])],
    #                         layer_target-1: [(position_target, hs_cache[layer_source-1][position_source])], layer_target-2: [(position_target, hs_cache[layer_source-2][position_source])]}
    # else:
    #     hs_patch_config = { layer_target: [(position_target, hs_cache[layer_source][position_source])] }
    
    skip_final_ln = (layer_source == layer_target == mt.num_layers - 1)


    patch_hooks = mt.set_hs_patch_hooks(mt.model, hs_patch_config,
        module=module, patch_input=False, skip_final_ln=skip_final_ln,
        generation_mode=True)

    # generate or predict
    if verbose:
        pass  # optional prints omitted
    if generation_mode:
        # ensure pad token scalar
        pad = mt.model.config.pad_token_id
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            max_length=len(inp_target["input_ids"][0]) + max_gen_len,
            pad_token_id=pad,
            temperature=temperature if temperature else 1.0,
            # do_sample=bool(temperature),
            # top_k=0, early_stopping=True
        )[0][len(inp_target["input_ids"][0])-3:]
        out_text = mt.tokenizer.decode(output_toks)
    else:
        out_raw = mt.model(**inp_target)
        probs = torch.softmax(out_raw.logits[0, -1, :], dim=0)
        t_id = torch.argmax(probs).item()
        out_text = decode_tokens(mt.tokenizer, [t_id])[0]
    remove_hooks(patch_hooks)

    return out_text

def evaluate_patch_next_token_prediction(
    mt,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    position_prediction=-1,
    transform=None,
):
  """Evaluate next token prediction."""
  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  # adjust position_target to be absolute rather than relative
  inp_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
  if position_target < 0:
    position_target = len(inp_target["input_ids"][0]) + position_target

  # first run the the model on without patching and get the results.
  inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
  output_orig = mt.model(**inp_source, output_hidden_states=True)
  dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
  _, answer_t_orig = torch.max(dist_orig, dim=0)
  hidden_rep = output_orig["hidden_states"][layer_source + 1][0][
      position_source
  ]
  if transform is not None:
    hidden_rep = transform(hidden_rep)

  # now do a second run on prompt, while patching the input hidden state.
  hs_patch_config = {layer_target: [(position_target, hidden_rep)]}
  if layer_source == layer_target == mt.num_layers - 1:
    skip_final_ln = True
  else:
    skip_final_ln = False
  patch_hooks = mt.set_hs_patch_hooks(
      mt.model,
      hs_patch_config,
      module=module,
      patch_input=False,
      skip_final_ln=skip_final_ln,
      generation_mode=True,
  )
  output = mt.model(**inp_target)
  dist = torch.softmax(output.logits[0, position_prediction, :], dim=0)
  _, answer_t = torch.max(dist, dim=0)

  # remove patching hooks
  remove_hooks(patch_hooks)

  prec_1 = (answer_t == answer_t_orig).detach().cpu().item()
  surprisal = -torch.log(dist_orig[answer_t]).detach().cpu().numpy()

  return prec_1, surprisal


def evaluate_patch_next_token_prediction_x_model(
    mt_1,
    mt_2,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    position_prediction=-1,
    transform=None,
):
  """evaluate next token prediction across models."""
  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  # adjust position_target to be absolute rather than relative
  inp_target = make_inputs(mt_2.tokenizer, [prompt_target], device=mt_2.device)
  if position_target < 0:
    position_target = len(inp_target["input_ids"][0]) + position_target

  # first run the the model on without patching and get the results.
  inp_source = make_inputs(mt_1.tokenizer, [prompt_source], device=mt_1.device)
  output_orig = mt_1.model(**inp_source, output_hidden_states=True)
  dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
  _, answer_t_orig = torch.max(dist_orig, dim=0)
  hidden_rep = output_orig["hidden_states"][layer_source + 1][0][
      position_source
  ]
  if transform is not None:
    hidden_rep = transform(hidden_rep)

  # now do a second run on prompt, while patching the input hidden state.
  hs_patch_config = {layer_target: [(position_target, hidden_rep)]}
  skip_final_ln = False
  patch_hooks = mt_2.set_hs_patch_hooks(
      mt_2.model,
      hs_patch_config,
      module=module,
      patch_input=False,
      skip_final_ln=skip_final_ln,
      generation_mode=True,
  )
  output = mt_2.model(**inp_target)
  dist = torch.softmax(output.logits[0, position_prediction, :], dim=0)
  _, answer_t = torch.max(dist, dim=0)

  # remove patching hooks
  remove_hooks(patch_hooks)

  prec_1 = answer_t.detach().cpu().item() == answer_t_orig.detach().cpu().item()
  surprisal = -torch.log(dist_orig[answer_t]).detach().cpu().numpy()

  return prec_1, surprisal