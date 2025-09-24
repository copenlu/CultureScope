import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from general_utils import ModelAndTokenizer
from patchscopes_utils import (
    set_hs_patch_hooks_llama,
    inspect,
)
import spacy

import argparse
import glob, random
from tqdm import tqdm
import ast
import os, re

# for pos tagging
en_lemmatizer = spacy.load("en_core_web_sm")

# -- Configuration --
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
import pandas as pd
import torch
from tqdm import tqdm
from general_utils import ModelAndTokenizer
from patchscopes_utils import (
    set_hs_patch_hooks_llama,
    inspect,
)
from transformers import CohereTokenizerFast

# -- Configuration --
DTYPE = torch.float16

generation_mode = True
max_gen_len = 50

position_source = -1
position_target = -1

# enable tqdm progress bars for pandas
tqdm.pandas()


# -- Model Initialization --

def load_model_and_tokenizer(args):
    MODEL_NAME = args.model_name

    if "aya" in MODEL_NAME.lower():
        tokenizer = CohereTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)
        mt = ModelAndTokenizer(MODEL_NAME, tokenizer=tokenizer, torch_dtype=DTYPE)
    else:
        mt = ModelAndTokenizer(MODEL_NAME, torch_dtype=DTYPE)
    # Attach the correct patch hooks based on model name
    if "pythia" in MODEL_NAME.lower():
        mt.set_hs_patch_hooks = set_hs_patch_hooks_neox
    elif "llama-2" in MODEL_NAME.lower():
        mt.set_hs_patch_hooks = set_hs_patch_hooks_llama2
    else:
        mt.set_hs_patch_hooks = set_hs_patch_hooks_llama
    mt.model.eval()

    # Ensure pad_token_id is a valid integer for generation
    pad_id = mt.tokenizer.pad_token_id
    if isinstance(pad_id, (list, tuple)) and pad_id:
        pad_id = pad_id[0]
    if pad_id is None:
        pad_id = mt.tokenizer.eos_token_id
    mt.tokenizer.pad_token_id = pad_id
    mt.model.config.pad_token_id = pad_id
    # Also update new generation_config if present
    if hasattr(mt.model, 'generation_config'):
        try:
            mt.model.generation_config.pad_token_id = pad_id
        except Exception:
            pass

    return mt


def select_answer_span(row, mt):
    if isinstance(row["LM_Answer"], list):
        # If LM_Answer is a list, take the first element
        answer_only_tok = mt.tokenizer.encode(row["LM_Answer"][0].split("\n\n")[0], add_special_tokens=False)
    else:
        answer_only_tok = mt.tokenizer.encode(row["LM_Answer"].split("\n\n")[0], add_special_tokens=False)
    answer_cut = []
    patience = 0
    for at in answer_only_tok:
        if at in avoid_token_ids:
            answer_cut.append(at)
            patience += 1
        else:
            answer_cut.append(at)
            break
            # patience += 1
            # if patience > 2:
            #     break

    answer_span = mt.tokenizer.decode(answer_only_tok)

    answer_span = answer_span.replace("[", "")
    answer_span = answer_span.replace("]", "")
    answer_span = answer_span.replace("<|eot_id|>", "")

    answer_span = answer_span.replace("<|END_OF_TURN_TOKEN|>", "")
    answer_span = answer_span.replace("<|im_end|>", "")

    return answer_span, len(answer_only_tok) - patience


def normad_answer_span(row, mt):
    
    answer_txt = row["LM_Answer"][0]

    if "Explanation" not in answer_txt:
        first_sen = "Explanation:" + answer_txt.split(".")[0]
        print("** patchscope input **")
        print(first_sen)

    else:
        exp_only = answer_txt.split("Explanation:")[1]

        first_sen = "Explanation:" + exp_only.split(". ")[0] + "."

    return first_sen


def tag_input(seq, mt, args):
    doc = en_lemmatizer(seq)
    pos_tagged = [word.pos_ for word in doc]
    seq_splitted = [" " + word.text for word in doc]

    encoding = mt.tokenizer(seq_splitted, is_split_into_words=True)
    tokens = mt.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    word_ids = encoding.word_ids(batch_index=0)

    # country ids
    country_ids = word_ids[-1]

    try:
        if "Qwen" in mt.model_name:
            assert len(set(word_ids)) == len(seq_splitted)
        else:
            assert (len(set(word_ids)) - 1) == len(seq_splitted)
    except:
        breakpoint()

    if args.dataset == "normad":
        answer_cnt = tokens.count("ĠExplanation")
        gubun = "ĠExplanation"

        # if explanation not in the tokens
        # do the exception
        # if answer_cnt == 0:
        #     answer_cnt = tokens.count("ĠAnswer")
        #     gubun = "ĠAnswer"

    else:
        answer_cnt = tokens.count("ĠAnswer")
        gubun = "ĠAnswer"

    start = 0
    for _ in range(answer_cnt):
        answer_start_idx = tokens[start:].index(gubun) + start
        start = answer_start_idx + 1
    
    
    answer_word_idx = word_ids[answer_start_idx]

    # Map tokens to words
    enc_pos = 0
    pos_mask = []  # len(tokens)
    already = []

    for mask_idx, wids in enumerate(word_ids):
        if wids == None:
            pos_mask.append(-1)
            continue

        if wids < answer_word_idx + 2:
            # answer_word_idx: skip Question part & "Answer / :" / 2: skip "Question:" part
            pos_mask.append(-1)
            continue

        # if wids in already:
            # if wids == country_ids:
            #     pos_mask.append(mask_idx)
            
            # else:
            # pos_mask = pos_mask[:-1]
            # pos_mask.append(-1)

        # else:
        #     if args.dataset == "normad":
        #         if pos_tagged[wids] in ["NOUN", "PROPN",  "ADP"]:
        #             pos_mask.append(mask_idx)
        #         else:
        #             pos_mask.append(-1)
        #     else:
        # else:
        if pos_tagged[wids] in ["NOUN", "VERB", "PROPN"]:
            pos_mask.append(mask_idx)
        else:
            pos_mask.append(-1)

        already.append(wids)
        

    try:
        assert len(pos_mask) == len(tokens)
    except:
        breakpoint()

    pos_mask = [pm for pm in pos_mask if pm != -1]

    if pos_mask == []:
        # take the last token
        pos_mask = [-1]

    # breakpoint()


    return encoding["input_ids"], pos_mask


# -- Function to inspect entities --
def inspect_entities(args, entities, country=None, topic_df=None, mt=None):
    """
    Inspect each entity in `entities` at layers 0-9 and save results.

    Args:
      entities (List[str]): list of entity strings to inspect.
      mt: ModelAndTokenizer instance with hooks attached.
      output_csv (str): path to the CSV file for output.

    Returns:
      pandas.DataFrame with columns subj, inspect_layer0...inspect_layer9.
    """
    # Build DataFrame

    if args.dataset != "normad":
        entities["answer_list"] = entities["answer_list"].apply(ast.literal_eval)

    if args.dataset != "fmlama":
        entities["LM_Answer"] = entities["LM_Answer"].apply(ast.literal_eval)

    df = pd.DataFrame(
        columns=["ID", "Topic", "Country", "Question", "Gold Answer", "LM Answer", "layer idx", "Decoded Concepts"])

    layer_num = mt.model.config.num_hidden_layers

    # sampling
    random_list = [140, 58, 382, 39, 271, 93, 242, 94, 152, 464, 367, 202, 387, 253, 166, 493, 21, 246, 323, 43, 308,
                   128, 488, 89, 351, 40, 161, 474, 309, 43]

    if args.start_num == 0:
        end_num = 1200

    else:
        end_num = len(entities)


    for row_idx in tqdm(range(args.start_num, end_num)):
        # if row_idx not in random_list:
        #     continue

        row = entities.iloc[row_idx]

        # skip the numeric answers
        if "Provide in Arabic numerals" in row.Question:
            continue

        if "Provide in HH" in row.Question:
            continue

        if "Provide in MM/DD" in row.Question:
            # print(row.Question)
            continue

        # skip no-answers
        if row["answer_list"] == []:
            continue

        if args.dataset == "normad":
            answer_span = normad_answer_span(row, mt)
        else:
            answer_span, answer_len = select_answer_span(row, mt)


        if args.dataset in ["cultural_bench", "normad"]:
            # seq = "Question: %s %s" %(row.Question, row.Country.capitalize())
            seq = "Question: %s %s %s" % (row.Question, answer_span, row.Country)
        
        else:
            seq = "Question: %s Answer:%s" %(row.Question, answer_span)
        # elif args.dataset == "normad":
        #     seq = "Question: %s %s %s" %(row.Question, answer_span, row.Country)

        # seq = "Question: %s Answer: Jujeh kabab" %(row.Question)
        seq_splitted, position_source_list = tag_input(seq, mt, args)

        # print(answer_span)
        if args.dataset == "blend":
            topic = topic_df[topic_df.ID == row.ID].iloc[0].Topic
        elif args.dataset == "fmlama":
            topic = "food"
        elif args.dataset == "cultural_bench":
            topic = "value"
        elif args.dataset == "normad":
            topic = "normad"
        elif args.dataset == "camel":
            topic = row.answer_list[0]

        # print(row.Question, row["LM_Answer"])
        if args.dataset == "fmlama":
            country = re.findall(r'\"(.*?)\"', row.Question)[0]
        elif args.dataset == "normad":
            country = row.Country

        for layer in range(0, layer_num):
            col = f"inspect_layer{layer}"
            # print("\n\n", col)
            # prompt_target = (
            #     "Generate a sentence with a cultural norm or value"
            #     "Austria: Arriving 5 to 10 minutes early for appointments and events is considered respectful. Additionally, always introduce oneself by name during phone calls., ..., "
            #     "North Macedonia: One must behave more formally and respectfully around their elders, avoiding rude or inappropriate language., ..., "
            #     "x"
            # )
            prompt_target = (
                "Generate associated words, Syria, Oman, Jordan, Qatar, West Asia, Turkey, Israel, Lebanon, ..., "
                "Leonardo DiCaprio, Tom Cruise, Kate Winslet, Brad Pitt, Actor, ..., "
                "Samsung, Cell Phone, TV, Apple, Nokia, South Korea, Electronics, , ..., "
                "x"
            )
            # % (row.country, topic)
            outputs = inspect(
                mt,
                seq_splitted,
                prompt_target,
                layer, layer,
                position_source_list, position_target,
                generation_mode=generation_mode,
                max_gen_len=max_gen_len,
                verbose=False, args=args
            )
            # print(f"Layer {layer} outputs:")
            # print(outputs)
            # print(row["answer_list"])
            # print("=" * 50)

            # ["ID", "Topic", "Country", "Question", "Gold Answer", "LM Answer", "layer_idx", "Decoded Concepts"]
            df.loc[len(df)] = [
                row["ID"], topic, country, row.Question, row["answer_list"], row["LM_Answer"], layer, outputs]
        # breakpoint()
    return df


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        print('No GPU available, using the CPU instead.')


def load_data(file_list):
    topic_df = None
    for b_csv in file_list:
        tmp = pd.read_csv(b_csv)
        if topic_df is None:
            topic_df = tmp
        else:
            topic_df = pd.concat([topic_df, tmp])
    return topic_df


# -- Main Execution --
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model : meta-llama/Llama-Guard-3-8B ,
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="geolama")
    parser.add_argument("--mode", type=str, default="short_answer")
    parser.add_argument("--collapse", type=str, default="avg")
    parser.add_argument("--start_num", type=int, default=0)

    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()
    print(args)
    set_seed(args)

    # load_model
    mt = load_model_and_tokenizer(args)

    avoid_tokens = ["Answer", ":", " Answer", ":"]
    avoid_token_ids = []
    for a_t in avoid_tokens:
        tmp = mt.tokenizer.encode(a_t, add_special_tokens=False)
        avoid_token_ids += tmp

    # topic info
    blend_csv_list = glob.glob("../Data/BLEnD/data/questions/*.csv")
    topic_df = load_data(blend_csv_list)

    # load_data
    # id_info_list = glob.glob("../data_final/blend_region/blend_region_based_*.csv")
    # id_info = load_data(id_info_list)

    # load results
    model_prefix = {"meta-llama/Meta-Llama-3.1-8B-Instruct": "llama", "CohereLabs/aya-expanse-8b": "aya",
                    "Qwen/Qwen2.5-7B-Instruct": "qwen"}[args.model_name]

    if args.dataset == "blend":
        result_path = "../lens_result/%s/short_answer" % model_prefix
        file_list = glob.glob("%s/*_result_.csv" % result_path)
    elif args.dataset == "fmlama":
        result_path = "../fmlama_result/%s" % model_prefix
        file_list = glob.glob("%s/fmlama_result_.csv" % result_path)
    elif args.dataset == "cultural_bench":
        result_path = "../cultural_bench_result"
        file_list = glob.glob("%s/%s_result_baseline.csv" % (result_path, model_prefix))
    elif args.dataset == "normad":
        result_path = "../normad_result"
        file_list = glob.glob("%s/%s_result_baseline_100.csv" % (result_path, model_prefix))
    elif args.dataset == "camel":
        result_path = "../camel_result"
        file_list = glob.glob("%s/%s_result_baseline.csv" % (result_path, model_prefix))
    # total_result = load_data(file_list)

    # print(sorted_file_list[args.start_num: args.start_num+4])
    if args.dataset == "blend":
        print(sorted(file_list))
        print(len(file_list))
        for f in sorted(file_list)[args.start_num: args.start_num + 4]:
            tmp_result = pd.read_csv(f)
            print(f"Processing {f}...")
            country = f.split("/")[-1].split("_result")[0]

            if country in ["North_Korea", "West_Java"]:
                continue

            if os.path.exists("%s/patchscope_result_%s_pos_%s.csv" % (result_path, country, args.collapse)):
                print(f"Skipping {country}, already processed.")
                continue

            # Run inspection and write results
            decoded_results = inspect_entities(args, tmp_result, country, topic_df, mt)
            # print(f"Done: results written to cultural_inspections.csv with {len(results)} rows")

            decoded_results.to_csv("%s/patchscope_result_%s_pos_%s.csv" % (result_path, country, args.collapse),
                                   index=False)

    elif args.dataset == "fmlama":
        for f in file_list:
            tmp_result = pd.read_csv(f)

            decoded_results = inspect_entities(args, tmp_result, mt=mt)
            # print(f"Done: results written to cultural_inspections.csv with {len(results)} rows")

            decoded_results.to_csv("%s/patchscope_result_%s.csv" % (result_path, args.collapse),
                                   index=False)

    elif args.dataset == "cultural_bench":
        result = pd.read_csv(file_list[0])
        decoded_results = inspect_entities(args, result, mt=mt)
        decoded_results.to_csv("%s/patchscope_%s_%s.csv" % (result_path, model_prefix, args.collapse),
                               index=False)

    elif args.dataset == "normad":
        result = pd.read_csv(file_list[0])
        decoded_results = inspect_entities(args, result, mt=mt)
        decoded_results.to_csv("%s/patchscope_%s_%s.csv" % (result_path, model_prefix, args.collapse),
                               index=False)
    
    elif args.dataset == "camel":
        result = pd.read_csv(file_list[0])
        decoded_results = inspect_entities(args, result, mt=mt)
        decoded_results.to_csv("%s/patchscope_%s_%s_%d.csv" % (result_path, model_prefix, args.collapse, args.start_num),
                               index=False)

