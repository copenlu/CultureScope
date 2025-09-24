import pandas as pd
import numpy as np

import argparse
import torch
import ast

import re, json, glob
import nltk, os
from nltk.corpus import stopwords
from tqdm import tqdm

from transformers import AutoModelForCausalLM

from transformers import AutoTokenizer, AutoModel

import torch.nn.functional as F
from collections import Counter


STOPWORDS = set(stopwords.words('english'))


def load_result(path):

    result = pd.read_csv(path)
    return result


def clean(txt, country, topic, gold_answer=None):
    
    # "Generate associated words, Syria: Oman, Jordan, Qatar, ..., "
    #                 "Leonardo DiCaprio: Tom Cruise, Kate Winslet, Brad Pitt, ..., "
    #                 "Samsung: Cell Phone, TV, Apple, Nokia, ..., "
    topic = topic.split("/")[-1]
    
    if "...," not in txt:
        return []
    
    tmp = txt.split("Generate associated")[0]

    # 21 May
    tmp = tmp.split("\n")[0]

    # ver 1
    tmp = tmp.split("...,")[1]

    # try:
    #     tmp = tmp.split(".")[0]
    #     tmp = tmp.split(topic+":")[1]
    #     tmp = tmp.split("\n")[0]
        
    # except:
    #     breakpoint()
    tmp = tmp.split(",")

    # ver 1
    # tmp = [t.split(":")[-1] for t in tmp]

    # remove any character that is not a word character or whitespace
    tmp = [re.sub(r'[^\w\s]', '', t) for t in tmp]

    # ver 1: remove target_prompt
    # tmp = [t.strip() for t in tmp if t not in [country, topic]]


    # collapse words by space
    # final = []
    # for t in tmp:
    #     if t != "":
    #         final += t.split()

    # 21 May
    tmp = [t.strip() for t in tmp if t.strip() != ""]

    # remove stopwords
    tmp = [t for t in tmp if t not in STOPWORDS]

    tmp = [t for t in tmp if len(t) > 1]

    for t in tmp:
        if t.startswith("x") and gold_answer.endswith(t.replace("x", "")):
            tmp.append(gold_answer)

    tmp = [t for t in tmp if not t[0].startswith("x")]

    tmp = [t.title() for t in tmp if len(t.split()) < 4]

    return tmp


def prepare_country_embeds(country_names, tokenizer, embedding_layer):
    country_embed_list = []
    country_list = []
    for country in country_names:
        country_embed = embedding_layer(tokenizer.encode(country, add_special_tokens=False, return_tensors="pt"))
        country_embed = torch.mean(country_embed, dim=1).squeeze(0)

        country_embed_list.append(country_embed)
        country_list.append(country)


    return country_list, torch.stack(country_embed_list)


def annotate_country(concept_emb, country_emb, country_list):
    
    scores = F.cosine_similarity(concept_emb, country_emb, dim=-1)
    
    top_3 = torch.argsort(scores, descending=True)[:3]

    selected_countries = []
    for t in top_3:
        if scores[t] > 0.3:
            selected_countries.append(country_list[t])
    
    return selected_countries


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def convert_country_map(country_map):

    region_map = {}
    for country, attributes in country_map.items():
        region_map.setdefault(attributes["subregion"], [])
        region_map[attributes["subregion"]].append(country)
    
    return region_map


PRECOMPUTED_COUNTRY_EMB = {}


def filter_with_other_countries(row, country_specific, cc_emb_list, country_map, region_to_countries, sentence_transformer, tokenizer):

    try:
        subregion = country_map[row.Country]["subregion"]
    except:
        subregion = country_map["Syria"]["subregion"]

    close_countries = region_to_countries[subregion]
    precompute_key = "%s-%s" %(subregion, row.Country)

    if precompute_key not in PRECOMPUTED_COUNTRY_EMB:
        country_emb_list = []
        for cc in close_countries:
            if cc == row.Country:
                continue
            
            encoded_cc = tokenizer(cc, return_tensors="pt").to("cuda")
            cc_outputs = sentence_transformer(**encoded_cc)

            cc_emb = mean_pooling(cc_outputs, encoded_cc["attention_mask"])
            cc_emb = F.normalize(cc_emb, p=2, dim=1)
            country_emb_list.append(cc_emb.squeeze(0))
        PRECOMPUTED_COUNTRY_EMB[precompute_key] = country_emb_list
        
    else:
        country_emb_list = PRECOMPUTED_COUNTRY_EMB[precompute_key]

    country_emb = torch.stack(country_emb_list)        

    filtered = []
    filtered_emb = []
    flatten_concepts = []
    for concept, cc_emb in zip(country_specific, cc_emb_list):
        sim_scores = F.cosine_similarity(cc_emb, country_emb)
        
        # threshold
        bigger = sim_scores > 0.5

        if sum(bigger) == 0:
            filtered.append(concept)
            filtered_emb.append(cc_emb)
            # pass

    return filtered, filtered_emb, flatten_concepts


def main(args, result_file_path):

    with open('countries.json', 'r', encoding='utf-8') as f:
        country_names = json.load(f)

    if args.dataset == "blend":
        country_map = json.load(open("../process_dataset/blend/country.json"))
    else:
        country_map = json.load(open("../process_dataset/blend/camel_country.json"))

    region_to_countries = convert_country_map(country_map)

    print(result_file_path)

    result = pd.read_csv(result_file_path)
    print(len(result))

    result["Gold Answer"] = result["Gold Answer"].apply(ast.literal_eval)

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    sentence_transformer = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(args.device)


    # for faster search
    none_list = []
    total_search = {}

    # result
    num_generated_concepts = []
    country_specific = []
    scores = []

    cnt = 0
    layers = []

    zero_count = 0
    has_answer = 0

    for row_idx in tqdm(range(len(result))):
        row = result.iloc[row_idx]
        
        if args.dataset == "camel":
            decoded_concepts = clean(row["Decoded Concepts"], row.Country, row.Topic, row["Gold Answer"][-1])
        else:
            decoded_concepts = clean(row["Decoded Concepts"], row.Country, row.Topic)

        if decoded_concepts == []:
            country_specific.append([])
            num_generated_concepts.append([])
            scores.append(0.0)
            continue

        # answer_span = row["LM Answer"].split("\n\n")[0].replace("Answer", "")
        # answer_span = answer_span.replace(":", "").strip()

        answer_span = "%s %s" %(row.Country, row.Topic)

        # answer_span = row.Question.replace(row.Country.replace("_", ""), "")

        # answer_span = "%s %s" %(row.Topic, row.Question)
        # answer_span = row.Question
        encoded_answer = tokenizer(answer_span, return_tensors="pt").to(args.device)
        answer_outputs = sentence_transformer(**encoded_answer)

        answer_emb = mean_pooling(answer_outputs, encoded_answer["attention_mask"])
        answer_emb = F.normalize(answer_emb, p=2, dim=1)


        # topic & answer ? 
        dc_emb_list = []
        for dc in decoded_concepts:
            encoded_dc = tokenizer(dc, return_tensors="pt").to(args.device)
            dc_outputs = sentence_transformer(**encoded_dc)

            dc_emb = mean_pooling(dc_outputs, encoded_dc["attention_mask"])
            dc_emb = F.normalize(dc_emb, p=2, dim=1)
            dc_emb_list.append(dc_emb.squeeze(0))

        dc_emb_stacked = torch.stack(dc_emb_list)

        sim_scores = F.cosine_similarity(answer_emb, dc_emb_stacked)

        # if args.dataset == "camel":
        #     camel_topic_seq = row.Topic
        #     encoded_topic = tokenizer(camel_topic_seq, return_tensors="pt").to(args.device)
        #     topic_outputs = sentence_transformer(**encoded_topic)

        #     topic_emb = mean_pooling(topic_outputs, encoded_topic["attention_mask"])
        #     topic_emb = F.normalize(topic_emb, p=2, dim=1)


        # first filtering
        country_relevant = []
        selected_dc_emb = []
        for idx, (dc, score) in enumerate(zip(decoded_concepts, sim_scores)):
            if score > 0.2:
                # selected_country = compute_flattening_score(question, topic, similar_countries, score, sentence_transformer, tokenizer)
                # topic_score = F.cosine_similarity(dc_emb_list[idx], topic_emb)

                # if topic_score > 0.1:
                country_relevant.append([dc, score.item()])
                selected_dc_emb.append(dc_emb_list[idx])
                    # breakpoint()
            
        # print(decoded_concepts)
        # print(sim_scores)
        # print(row["Gold Answer"])
        # print()

        # flattening filter
        final_country_relevant, final_dc_emb, flatten_concepts = filter_with_other_countries(row, country_relevant, selected_dc_emb, country_map, region_to_countries, sentence_transformer, tokenizer)


        if row["Gold Answer"][-1] in final_country_relevant:
            has_answer += 1
        
        if final_country_relevant == []:
            zero_count += 1

        # print(row["Decoded Concepts"])
        # print(decoded_concepts)
        # print(row["Gold Answer"])
        # print(sim_scores)
        # print(country_relevant)
        # print(final_country_relevant)
        
        rel_score = []

        if final_dc_emb == []:
            country_specific.append(final_country_relevant)
            num_generated_concepts.append(country_relevant)
            scores.append(0.0)
            continue

        # relevance score with gold answers
        # if row["Gold Answer"] == []:

        country_specific.append(final_country_relevant)
        num_generated_concepts.append(country_relevant)
        scores.append(0.0)
            # continue

        # ga_emb_list = []
        # for ga in row["Gold Answer"]:
        #     encoded_dc = tokenizer(ga, return_tensors="pt").to(args.device)
        #     dc_outputs = sentence_transformer(**encoded_dc)

        #     dc_emb = mean_pooling(dc_outputs, encoded_dc["attention_mask"])
        #     ga_emb = F.normalize(dc_emb, p=2, dim=1)
        #     ga_emb_list.append(ga_emb.squeeze(0))

        # ga_emb_stacked = torch.stack(ga_emb_list)

        # for dc_emb in final_dc_emb:
        #     sim_scores = F.cosine_similarity(ga_emb_stacked, dc_emb.unsqueeze(0))
        #     rel_score.append(sim_scores.max().item())

        # country_specific.append(final_country_relevant)
        # num_generated_concepts.append(country_relevant)

        # if rel_score == []:
            # scores.append([0.0])
        # else:
            # scores.append(rel_score)

    result["concepts_incl_flatten"] = num_generated_concepts
    result["country_specific"] = country_specific
    result["rel_score"] = scores

    print(zero_count, len(result))
    print(has_answer, len(result))

    save_filename = result_file_path.split("/")[-1]

    model_prefix = {"meta-llama/Meta-Llama-3.1-8B-Instruct": "llama", "CohereLabs/aya-expanse-8b": "aya",
                    "Qwen/Qwen2.5-7B-Instruct": "qwen"}[args.model_name]
    
    if args.dataset == "camel":
        result.to_csv("./camel_final/ver2_%s" % save_filename, index=False)
    else:
        result.to_csv("./result_blend_%s/%s" %(model_prefix, save_filename), index=False)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="blend")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--result_path", type=str, default="../lens_result/%s/short_answer/patchscope_result_*_pos_sum.csv")
    parser.add_argument("--mode", type=str, default="short_answer")
    
    parser.add_argument("--seed", type=int, default=10)
    

    args = parser.parse_args()
    print(args)

    model_prefix = {"meta-llama/Meta-Llama-3.1-8B-Instruct": "llama", "CohereLabs/aya-expanse-8b": "aya",
                    "Qwen/Qwen2.5-7B-Instruct": "qwen"}[args.model_name]

    result_path = args.result_path # %(model_prefix)
    file_list = glob.glob(result_path)

    print(file_list)
    
    for f in file_list:
        if args.dataset == "fmlama":
            save_filename = "patchscope_fmlama_result.csv"
        else:
            save_filename = f.split("/")[-1]

        # if os.path.exists("./result_blend_%s/%s" %(model_prefix, save_filename)):
        #     continue

        main(args, f)