import os
import pandas as pd
import torch
from esm import FastaBatchedDataset, pretrained
import requests
import csv
import numpy as np


def prep_dir(receptor, variant):
    receptor = receptor.upper()
    variant = variant.upper()
    if not os.path.exists(f"protein/{receptor}"):
        os.mkdir(f"protein/{receptor}")
    if not os.path.exists(f"protein/{receptor}/{variant}"):
        os.mkdir(f"protein/{receptor}/{variant}")
        print(f"prepare directory: data/{receptor}/{variant}")


def get_fasta(receptor):
    receptor = receptor.upper()
    if not os.path.exists(f"protein/{receptor}/WT/input.fasta"):
        print(f"retrieving fatsa for {receptor}_HUMAN")
        url = "https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28%28id%3A"+receptor.lower()+"_human%29%20OR%20%28gene%3A"+receptor.lower()+"%29%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28organism_id%3A9606%29"
        fasta = requests.get(url).text
        if fasta == "":
            open("temp/fetch_fasta_fail.txt", 'w').write(receptor+"\n")
            print(receptor + " query get no result")
        else:
            prep_dir(receptor, "WT")
            fasta = fasta.split(">")[1]
            uniprot_id = fasta.split(" ")[0].split("|")[1]
            seq = "".join(fasta.split("\n")[1:-1])
            input = f">{receptor}_WT\n{seq}\n"
            open(f"protein/{receptor}/WT/input.fasta", 'w').write(input)


def prep_mutant_fasta(receptor, variant):
    if not os.path.exists(f"protein/{receptor.upper()}/WT/input.fasta"):
        get_fasta(receptor)
    file = open(f"protein/{receptor.upper()}/WT/input.fasta")
    lines = file.readlines()
    file.close()
    seq = lines[1].rstrip()
    mut_pos = int(variant[1:-1])-1
    mut_from = variant[0]
    if seq[mut_pos] == mut_from:
        mut_to = variant[-1]
        seq = seq[:mut_pos] + mut_to + seq[mut_pos+1:]
        prep_dir(receptor, variant)
        input = f">{receptor}_{variant}\n{seq}\n"
        open(f"protein/{receptor}/{variant}/input.fasta", 'w').write(input)
        print(f"{receptor}_{variant} complete")
    else:
        print(f"receptor {receptor} or mutation {variant} is incorrect")


def deep_mut(receptor):
    receptor = receptor.upper()
    with open(f"protein/{receptor}/WT/input.fasta", "r") as input_file:
        lines = input_file.readlines()
    seq = lines[1].rstrip()
    for idx in range(len(seq)):
        for mut_to in ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]:
            if not seq[idx] == mut_to:
                variant = seq[idx] + str(idx+1) + mut_to
                # if not os.path.exists(f"protein/{receptor}/{variant}/input.fasta"):
                variant_seq = seq[:idx] + mut_to + seq[idx+1:]
                prep_dir(receptor, variant)
                input = f">{receptor}_{variant}\n{variant_seq}\n"
                open(f"protein/{receptor}/{variant}/input.fasta", 'w').write(input)
        print(seq[idx] + str(idx+1), " complete")


def check_missing_embed(variants_list, model_name="esm2"):
    print(f"checking missing embed of {len(variants_list)} variants")
    wt_embed = {}
    def get_wt_embed(receptor, model_name="esm2"):
        nonlocal wt_embed
        wt_embed[receptor] = torch.load(f"protein/{receptor}/WT/{model_name}_{receptor}_WT.pt")["mean_representations"][33]
    missing_list = []
    for variant in variants_list:
        receptor, mut = variant.upper().split("_")
        # print(f"start {receptor}_{mut}")
        if receptor not in wt_embed.keys():
            get_wt_embed(receptor, model_name)
        if not mut == "WT":
            if not os.path.exists(f"protein/{receptor}/{mut}/{model_name}_{receptor}_{mut}.pt"):
                missing_list.append(variant)
                print(f"{variant}'s embed is not available")
            else:
                embed = torch.load(f"protein/{receptor}/{mut}/{model_name}_{receptor}_{mut}.pt")["mean_representations"][33]
                if (embed == wt_embed[receptor]).all():
                    missing_list.append(variant)
                    print(f"{variant}'s embed is identical to WT")
                # else:
                    # print(f"{variant} is available")
    print(f"there are {len(missing_list)} missing embed")
    return missing_list


def prep_fasta(variants_list, output_path):
    print("fetching fasta sequences")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if os.path.exists(f"{output_path}/extract.fasta"):
        os.remove(f"{output_path}/extract.fasta")
    total = len(variants_list)
    for idx, variant in enumerate(variants_list):
        receptor, mut = variant.upper().split("_")
        if not os.path.exists(f"protein/{receptor}/{mut}/input.fasta"):
            prep_mutant_fasta(receptor, variant)
        with open(f"protein/{receptor}/{mut}/input.fasta", "r") as input_file:
            lines = input_file.readlines()
        with open(f"{output_path}/extract.fasta", "a") as output_file:
            output_file.writelines(lines)
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"{idx+1}/{total} done")
    return f"{output_path}/extract.fasta"


def extract(input_file, repr_layer=[i for i in range(34)], full_model_name="esm2_t33_650M_UR50D"):
    if os.path.exists(f"esm_pretrained/{full_model_name}.pt"):
        model_location = f"esm_pretrained/{full_model_name}.pt"
    else:
        model_location = full_model_name
    toks_per_batch = 4096
    short_model_name = full_model_name.split("_")[0]
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    dataset = FastaBatchedDataset.from_file(input_file)
    print("read fasta file")
    batches = dataset.get_batch_indices(int(toks_per_batch), extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
    total = len(data_loader)
    print(f"total {total} batches in dataloader")
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layer)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layer]
    with torch.no_grad():
        for batch_idx, (variants, strs, toks) in enumerate(data_loader):
            # print(f"start batch {batch_idx+1}/{total}")
            out = model(toks, repr_layers=repr_layers, need_head_weights=True, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            for i, variant in enumerate(variants):
                receptor = variant.split("_")[0]
                mut = variant.split("_")[1]
                embed_file = f"protein/{receptor}/{mut}/{short_model_name}_{receptor}_{mut}.pt"
                result = {"variant": variant}
                result["representations"] = {layer: t[i, 1: len(strs[i]) + 1].clone() for layer, t in representations.items()}
                result["mean_representations"] = {layer: t[i, 1: len(strs[i]) + 1].mean(0).clone() for layer, t in representations.items()}
                torch.save(result, embed_file)
                # print(f"{receptor}_{mut} done, variant idx: {i}")
            del out, representations
            print(f"finished batch {batch_idx+1}/{total}")


def get_embed(variants_list, output_path, layer=33, model_name="esm2", embed_type="avg"):
    buf = torch.Tensor([])
    total = len(variants_list)
    print(f"gathering embed, {total} lines in total")
    for idx, variant in enumerate(variants_list):
        receptor, mut = variant.upper().split("_")
        if embed_type == "avg":
            embed = torch.load(f"protein/{receptor}/{mut}/{model_name}_{receptor}_{mut}.pt")["mean_representations"][layer]
        elif embed_type == "layer":
            embed = torch.load(f"protein/{receptor}/{mut}/{model_name}_{receptor}_{mut}.pt")["representations"][layer]
        buf = torch.stack([*buf, embed])
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"{idx+1}/{total} done")
    # pd.DataFrame(buf).to_csv(f"{output_path}/{model_name}_embed.csv", index=None, header=None)
    torch.save(buf, f"{output_path}/{model_name}_embed.pt")
    return buf






