import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from Model import RefinementLayers, DynamicFC
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import check_missing_embed, prep_fasta, extract, get_embed
import datetime
import os


def prep_dataset(work_dir, use_exp_data="expression", train_ratio=None, default_receptor="GLP1R", batch_size=1, model_name="esm2"):
    data_df = pd.read_csv(f"{work_dir}/exp_data.csv")
    if os.path.exists(f"{work_dir}/{model_name}_embed.pt"):
        embed_tensor = torch.load(f"{work_dir}/{model_name}_embed.pt")
    else:
        if "receptor" not in data_df.columns:# only mut info no receptor in data table
            data_df["variant"] = f"{default_receptor}_" + data_df["mut"]
        else:
            data_df["variant"] = data_df["receptor"] + "_" + data_df["mut"]
        variants_list = data_df["variant"].tolist()
        unique_variants_list = data_df["variant"].unique().tolist()
        missing_list = check_missing_embed(unique_variants_list)
        if not len(missing_list) == 0:
            fasta_file = prep_fasta(variants_list=missing_list, output_path=work_dir)
            extract(input_file=fasta_file)
        embed_tensor = get_embed(variants_list=variants_list, output_path=work_dir, embed_type="layer", model_name=model_name)
    labels = data_df[use_exp_data].values  # Replace 'label_column' with the actual column name
    # embed_tensor = torch.tensor(embed, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Adjust dtype accordingly
    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    if train_ratio is not None:
        # Split the dataset into training and testing sets
        num_train = int(labels.shape[0] * train_ratio)
        train_input = embed_tensor[:num_train]
        train_labels = labels_tensor[:num_train]
        test_input = embed_tensor[num_train:]
        test_labels = labels_tensor[num_train:]
        # Create DataLoader for training and testing sets
        train_dataset = TensorDataset(train_input, train_labels)
        test_dataset = TensorDataset(test_input, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=pin_memory)
    else:
        train_dataset = TensorDataset(embed_tensor, labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        test_loader = None
    return train_loader, test_loader


def single_test(train_loader, test_loader=None, output_path=None, embed_dim=256, num_layers=1, attention_heads=8, target_type="value", num_epochs=400, learning_rate=0.00001, use_cuda=True, model_type="tf", fc_list=[256]):
    if torch.cuda.is_available() and use_cuda:
        device = torch.device("cuda")
        non_blocking = True
    else:
        device = torch.device("cpu")
        non_blocking = False
    result = {}
    input_dim = train_loader.dataset.tensors[0].shape[2]
    if len(train_loader.dataset.tensors[1].shape) == 1:
        output_dim = 1
    else:
        output_dim = train_loader.dataset.tensors[1].shape[1]
    if model_type == "tf":
        model = RefinementLayers(input_dim, output_dim, num_layers, embed_dim, attention_heads)
    elif model_type == "fc":
        model = DynamicFC(input_dim, output_dim, fc_list)
    model = model.to(device)
    train_loss = []
    if target_type == "value":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
    elif target_type == "class":
        criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=num_epochs, gamma=0.9822)
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            out = model(inputs)["output"]
            loss = criterion(out.view(-1), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
        if (epoch + 1) % 50 == 0 or epoch + 1 == num_epochs:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    result["train_loss"] = train_loss
    test_predict = {"predict": [], "target": []}
    if not test_loader is None:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, target in test_loader:
                predict = model(inputs)
                test_predict["predict"].append(predict)
                test_predict["target"].append(target)
                test_loss += criterion(predict.view(-1), target.view(-1)).item()
        avg_test_loss = test_loss / len(test_loader)
    else:
        avg_test_loss = None
    result["test_predict"] = test_predict
    result["train_para"] = {"embed_dim": embed_dim, "num_layers": num_layers, "attention_heads": attention_heads, "num_epochs": num_epochs, "learning_rate": learning_rate, "avg_test_loss":avg_test_loss, "fc_list": "-".join([str(i) for i in fc_list])}
    del model
    return result


def save_result(result, output_path, postfix):
    plt_filename = f"{output_path}/loss_curve_{postfix}.png"
    train_loss = result["train_loss"]
    fig, axs = plt.subplots(ncols=1)
    axs[0].plot(range(len(train_loss)), train_loss, label="train loss")
    axs[0].set_xlabel('step')
    axs[0].set_ylabel('value')
    axs[0].legend()
    plt.savefig(plt_filename)
    plt.close(fig)
    del (fig, axs)


def test_architecture(work_dir, test_para_name, test_para_list, train_loader, test_loader=None, repeat=1):
    embed_dim = 512
    num_layers = 1
    attention_heads = 4
    num_epochs = 1280
    learning_rate = 0.01
    fc_list = [512]
    for value in test_para_list:
        if test_para_name == "embed_dim":
            embed_dim = value
            model_type = "tf"
        elif test_para_name == "num_layers":
            num_layers = value
            model_type = "tf"
        elif test_para_name == "attention_heads":
            attention_heads = value
            model_type = "tf"
        elif test_para_name == "num_epochs":
            num_epochs = value
            model_type = "tf"
        elif test_para_name == "learning_rate":
            learning_rate = value
            model_type = "tf"
        elif test_para_name == "fc_list":
            fc_list = value
            model_type = "fc"
        print(f"start test: repeat {repeat}, {test_para_name} {value}")
        start_time = datetime.datetime.now()
        result = single_test(train_loader=train_loader, test_loader=test_loader, embed_dim=embed_dim, num_layers=num_layers, attention_heads=attention_heads, num_epochs=num_epochs, learning_rate=learning_rate, model_type=model_type, fc_list=fc_list)
        end_time = datetime.datetime.now()
        print(f"time used: {str(end_time - start_time)}")
        if isinstance(value, list):
            value = "-".join([str(i) for i in value])
        postfix = f"{test_para_name}_{value}_repeat{repeat}"
        output_path = f"{work_dir}/{test_para_name}"
        if not os.path.exists(f"{output_path}"):
            os.mkdir(f"{output_path}")
        result = save_result(result, output_path, postfix)
        result.update({"repeat": repeat, "train_time": str(end_time - start_time), "time_stamp": str(end_time)[:19]})
        if not os.path.exists(f"{work_dir}/test_batch_result.csv"):
            result_df = pd.DataFrame(columns=["repeat", "embed_dim", "num_layers", "attention_heads", "fc_list", "num_epochs", "learning_rate", "avg_test_loss", "train_time", "time_stamp"])
        else:
            result_df = pd.read_csv(f"{work_dir}/test_batch_result.csv")
        result_df = pd.concat([result_df, pd.DataFrame(result, index=[0])], ignore_index=True)
        result_df.to_csv(f"{work_dir}/test_batch_result.csv", index=None)
