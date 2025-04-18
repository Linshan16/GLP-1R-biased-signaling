import os
import pandas as pd
from utils import load_config, data_parser, update_config, update_progress


def integrate_data(data_dir="data/uniform"):
    result_file = f"{data_dir}/all_data.csv"
    if os.path.exists(result_file):
        os.remove(result_file)
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    all_data = pd.DataFrame(columns=["date","type","plate","receptor","ligand","conc","time","values"])
    all_config = load_config(f"{data_dir}/config.json")
    total = len(file_list)
    idx_2 = 0
    for file_name in file_list:
        info_dict = all_config[file_name]
        data_df = pd.read_csv(f"{data_dir}/{file_name}")
        aliquot = info_dict["aliquot"]
        receptor_list = info_dict["receptor"]
        ligand_gradient = info_dict["ligand_gradient"]
        sample_num = info_dict["sample_num"]
        ligand_num = len(info_dict["ligand"])
        data_array = data_parser(data_df, info_dict["plate_size"], [0,0]).reshape(-1,aliquot)
        for idx, row in enumerate(data_array):
            receptor_id = divmod(idx,sample_num)[1]
            conc_id = divmod(idx,sample_num)[0]
            ligand_id = divmod(idx,ligand_num)[1]
            values = []
            for value in row:
                values.append(str(value))
            annotate_data = {"date": info_dict["date"],
                             "type": info_dict["type"],
                            "plate": info_dict["plate"],
                            "receptor": receptor_list[receptor_id],
                            "ligand":info_dict["ligand"][ligand_id],
                            "conc": str(ligand_gradient[conc_id]),
                            "time": str(info_dict["time"]),
                            "values": ";".join(values)}
            all_data = pd.concat([all_data,pd.DataFrame([annotate_data])],ignore_index=True)
        idx_2 = idx_2 +1
        update_progress(idx_2, total, "intergrate: ")
    all_data.to_csv(result_file,index=None)
    return all_data

def increment_intergration(data_dir="data/uniform_incre"):
    all_data = integrate_data(data_dir)
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    exist_data = pd.read_csv("data/uniform/all_data.csv")
    new_data = pd.concat([exist_data,all_data])
    new_data.to_csv("data/uniform/all_data.csv")
    config = load_config(f"{data_dir}/config.json")
    update_config(config,"data/uniform/config.json")


if __name__ == "__main__":
    # root_dir = "data/glo"
    # # folder_list = [f for f in os.listdir(root_dir) if os.path.isdir(f"{root_dir}/{f}") and f[:8].isdigit()]
    # folder_list = ["20250402","20250405","20250408"]
    # batch_config(folder_list)
    integrate_data("data/uniform")