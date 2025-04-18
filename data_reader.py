import os
import pandas as pd
import numpy as np
from utils import update_progress, filter_data, find_a1_loc, data_parser, output


def sort_plate_info(date,assay_type,plate,info_path="info/mut_info.csv",ligand_gradients="info/ligand_gradient.csv",plate_sz_dict={"glo":384,"bret":96}):
    info_df = pd.read_csv(info_path,dtype=str)
    gradients_df = pd.read_csv(ligand_gradients)
    select_info = filter_data(info_df,{"date":date,"type":assay_type,"plate":plate})
    assert len(select_info) > 0, f"没有找到{date}-{plate}记录"
    assert len(select_info) < 2, f"找到多条{date}-{plate}记录"
    row = select_info.iloc[0]
    sample_num = int(row[6:].count())
    plate_size = plate_sz_dict[assay_type]
    col_num = int(np.sqrt(plate_size/6)*3)
    row_num = int(np.sqrt(plate_size/6)*2)
    aliquot = int(col_num/sample_num)
    sample_column = row[range(6,6+sample_num)]
    gradient = list(gradients_df[row["ligand_gradient"]])[:row_num]
    mut_list = []
    ligand_list = []
    for i,sample in enumerate(sample_column):
        if "-" in sample:
            mut_list.append(sample.split("-")[0])
            ligand_list.append(sample.split("-")[1])
        else:
            mut_list.append(sample)
    if len(ligand_list) == 0:
        ligand_list.append(row[3])
    plate_info = {"date":date,"type":assay_type,"plate":plate,"receptor":mut_list,"ligand":ligand_list,"ligand_gradient":gradient,"aliquot": aliquot, "sample_num": sample_num, "plate_size":plate_size, "col_num":col_num, "row_num":row_num, "note":row["note"]}
    return plate_info


def csv2array_glo(data_dir, root_dir="data/glo", out_path="data/uniform", first_id=("A","01")):
    date = data_dir[:8]
    file_list = [f for f in os.listdir(f"{root_dir}/{data_dir}") if f.endswith('.csv')]
    total = len(file_list)
    idx_2 = 0
    for file_name in file_list:
        if "MIN" in file_name.upper():
            time_point = float(file_name[:-4].split("-")[-1][:-3])
        else:
            time_point = 30.0
        data_df = pd.read_csv(f"{root_dir}/{data_dir}/{file_name}", names=range(30), dtype=str)
        _, _, unique_array = find_a1_loc(data_df, first_id[0], first_id[1])
        for idx_1, data_array in enumerate(unique_array):
            if "-1-" in file_name:
                plate_name = "1"
            elif "-2-" in file_name:
                plate_name = "2"
            elif "-3-" in file_name:
                plate_name = "3"
            else:
                plate_name = f"{idx_1+1}"
            info_dict = sort_plate_info(date, "glo", plate_name)
            info_dict["time"] = time_point
            info_dict["type"] = "cAMP"
            data_array = data_array.reshape(info_dict["row_num"], info_dict["col_num"])
            output(data_array, info_dict, out_path)
            idx_2 = idx_2 + 1
            update_progress(idx_2,total,"glo: ")


def csv2array_bret_e(data_dir, root_dir="data/bret_e", out_path="data/uniform"):
    date = data_dir.split("-")[0]
    plate_name = data_dir.split("-")[1]
    info_dict = sort_plate_info(date, "bret", plate_name)
    file_list = [f for f in os.listdir(f"{root_dir}/{data_dir}") if f.endswith('.csv')]
    total = len(file_list)
    idx_2 = 0
    # get base value
    for base_file in [f for f in file_list if "BASE" in f]:
        data_df = pd.read_csv(f"{root_dir}/{data_dir}/{base_file}", names=range(30), dtype=str)
        _, _, unique_array = find_a1_loc(data_df)
        if "&" in base_file:
            b1_base = unique_array[1] / unique_array[0]
            b2_base = unique_array[3] / unique_array[2]
        elif "B1" in base_file:
            b1_base = unique_array[1] / unique_array[0]
        elif "B2" in base_file:
            b2_base = unique_array[1] / unique_array[0]
    for filename in [f for f in file_list if "BASE" not in f]:
        data_df = pd.read_csv(f"{root_dir}/{data_dir}/{filename}", names=range(30), dtype=str)
        time_list = filename[:len(filename) - 7].split()[5].split("-")
        _, _, unique_array = find_a1_loc(data_df)
        num_of_read = int((float(time_list[1]) - float(time_list[0])) / 2.5)
        for idx_1 in range(num_of_read):
            time_point = float(time_list[0]) + idx_1 * 2.5
            data_array = unique_array[2*idx_1+1] / unique_array[2*idx_1]
            if "B1" in filename:
                data_array = data_array - b1_base
                assay_type = "Arr1"
            if "B2" in filename:
                data_array = data_array - b2_base
                assay_type = "Arr2"
            info_dict = sort_plate_info(date, "bret", plate_name)
            info_dict["time"] = time_point
            info_dict["type"] = assay_type
            data_array = data_array.reshape(info_dict["row_num"], info_dict["col_num"])
            output(data_array, info_dict, out_path)
            idx_2 = idx_2 + 1
            update_progress(idx_2,total,"bret_e: ")
    if 'b1_base' in locals():
        del b1_base
    else:
        print("no b1_base")
    if 'b2_base' in locals():
        del b2_base
    else:
        print("no b2_base")


def csv2array_bret_b(data_dir, root_dir="data/bret_b", out_path="data/uniform"):
    a1_loc = [44, 1]
    date = data_dir[:8]
    plate_name = data_dir[-1]
    file_list = [f for f in os.listdir(f"{root_dir}/{data_dir}") if f.endswith('.xlsx')]
    total = len(file_list)
    idx_2 = 0
    # get base value
    for base_file in [f for f in file_list if "BASE" in f]:
        data_df = pd.read_excel(f"{root_dir}/{data_dir}/{base_file}", skiprows=range(4))
        data_df.columns = range(len(data_df.columns))
        a1_loc = [data_df[data_df[0] == "A"].index[1], 1]
        if "B1" in base_file:
            b1_base = data_parser(data_df, 96, a1_loc)
        if "B2" in base_file:
            b2_base = data_parser(data_df, 96, a1_loc)
    # subtract base from signal
    for filename in [f for f in file_list if not "BASE" in f]:
        data_df = pd.read_excel(f"{root_dir}/{data_dir}/{filename}", skiprows=range(4))
        data_df.columns = range(len(data_df.columns))
        a1_loc = [data_df[data_df[0] == "A"].index[1], 1]
        data_array = data_parser(data_df, 96, a1_loc)
        if "B1" in filename:
            data_array = data_array - b1_base
            assay_type = "Arr1"
        if "B2" in filename:
            data_array = data_array - b2_base
            assay_type = "Arr2"
        time_point = float(filename[len(filename)-14:len(filename)-8].split()[1])
        info_dict = sort_plate_info(date, "bret", plate_name)
        info_dict["time"] = time_point
        info_dict["type"] = assay_type
        data_array = data_array.reshape(info_dict["row_num"], info_dict["col_num"])
        output(data_array, info_dict, out_path)
        idx_2 = idx_2 + 1
        update_progress(idx_2, total, "bret_b: ")
    if 'b1_base' in locals():
        del b1_base
    else:
        print("no b1_base")
    if 'b2_base' in locals():
        del b2_base
    else:
        print("no b2_base")


def csv2array_bret_k(data_dir, root_dir="data/bret_k", out_path="data/uniform"):
    date = data_dir[:8]
    plate_name = data_dir[-1]
    file_list = [f for f in os.listdir(f"{root_dir}/{data_dir}") if f.endswith('.xlsx')]
    total = len(file_list)
    idx_2 = 0
    # get base value
    for base_file in [f for f in file_list if "BASE" in f]:
        data_df = pd.read_excel(f"{root_dir}/{data_dir}/{base_file}")
        data_df.columns = range(len(data_df.columns))
        a1_loc = [data_df[data_df[0] == "A"].index[0], 1]
        if "B1" in base_file:
            b1_base = data_parser(data_df, 96, a1_loc)
        if "B2" in base_file:
            b2_base = data_parser(data_df, 96, a1_loc)
    for filename in [f for f in file_list if "BASE" not in f]:
        data_df = pd.read_excel(f"{root_dir}/{data_dir}/{filename}")
        data_df.columns = range(len(data_df.columns))
        a1_list = list(data_df[data_df[1] == "A1"].index)
        a1_loc = a1_list[2]
        # time_list = filename[:len(filename) - 8].split()[-1].split("-")
        num_of_read = a1_list[3]-a1_list[2]-3
        data_df = data_df.loc[a1_loc+1:a1_loc+num_of_read]
        for idx, row in data_df.iterrows():
            time_point = round(row[0]/60, 1)
            data_array = np.array(row[1:]).reshape(12, 8).transpose()
            if "B1" in filename:
                data_array = data_array.reshape(1, 96) - b1_base
                assay_type = "Arr1"
            if "B2" in filename:
                data_array = data_array.reshape(1, 96) - b2_base
                assay_type = "Arr2"
            info_dict = sort_plate_info(date, "bret", plate_name)
            info_dict["time"] = time_point
            info_dict["type"] = assay_type
            data_array = data_array.reshape(info_dict["row_num"], info_dict["col_num"])
            output(data_array, info_dict, out_path)
            idx_2 = idx_2 + 1
            update_progress(idx_2, total, "bret_k: ")
    if 'b1_base' in locals():
        del b1_base
    else:
        print("no b1_base")
    if 'b2_base' in locals():
        del b2_base
    else:
        print("no b2_base")


def batch_reader(root_dir, reader):
    folder_list = [f for f in os.listdir(root_dir) if os.path.isdir(f"{root_dir}/{f}") and f[:8].isdigit()]
    for folder in folder_list:
        if reader == "glo":
            csv2array_glo(folder)
        elif reader == "bret_e":
            csv2array_bret_e(folder)
        elif reader == "bret_b":
            csv2array_bret_b(folder)
        elif reader == "bret_k":
            csv2array_bret_k(folder)


def single_reader(data_dir,out_path,reader,first_id=("A","01")):
    if reader == "glo":
        csv2array_glo(data_dir=data_dir,out_path=out_path,first_id=first_id)
    elif reader == "bret_e":
        csv2array_bret_e(data_dir=data_dir,out_path=out_path)
    elif reader == "bret_b":
        csv2array_bret_b(data_dir=data_dir,out_path=out_path)
    elif reader == "bret_k":
        csv2array_bret_k(data_dir=data_dir,out_path=out_path)
