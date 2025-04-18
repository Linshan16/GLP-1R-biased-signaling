import os
from tkinter.font import names

import numpy as np
import pandas as pd
from utils import load_config, filter_data, output, update_progress


def prepare_wt_ref(fit_para_df, all_info, out_path):
    ref_df = filter_data(fit_para_df, {"receptor":"WT","ligand":"glp1"})
    bad_records = [value for key, value in all_info.items() if "bad WT" in str(value.get("note", ""))]
    bad_records_df = pd.DataFrame(bad_records, columns=['date', 'type', 'plate', 'time'])
    # 在 ref_df 中找到与 bad_records 匹配的记录
    mask = ref_df.merge(bad_records_df, on=['date', 'type', 'plate', 'time'], how='left', indicator=True)
    # 从 ref_df 中删除这些匹配的记录
    ref_df = mask[mask['_merge'] != 'both'].drop(columns=['_merge'])
    # 尝试将 'mean' 列转换为浮点数，无法转换的值设置为 NaN
    ref_df["emax_mean"] = pd.to_numeric(ref_df["emax_mean"], errors='coerce')
    # 去除 'mean' 列不是浮点数的记录
    ref_df = ref_df.dropna(subset=["emax_mean"])
    if out_path is not None:
        ref_df.to_csv(out_path,index=None)
    return ref_df


def find_ref(info_dict, fit_para_df, ref_df):
    filter_dict = {key: value for key, value in info_dict.items() if key in ["date", "type", "plate", "time"]}
    select_data = filter_data(fit_para_df, filter_dict)
    ref_base = dict(zip(select_data["receptor"], select_data["base_mean"]))
    max_emax = np.max(select_data["emax_mean"])
    select_wt = filter_data(ref_df, filter_dict)
    if len(select_wt) > 0:
        ref_max = select_wt["emax_mean"][0]
    else:
        del filter_dict["plate"]
        select_wt = filter_data(ref_df, filter_dict)
        if len(select_wt) > 0:
            ref_max = select_wt["emax_mean"][0]
        else:
            ref_max = max_emax
    if np.isnan(ref_max):
        ref_max = 100
    return ref_max, ref_base


def normlize(file_name, root_dir, info_dict, ref_max, ref_base):
    aliquot = info_dict["aliquot"]
    sample_num = info_dict["sample_num"]
    mutation = info_dict["receptor"]
    data_arr = np.array(pd.read_csv(f"{root_dir}/uniform/{file_name}"))
    norm_data = data_arr.copy()
    for idx_1 in range(sample_num):
        sub_arr = data_arr[:, idx_1*aliquot:(idx_1+1)*aliquot]
        norm_arr = (sub_arr - ref_base[mutation[idx_1]]) / ref_max * 100
        norm_data[:, idx_1*aliquot:(idx_1+1)*aliquot] = norm_arr
    output(norm_data,info_dict,out_path=f"{root_dir}/norm")


def batch_norm(root_dir, fit_para_file = "first_fit.csv"):
    fit_para_df = pd.read_csv(f"{root_dir}/result/{fit_para_file}",dtype={"date":str,"plate":str})
    file_list = [f for f in os.listdir(f"{root_dir}/uniform") if f.endswith(".csv") and f[:8].isdigit()]
    all_info = load_config(f"{root_dir}/uniform/config.json")
    if os.path.exists(f"{root_dir}/result/wt_ref.csv"):
        ref_df = pd.read_csv(f"{root_dir}/result/wt_ref.csv")
    else:
        ref_df =  prepare_wt_ref(fit_para_df, all_info , out_path=f"{root_dir}/result/wt_ref.csv")
    total = len(file_list)
    idx_2 = 0
    for file_name in file_list:
        info_dict = all_info[file_name]
        ref_max, ref_base = find_ref(info_dict, fit_para_df, ref_df)
        normlize(file_name, root_dir, info_dict, ref_max, ref_base)
        idx_2 = idx_2 + 1
        update_progress(idx_2, total, "norm: ")


if __name__ == "__main__":
    root_dir = "data"
    batch_norm(root_dir)