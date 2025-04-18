import os
import pandas as pd
from utils import update_config, load_config
from reader3 import single_reader
from intergration import integrate_data
from fitting4 import batch_fit
from normalize import batch_norm

def pipline(data_dir,reader,first_id=("A","01")):
    path = f"data/{reader}/{data_dir}"
    if not os.path.exists(f"{path}/uniform"):
        os.mkdir(f"{path}/uniform")
    if not os.path.exists(f"{path}/result"):
        os.mkdir(f"{path}/result")
    if not os.path.exists(f"{path}/norm"):
        os.mkdir(f"{path}/norm")
    single_reader(data_dir, out_path=f"{path}/uniform", reader=reader, first_id=first_id)
    config_dict = load_config(f"{path}/uniform/config.json")
    update_config(config_dict,"data/uniform/config.json")
    data_df = integrate_data(f"{path}/uniform")
    concat_result(data_df, "data/uniform/all_data.csv")
    fit_para_df = batch_fit(data_df, out_path=f"{path}/result/first_fit.csv")
    concat_result(fit_para_df, "data/result/first_fit.csv")
    batch_norm(path)
    config_dict = load_config(f"{path}/norm/config.json")
    update_config(config_dict,"data/norm/config.json")
    ref_df = pd.read_csv(f"{path}/result/wt_ref.csv")
    concat_result(ref_df, "data/result/wt_ref.csv")
    data_df = integrate_data(f"{path}/norm")
    concat_result(data_df, "data/norm/all_data.csv")
    fit_para_df = batch_fit(data_df, out_path=f"{path}/result/norm_fit.csv")
    concat_result(fit_para_df, "data/result/norm_fit.csv")


def concat_result(new_df, old_file):
    new_df = new_df.astype(str)
    old_df = pd.read_csv(old_file,dtype=str)
    # 找出 new_df 中不存在于 old_df 中的记录
    merged_df = new_df.merge(old_df, on=list(old_df.columns), how='left', indicator=True)
    new_records = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    # 将这些新记录追加到 old_df 的末尾
    updated_df = pd.concat([old_df, new_records], ignore_index=True)
    updated_df.to_csv(old_file,index=None)


if __name__ == "__main__":
    pipline(data_dir="20250417", reader="glo", first_id=("A","1"))


