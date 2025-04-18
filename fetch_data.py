import os
import pandas as pd
import numpy as np
from utils import filter_data, update_progress


def get_diverse_col(data_df):
    unique_dict = {}
    diverse_col = []
    columns_list = list(data_df.columns)
    columns_list.remove("values")
    for col in columns_list:
        col_value = data_df[col].unique()
        if len(col_value) == 1:
            unique_dict[col] = col_value[0]
        else:
            diverse_col.append(col)
    return unique_dict, diverse_col


def reshape_data(data_df, row, column, output_path="data/export"):
    """
    输入：intergrat_data输出的数据表，标注了数据的相关参数，经过filter_data筛选需要的数据，其中两列参数作为变量（需要其他参数列为常量）
    输出：将两列变量参数指定为row和column，将数据表values中的数据整理为二维数组
    """
    unique_dict, diverse_col = get_diverse_col(data_df)
    assert len(diverse_col) < 3 , f"{diverse_col},变量数大于2"
    result_data = {row:[]}
    for row_value in data_df[row].unique():
        result_data[row].append(row_value)
        sub_data = data_df[data_df[row] == row_value]
        for _,line in sub_data.iterrows():
            values_list = line["values"].split(";")
            for i, value in enumerate(values_list):
                col_id = f"{line[column]}_{i}"
                if col_id not in result_data.keys():
                    result_data[col_id] = [value]
                else:
                    result_data[col_id].append(value)
    result_df = pd.DataFrame(result_data)
    output_file = "_".join(list(unique_dict.values())) + f"_{row}_{column}.csv"
    if output_path is not None:
        result_df.to_csv(f"{output_path}/{output_file}", index=None)
    return unique_dict, result_df


def batch_reshape(data_df, row=None, column=None, single_col=False, output_path="data/export"):
    """
    输入：intergrat_data输出的数据表，经过filter_data筛选需要的数据
    1）解析dataframe中除values之外的列名，选择两列作为row,column
    2）将其他变量列组合，并遍历它们组合的unique值，作为filter_dict
    3）提取sub_dataframe，并进行reshape
    """
    reshaped_list = []
    col_list = list(data_df.columns)
    col_list.remove("values")
    if row is None:
        row = input(f"请从{col_list}中选一个作为row：")
    col_list.remove(row)
    if column is None:
        column = input(f"请从{col_list}中选一个作为column：")
    if not single_col:
        col_list.remove(column)
    unique_combinations = data_df.groupby(col_list).first().reset_index()
    total = len(unique_combinations)
    for idx_1,uni_comb in unique_combinations.iterrows():
        filter_dict = {c: uni_comb[c] for c in col_list}
        sub_data = filter_data(data_df, filter_dict)
        result_tuple = reshape_data(sub_data, row, column, output_path)
        reshaped_list.append(result_tuple)
        update_progress(len(reshaped_list),total,"reshaping: ")
    return reshaped_list


if __name__ == "__main__":
    data_path = "data/uniform/all_data.csv"
    data_df = pd.read_csv(data_path, dtype=str)
    filter_dict = {"date":"20250405","plate":"1"}
    data_df = filter_data(data_df, filter_dict)
    reshape_list = batch_reshape(data_df,"conc","receptor",single_col=True)