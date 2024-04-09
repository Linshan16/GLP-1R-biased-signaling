import os
import pandas as pd
import numpy as np

def csv2array_bret(data_dir):
    mut_info_file = "export/mut_info.csv"
    mut_info_df = pd.read_csv(mut_info_file)
    mut_info_df = mut_info_df[mut_info_df["type"] == "bret"]
    mut_info_df["date"] = mut_info_df["date"].astype(str)
    mut_info_df["batch"] = mut_info_df["batch"].astype(str)
    all_data = pd.DataFrame(columns=range(96))
    info_sheet = pd.DataFrame(columns=range(14))
    folder_list = [f for f in os.listdir(f"{data_dir}") if os.path.isdir(f"{data_dir}/{f}") and not f.endswith("export")]

    for folder in folder_list:
        date_str = folder[:8]
        batch_num = folder[-1:]
        info_idx = mut_info_df[(mut_info_df["date"] == date_str) & (mut_info_df["batch"] == batch_num)].index[0]
        info_row = mut_info_df.loc[info_idx]
        csv_files = [f for f in os.listdir(f"{data_dir}/{folder}") if f.endswith('.xlsx')]
        if not os.path.exists(f"{data_dir}/{folder}/sub_base"):
            os.mkdir(f"{data_dir}/{folder}/sub_base")

        # get base value
        for base_file in [f for f in csv_files if "BASE" in f]:
            data_df = pd.read_excel(f"{data_dir}/{folder}/{base_file}", header=None, skiprows=[0])
            a1_coord = data_df[data_df[0] == "A"].index[0]
            if "B1" in base_file:
                b1_base = np.array(data_df.iloc[a1_coord:a1_coord + 8, 1:13]).astype(np.float32)
            if "B2" in base_file:
                b2_base = np.array(data_df.iloc[a1_coord:a1_coord + 8, 1:13]).astype(np.float32)

        for filename in [f for f in csv_files if "BASE" not in f]:
            data_df = pd.read_excel(f"{data_dir}/{folder}/{filename}", header=None, skiprows=[0])
            a1_coord = data_df[data_df[1] == "A1"].index[2]
            num_of_read = data_df[data_df[1] == "A1"].index[3]-data_df[data_df[1] == "A1"].index[2]-3
            data_df = data_df.loc[a1_coord+1:a1_coord+num_of_read]
            for idx, row in data_df.iterrows():
                time_str = str(round(row[0]/60, 1))
                data_array = np.array(row[1:]).reshape(12, 8).transpose()
                if "B1" in filename:
                    data_array = (data_array - b1_base).reshape(1, 96)
                    assat_type = "bret_b1"
                if "B2" in filename:
                    data_array = (data_array - b2_base).reshape(1, 96)
                    assat_type = "bret_b2"
                pd.DataFrame(data_array.reshape(8, 12)).to_csv(f"{data_dir}/{folder}/sub_base/{date_str}-{batch_num} {assat_type} GLP1 {time_str}MIN sub_base.csv", header=None, index=None)
                # organize info of this plate
                plate_info = [info_row["date"], assat_type, info_row["batch"], info_row["ligand"], time_str, info_row["mut_1"], info_row["mut_2"], info_row["mut_3"], info_row["mut_4"], info_row["mut_5"], info_row["mut_6"], info_row["mut_7"], info_row["mut_8"]]

                # collect info and data
                all_data = all_data.append(data_array.tolist(), ignore_index=True)
                info_sheet = info_sheet.append([plate_info], ignore_index=True)

        if 'b1_base' in locals():
            del b1_base
        else:
            print("no b1_base")
        if 'b2_base' in locals():
            del b2_base
        else:
            print("no b2_base")
        print("finished file: ", folder)

    info_sheet.columns = ["date", "assay_type", "batch", "ligand", "time", "mut1", "mut2", "mut3", "mut4", "mut5", "mut6", "mut7", "mut8", "note"]
    all_data.to_csv(f"{data_dir}/export/{data_dir}_data.csv", index=None, header=None)
    info_sheet.to_csv(f"{data_dir}/export/{data_dir}_info.csv", index=None)


def main():

    csv2array_bret("bret_data")


if __name__ == "__main__":
    main()
