import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def concat_data(file_list, export_path):
    export_df = pd.DataFrame()
    for file in file_list:
        data_df = pd.read_csv(file)
        export_df = pd.concat([export_df, data_df], ignore_index=True)
    export_df.to_csv(export_path, index=None)


def fit_dec(y_data, plate_size, fit_type="log"):
    x_384 = [0.000000000000075, 0.00000000000025, 0.00000000000075, 0.0000000000025, 0.0000000000075, 0.000000000025, 0.000000000075, 0.00000000025, 0.00000000075, 0.0000000025, 0.0000000075, 0.000000025, 0.000000075, 0.00000025, 0.00000075, 0.0000025]
    x_96 = [0.0000000000002083, 0.000000000002083, 0.00000000002083, 0.0000000002083, 0.000000002083, 0.00000002083, 0.0000002083, 0.000002083]
    if plate_size == 384:
        x_data = np.array([x_384, x_384, x_384]).transpose().flatten()
    if plate_size == 96:
        x_data = np.array([x_96, x_96]).transpose().flatten()

    if fit_type == "log":
        def func(log_x, log_ec50, emax, c):
            return emax / (1 + 10 ** (log_ec50 - log_x)) + c
        initial_guess = [-8.5, max(y_data) - min(y_data), min(y_data)]
        x_data = np.log10(x_data)

    if fit_type == "nonlog":
        def func(x, ec50, emax, c):
            return emax * x / (ec50 + x) + c
        initial_guess = [0.000000003, max(y_data) - min(y_data), min(y_data)]

    try:
        optimized_params, cov = curve_fit(func, x_data, y_data, p0=initial_guess, maxfev=2000)
        y_predicted = func(x_data, *optimized_params)  # Calculate predicted y-values from the model
        ssr = np.sum((y_data - y_predicted) ** 2)  # Calculate the sum of squared residuals (SSR)
        y_mean = np.mean(y_data)
        sst = np.sum((y_data - y_mean) ** 2)  # Calculate the total sum of squares (SST)
        r_sqr = np.round(1 - (ssr / sst), 3)  # Calculate R-squared


        if cov[0, 0] < 0 or cov[1, 1] < 0 or cov[2, 2] < 0:
            fit_para = {"date": "", "assay_type": "", "batch": "", "ligand": "", "mut": "", "time": "", "pec50_mean": "negetive covariance", "base_mean": 0}
        else:
            fit_para = {"date": "", "assay_type": "", "batch": "", "ligand": "", "mut": "", "time": "", "pec50_mean": -optimized_params[0], "pec50_err": np.sqrt(cov[0, 0]), "min_max": max(y_data)-min(y_data), "emax_mean": optimized_params[1], "emax_err": np.sqrt(cov[1, 1]), "base_mean": optimized_params[2], "base_err": np.sqrt(cov[2, 2]), "r_sqr": r_sqr}

    except RuntimeError as e:
        fit_para = {"date": "", "assay_type": "", "batch": "", "ligand": "", "mut": "", "time": "", "pec50_mean": str(e), "base_mean": 0}

    return fit_para


def batch_fit(data_dir, normed=False):
    if normed:
        read_postfix = "data_normed"
        export_postfix = "norm_fit"
    else:
        read_postfix = "data"
        export_postfix = "first_fit"
    data_file = f"{data_dir}/export/{data_dir}_{read_postfix}.csv"
    info_file = f"{data_dir}/export/{data_dir}_info.csv"
    print("start: ", f"{data_dir}/export/{data_dir}_info.csv")
    data_df = pd.read_csv(data_file, header=None)
    info_df = pd.read_csv(info_file)
    info_df["date"] = info_df["date"].astype(str)
    info_df["batch"] = info_df["batch"].astype(str)
    result_df = pd.DataFrame()
    norm_data_df = pd.DataFrame()

    for idx, row in info_df.iterrows():
        batch_result_df = pd.DataFrame()
        if row["assay_type"].startswith("glo"):
            mut_num = 8
            aliquot = 3
            plate_col = 24
            plate_row = 16
        elif row["assay_type"].startswith("bret"):
            mut_num = 6
            aliquot = 2
            plate_col = 12
            plate_row = 8
        norm_data_array = np.zeros([plate_row, plate_col])
        data_array = np.array(data_df.loc[idx]).reshape(plate_row, plate_col)
        for i in range(mut_num):
            print(row["date"], row["assay_type"], row["batch"], row["ligand"], row["time"], row[i+5])
            fit_data = data_array[:, i*aliquot:(i+1)*aliquot].copy()
            fit_para = fit_dec(fit_data.flatten(), plate_row*plate_col)
            fit_para["date"], fit_para["assay_type"], fit_para["batch"], fit_para["ligand"], fit_para["time"] = row[0:5]
            fit_para["mut"] = row[i + 5]
            batch_result_df = batch_result_df.append(fit_para, ignore_index=True)
            if row[i + 5] == "WT":
                ref_emax = fit_para["emax_mean"]
                ref_pec50 = fit_para["pec50_mean"]
            fit_data = fit_data - fit_para["base_mean"]
            norm_data_array[:, i * aliquot:(i + 1) * aliquot] = fit_data

        if 'ref_emax' not in locals():
            ref_emax = result_df[(result_df["date"] == row["date"]) & (result_df["mut"] == "WT")]["emax_mean"].values[0]
            ref_pec50 = result_df[(result_df["date"] == row["date"]) & (result_df["mut"] == "WT")]["pec50_mean"].values[0]

        if not normed:
            norm_data_array = norm_data_array / ref_emax * 100
            if not os.path.exists(f"{data_dir}/{row[0]}-{row[2]}"):
                os.mkdir(f"{data_dir}/{row[0]}-{row[2]}")
            if not os.path.exists(f"{data_dir}/{row[0]}-{row[2]}/norm"):
                os.mkdir(f"{data_dir}/{row[0]}-{row[2]}/norm")
            pd.DataFrame(norm_data_array).to_csv(f"{data_dir}/{row[0]}-{row[2]}/norm/{row[0]}-{row[2]} {row[1]} {row[3]} {row[4]}MIN norm.csv", header=None, index=None)
            norm_data_df = norm_data_df.append(pd.DataFrame(norm_data_array.reshape(1, plate_row*plate_col)), ignore_index=True)


        batch_result_df.insert(len(batch_result_df.columns), "ref_emax", ref_emax, allow_duplicates=True)
        batch_result_df.insert(len(batch_result_df.columns), "ref_pec50", ref_pec50, allow_duplicates=True)
        result_df = pd.concat([result_df, batch_result_df], ignore_index=True)

        if 'ref_emax' in locals():
            del ref_emax
        else:
            print("no ref_max")
        if 'ref_pec50' in locals():
            del ref_pec50
        else:
            print("no ref_ec50")

    if not normed:
        norm_data_df.to_csv(f"{data_dir}/export/{data_dir}_data_normed.csv", header=None, index=None)
    result_df.to_csv(f"{data_dir}/export/{data_dir}_{export_postfix}.csv", index=None)


def main():
    batch_fit("data_dir")
    batch_fit("data_dir", normed=True)


if __name__ == "__main__":
    main()