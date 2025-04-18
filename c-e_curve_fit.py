import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from fetch_data import batch_reshape
from utils import update_progress


def fit_dec(x_data, y_data):
    initial_guess = [np.median(x_data), max(y_data) - min(y_data), min(y_data)]
    def func(log_x, log_ec50, emax, base):
        return emax / (1 + 10 ** (log_ec50 - log_x)) + base
    try:
        optimized_params, cov = curve_fit(func, x_data, y_data, p0=initial_guess, maxfev=2000)
        y_predicted = func(x_data, *optimized_params)  # Calculate predicted y-values from the model
        ssr = np.sum((y_data - y_predicted) ** 2)  # Calculate the sum of squared residuals (SSR)
        y_mean = np.mean(y_data)
        sst = np.sum((y_data - y_mean) ** 2)  # Calculate the total sum of squares (SST)
        r_sqr = np.round(1 - (ssr / sst), 3)  # Calculate R-squared
        if cov[0, 0] < 0 or cov[1, 1] < 0 or cov[2, 2] < 0:
            fit_para = {"pec50_mean": "negetive covariance", "base_mean": np.mean(y_data)}
        else:
            fit_para = {"pec50_mean": np.round(-optimized_params[0],3), "pec50_err": np.round(np.sqrt(cov[0, 0]),3), "min_max": np.round(max(y_data)-min(y_data),3), "emax_mean": np.round(optimized_params[1],3), "emax_err": np.round(np.sqrt(cov[1, 1]),3), "base_mean": np.round(optimized_params[2],3), "base_err": np.round(np.sqrt(cov[2, 2]),3), "r_sqr": r_sqr}
        # print(m, " done")

    except RuntimeError as e:
        fit_para = {"pec50_mean": str(e), "base_mean": np.mean(y_data)}
    return fit_para


def batch_fit(data_df, out_path="result/first_fit.csv"):
    reshaped_list = batch_reshape(data_df=data_df,row="conc",column="receptor",single_col=True,output_path=None)
    result_df = pd.DataFrame(columns=['date', 'type', 'plate', 'ligand', 'receptor', 'time', 'pec50_mean', 'pec50_err', 'min_max', 'emax_mean', 'emax_err', 'base_mean', 'base_err', 'r_sqr'])
    total = len(reshaped_list)
    for info_dict, data_arr in reshaped_list:
        x_data = np.array(list(data_arr["conc"]) * (data_arr.shape[1] - 1)).astype(np.float32)
        y_data = np.array(data_arr.iloc[:,1:]).transpose().flatten().astype(np.float32)
        fit_para = fit_dec(x_data, y_data)
        fit_para.update(info_dict)
        result_df = pd.concat([result_df,pd.DataFrame([fit_para])], ignore_index=True)
        update_progress(len(result_df),total,"fitting: ")
    if out_path is not None:
        result_df.to_csv(out_path,index=None)
    return result_df
