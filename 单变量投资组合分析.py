import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 读取数据集
returns_df = pd.read_csv(r"D:\Desktop\TRD_Yearnew.csv")
volatility_df = pd.read_csv(r"D:\Desktop\idiosyncratic_volatility_yearlyt_1.csv")
nrrate_df = pd.read_csv(r"D:\Desktop\TRD_Nrrateyear.csv")
ff3_factors_df = pd.read_csv(r"D:\Desktop\年度数据\STK_MKT_THRFACYEAR.csv")

# 修改时间变量名为 "Date"
returns_df.rename(columns={'Year': 'Date'}, inplace=True)
volatility_df.rename(columns={'Year': 'Date'}, inplace=True)
ff3_factors_df.rename(columns={'Year': 'Date'}, inplace=True)

# 将无风险收益率除以100修正单位不统一的问题
nrrate_df['Nrrdata'] /= 100

# 转换无风险收益率的日期格式，并按年聚合
nrrate_df['Date'] = pd.to_datetime(nrrate_df['Date'], format='%Y/%m/%d')  # 确认日期格式
nrrate_df['Year'] = nrrate_df['Date'].dt.year
annual_nrrate = nrrate_df.groupby('Year')['Nrrdata'].mean().reset_index()
annual_nrrate.columns = ['Date', 'Risk_Free_Rate']

# 筛选主板数据：仅保留与特质波动率数据集中的股票代码匹配的数据
main_board_stkcds = volatility_df['Stkcd'].unique()
returns_df = returns_df[returns_df['Stkcd'].isin(main_board_stkcds)]

# 合并收益率数据集和特质波动率数据集
data_df = pd.merge(returns_df, volatility_df, on=['Stkcd', 'Date'])

# 合并无风险收益率数据
data_df = pd.merge(data_df, annual_nrrate, on='Date')

# 只保留 FF3 因子数据中与收益率年份匹配的数据
ff3_factors_df = ff3_factors_df[ff3_factors_df['Date'].isin(data_df['Date'].unique())]

# 初始化用于存储结果的列表
results_eq_freq_ew = []
results_eq_freq_vw = []

# 初始化用于存储每年每个组合平均收益率的 DataFrame
portfolio_avg_returns_ew = pd.DataFrame(columns=['Year'] + [f'Portfolio{i}' for i in range(1, 6)])
portfolio_avg_returns_vw = pd.DataFrame(columns=['Year'] + [f'Portfolio{i}' for i in range(1, 6)])

# 初始化用于存储每年每个组合中股票代码的列表
portfolio_stkcds = []

# 获取所有日期列表
dates = sorted(data_df['Date'].unique())

# 分组数量
num_groups = 5

# 等频分组函数
def equal_frequency_grouping(data, num_groups):
    data['Volatility_Quantile_eq_freq'] = pd.qcut(data['Volatility'], num_groups, labels=False)
    data['Volatility_Quantile_eq_freq'] += 1
    return data

# 对每个日期进行再平衡
for date in dates:
    # 提取当前日期的数据
    date_data = data_df[data_df['Date'] == date].copy()

    # 分类方式：等频分组（基于股票数量）
    date_data = equal_frequency_grouping(date_data, num_groups)

    # 计算每个分组中异质波动率的平均值
    volatility_mean = date_data.groupby('Volatility_Quantile_eq_freq')['Volatility'].mean().reset_index()
    volatility_mean.columns = ['Volatility_Quantile', 'Mean_Volatility']

    # 计算等权重（EW）组的平均收益率和原始超额收益率
    date_data['Raw_Excess_Return_EW'] = date_data['Yretwd'] - date_data['Risk_Free_Rate']
    portfolio_analysis_eq_freq_ew = date_data.groupby('Volatility_Quantile_eq_freq').agg(
        {'Yretwd': 'mean', 'Raw_Excess_Return_EW': 'mean'}).reset_index()
    portfolio_analysis_eq_freq_ew.columns = ['Volatility_Quantile', 'Average_Return', 'Average_Raw_Excess_Return_EW']
    portfolio_analysis_eq_freq_ew['Date'] = date

    # 计算市值加权（VW）组的平均收益率和原始超额收益率
    date_data['Raw_Excess_Return_VW'] = date_data['Yretwd'] - date_data['Risk_Free_Rate']
    portfolio_analysis_eq_freq_vw = date_data.groupby('Volatility_Quantile_eq_freq', as_index=False).apply(
        lambda x: pd.Series({
            'Volatility_Quantile': x['Volatility_Quantile_eq_freq'].iloc[0],
            'Average_Return': np.average(x['Yretwd'], weights=x['FreeFlowMarketcap']),
            'Average_Raw_Excess_Return_VW': np.average(x['Raw_Excess_Return_VW'], weights=x['FreeFlowMarketcap'])
        })
    ).reset_index(drop=True)
    portfolio_analysis_eq_freq_vw['Date'] = date

    # 合并结果
    results_eq_freq_ew.append(pd.merge(volatility_mean, portfolio_analysis_eq_freq_ew, on='Volatility_Quantile'))
    results_eq_freq_vw.append(pd.merge(volatility_mean, portfolio_analysis_eq_freq_vw, on='Volatility_Quantile'))

    # 保存当前日期的组合股票代码
    for i in range(1, num_groups + 1):
        stkcds_in_group = date_data[date_data['Volatility_Quantile_eq_freq'] == i]['Stkcd'].tolist()
        portfolio_stkcds.append({'Date': date, 'Volatility_Quantile': i, 'Stkcd_List': stkcds_in_group})

    # 提取当前年份
    year = date_data['Date'].iloc[0]

    # 计算每个组合的平均收益率 (EW)
    avg_returns_per_portfolio_ew = portfolio_analysis_eq_freq_ew.pivot(index='Date', columns='Volatility_Quantile',
                                                                       values='Average_Return')
    avg_returns_per_portfolio_ew.columns = [f'Portfolio{i}' for i in range(1, 6)]
    avg_returns_per_portfolio_ew['Year'] = year

    # 计算每个组合的平均收益率 (VW)
    avg_returns_per_portfolio_vw = portfolio_analysis_eq_freq_vw.pivot(index='Date', columns='Volatility_Quantile',
                                                                       values='Average_Return')
    avg_returns_per_portfolio_vw.columns = [f'Portfolio{i}' for i in range(1, 6)]
    avg_returns_per_portfolio_vw['Year'] = year

    # 将当前年份的组合平均收益率添加到 DataFrame
    portfolio_avg_returns_ew = pd.concat(
        [portfolio_avg_returns_ew, avg_returns_per_portfolio_ew.reset_index(drop=True)], ignore_index=True)
    portfolio_avg_returns_vw = pd.concat(
        [portfolio_avg_returns_vw, avg_returns_per_portfolio_vw.reset_index(drop=True)], ignore_index=True)

# 合并所有日期的结果
final_results_eq_freq_ew = pd.concat(results_eq_freq_ew)
final_results_eq_freq_vw = pd.concat(results_eq_freq_vw)

# 将组合股票代码的结果转换为 DataFrame
portfolio_stkcds_df = pd.DataFrame(portfolio_stkcds)

# 保存组合股票代码信息
portfolio_stkcds_df.to_csv(r"D:\Desktop\portfolio_stkcds.csv", index=False)

# 设置显示的最大行数和最大列数
pd.set_option('display.max_rows', 1000)  # 设置最大行数为 1000
pd.set_option('display.max_columns', 100)  # 设置最大列数为 100

# 打印结果
print("\nEqual Frequency Quantiles (Equally Weighted, Yearly Rebalance)")
print(final_results_eq_freq_ew.groupby('Volatility_Quantile').agg(
    {'Mean_Volatility': 'mean', 'Average_Return': 'mean', 'Average_Raw_Excess_Return_EW': 'mean'}).reset_index())

print("\nEqual Frequency Quantiles (Value Weighted, Yearly Rebalance)")
print(final_results_eq_freq_vw.groupby('Volatility_Quantile').agg(
    {'Mean_Volatility': 'mean', 'Average_Return': 'mean', 'Average_Raw_Excess_Return_VW': 'mean'}).reset_index())

# 打开文件并写入CSV输出内容
final_results_eq_freq_ew_grouped = final_results_eq_freq_ew.groupby('Volatility_Quantile').agg(
    {'Mean_Volatility': 'mean', 'Average_Return': 'mean', 'Average_Raw_Excess_Return_EW': 'mean'}).reset_index()
final_results_eq_freq_vw_grouped = final_results_eq_freq_vw.groupby('Volatility_Quantile').agg(
    {'Mean_Volatility': 'mean', 'Average_Return': 'mean', 'Average_Raw_Excess_Return_VW': 'mean'}).reset_index()

final_results_eq_freq_ew_grouped.to_csv(r'D:\Desktop\final_results_eq_freq_ew.csv', index=False)
final_results_eq_freq_vw_grouped.to_csv(r'D:\Desktop\final_results_eq_freq_vw.csv', index=False)


# 打印每年每个组合的平均收益率 (EW)
print("\nAverage Returns of Each Portfolio by Year (Equally Weighted)")
print(portfolio_avg_returns_ew)

# 保存每年每个组合的平均收益率到CSV文件 (EW)
portfolio_avg_returns_ew.to_csv(r'D:\Desktop\portfolio_avg_returns_ew.csv', index=False)

# 打印每年每个组合的平均收益率 (VW)
print("\nAverage Returns of Each Portfolio by Year (Value Weighted)")
print(portfolio_avg_returns_vw)

# 保存每年每个组合的平均收益率到CSV文件 (VW)
portfolio_avg_returns_vw.to_csv(r'D:\Desktop\portfolio_avg_returns_vw.csv', index=False)

# 可视化等频分组（等权重）
plt.figure(figsize=(10, 6))
avg_returns_eq_freq_ew = final_results_eq_freq_ew.groupby('Volatility_Quantile')['Average_Return'].mean().reset_index()
plt.bar(avg_returns_eq_freq_ew['Volatility_Quantile'], avg_returns_eq_freq_ew['Average_Return'], color='blue')
plt.xlabel('Volatility Quantile (Equal Frequency)')
plt.ylabel('Average Return')
plt.title('Univariate Portfolio Analysis (Equal Frequency, Equally Weighted, Yearly Rebalance)')
plt.xticks(ticks=avg_returns_eq_freq_ew['Volatility_Quantile'])
plt.grid(True, axis='y')
plt.show()

# 可视化等频分组（市值加权）
plt.figure(figsize=(10, 6))
avg_returns_eq_freq_vw = final_results_eq_freq_vw.groupby('Volatility_Quantile')['Average_Return'].mean().reset_index()
plt.bar(avg_returns_eq_freq_vw['Volatility_Quantile'], avg_returns_eq_freq_vw['Average_Return'], color='orange')
plt.xlabel('Volatility Quantile (Equal Frequency)')
plt.ylabel('Average Return')
plt.title('Univariate Portfolio Analysis (Equal Frequency, Value Weighted, Yearly Rebalance)')
plt.xticks(ticks=avg_returns_eq_freq_vw['Volatility_Quantile'])
plt.grid(True, axis='y')
plt.show()

# 回归分析计算 Alpha
alphas_ff3_ew = []
alphas_ff3_vw = []
alphas_capm_ew = []
alphas_capm_vw = []
alphas_ch3_ew = []
alphas_ch3_vw = []


for i in range(1, 6):
    # 组合 i 的等权重和市值加权的原始超额收益率
    ew_excess_returns = final_results_eq_freq_ew[final_results_eq_freq_ew['Volatility_Quantile'] == i].set_index('Date')['Average_Raw_Excess_Return_EW']
    vw_excess_returns = final_results_eq_freq_vw[final_results_eq_freq_vw['Volatility_Quantile'] == i].set_index('Date')['Average_Raw_Excess_Return_VW']
    years = ew_excess_returns.index

    # FF3 模型
    X_ff3 = ff3_factors_df[ff3_factors_df['Date'].isin(years)].set_index('Date')[['RiskPremium1', 'SMB1', 'HML1']]
    X_ff3 = sm.add_constant(X_ff3)

    model_ff3_ew = sm.OLS(ew_excess_returns, X_ff3).fit()
    model_ff3_vw = sm.OLS(vw_excess_returns, X_ff3).fit()

    alphas_ff3_ew.append(model_ff3_ew.params['const'])
    alphas_ff3_vw.append(model_ff3_vw.params['const'])

    # CAPM 模型
    X_capm = ff3_factors_df[ff3_factors_df['Date'].isin(years)].set_index('Date')['RiskPremium1']
    X_capm = sm.add_constant(X_capm)

    model_capm_ew = sm.OLS(ew_excess_returns, X_capm).fit()
    model_capm_vw = sm.OLS(vw_excess_returns, X_capm).fit()

    alphas_capm_ew.append(model_capm_ew.params['const'])
    alphas_capm_vw.append(model_capm_vw.params['const'])

    # CH3 模型
    X_ch3 = ff3_factors_df[ff3_factors_df['Date'].isin(years)].set_index('Date')[['SMB_CH3', 'MKT_CH3', 'VMG_CH3']]
    X_ch3 = sm.add_constant(X_ch3)

    model_ch3_ew = sm.OLS(ew_excess_returns, X_ch3).fit()
    model_ch3_vw = sm.OLS(vw_excess_returns, X_ch3).fit()

    alphas_ch3_ew.append(model_ch3_ew.params['const'])
    alphas_ch3_vw.append(model_ch3_vw.params['const'])


# 打印回归结果
print("\nAlpha (FF3, Equally Weighted):", alphas_ff3_ew)
print("Alpha (FF3, Value Weighted):", alphas_ff3_vw)
print("\nAlpha (CAPM, Equally Weighted):", alphas_capm_ew)
print("Alpha (CAPM, Value Weighted):", alphas_capm_vw)
print("\nAlpha (CH3, Equally Weighted):", alphas_ch3_ew)
print("Alpha (CH3, Value Weighted):", alphas_ch3_vw)

# 保存结果到 CSV 文件
alphas_df = pd.DataFrame({
    'Portfolio': [f'Portfolio{i}' for i in range(1, 6)],
    'Alpha_FF3_EW': alphas_ff3_ew,
    'Alpha_FF3_VW': alphas_ff3_vw,
    'Alpha_CAPM_EW': alphas_capm_ew,
    'Alpha_CAPM_VW': alphas_capm_vw,
    'Alpha_CH3_EW': alphas_capm_ew,
    'Alpha_CH3_VW': alphas_capm_vw

})
alphas_df.to_csv(r'D:\Desktop\alphas.csv', index=False)



