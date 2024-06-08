import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t, mstats

# 读取数据集
df_control_factors = pd.read_csv(r"D:\Desktop\partial result\part3FM回归\dataForFM.csv")
df_rf_returns = pd.read_csv(r"D:\Desktop\年度数据\TRD_Nrrateyear.csv")
df_idiosyncratic_volatility = pd.read_csv(r"D:\Desktop\partial result\part1异质波动率\idiosyncratic_volatility_yearlyt_1.csv")
df_stock_returns = pd.read_csv(r"D:\Desktop\年度数据\TRD_Yearnew.csv")

# 筛选沪深主板数据
df_stock_returns = df_stock_returns[df_stock_returns['Yarkettype'].isin([1, 4])]

# 将年无风险收益率除以100以统一收益率单位，并按年聚合
df_rf_returns['Nrrdata'] = df_rf_returns['Nrrdata'] / 100
df_rf_returns['Year'] = pd.to_datetime(df_rf_returns['Date']).dt.year
df_rf_annual = df_rf_returns.groupby('Year')['Nrrdata'].mean().reset_index()

# 合并数据集
df = pd.merge(df_stock_returns[['Year', 'Stkcd', 'Yretwd']], df_rf_annual, on='Year')
df = pd.merge(df, df_control_factors[['Year', 'Stkcd', 'Size', 'BM']], on=['Year', 'Stkcd'])
df = pd.merge(df, df_idiosyncratic_volatility[['Year', 'Stkcd', 'Volatility']], on=['Year', 'Stkcd'])

# 调整 Size 和 BM 为t-1期
df_control_factors['Year'] += 1
df = pd.merge(df, df_control_factors[['Year', 'Stkcd', 'Size', 'BM']], on=['Year', 'Stkcd'], suffixes=('', '_t-1'))

# 温莎化处理自变量
variables_to_winsorize = ['Size_t-1', 'BM_t-1', 'Volatility']
for variable in variables_to_winsorize:
    df[variable] = mstats.winsorize(df[variable], limits=[0.005, 0.005])

# 筛选出所有变量数据完整的时间
df.dropna(inplace=True)

# 计算Excess_return
df['Excess_return'] = df['Yretwd'] - df['Nrrdata']

# 定义Fama-Macbeth回归函数，使用Newey-West标准误
def fama_macbeth_regression(data, variables):
    X = data[variables]
    y = data['Excess_return']

    # 每年进行截面回归，保存回归系数和调整后的R^2
    annual_results = []
    annual_adj_r2 = []
    for year in data['Year'].unique():
        yearly_data = data[data['Year'] == year]
        X_yearly = sm.add_constant(yearly_data[X.columns])  # 添加常数项
        model = sm.OLS(yearly_data['Excess_return'], X_yearly)
        results = model.fit()
        annual_results.append(results.params)
        annual_adj_r2.append(results.rsquared_adj)

    # 将截面回归的结果汇总成一个DataFrame
    df_annual_results = pd.DataFrame(annual_results, columns=X.columns.insert(0, 'const'))

    # 计算平均值
    means = df_annual_results.mean()
    mean_adj_r2 = np.mean(annual_adj_r2)

    # 使用Newey-West方法计算标准误差和t统计量
    newey_west_errors = []
    for col in df_annual_results.columns:
        model = sm.OLS(df_annual_results[col], np.ones(len(df_annual_results)))
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})  # 这里的maxlags可以根据需要调整
        newey_west_errors.append(results.bse.iloc[0])  # 使用iloc按位置访问元素

    newey_west_errors = np.array(newey_west_errors)
    t_values = means / newey_west_errors

    # 计算p值
    dof = len(df_annual_results) - 1
    p_values = 2 * (1 - t.cdf(np.abs(t_values), dof))

    return means, t_values, newey_west_errors, p_values, mean_adj_r2

# 执行不同的Fama-Macbeth回归，并保存结果
all_results = []

variables_list = [['Size_t-1'],
                  ['BM_t-1'],
                  ['Volatility'],
                  ['Size_t-1', 'BM_t-1', 'Volatility']]

for i, variables in enumerate(variables_list, start=1):
    means, t_values, std_errors, p_values, mean_adj_r2 = fama_macbeth_regression(df, variables)

    # 构造结果的DataFrame
    results_df = pd.DataFrame({
        'Variable': ['const'] + variables,
        'Mean Coefficient': means,
        'Newey-West Standard Error': std_errors,
        't-value': t_values,
        'p-value': p_values
    })
    results_df['Mean Adjusted R^2'] = mean_adj_r2

    all_results.append(results_df)

# 合并所有回归结果
final_results_df = pd.concat(all_results)

# 输出结果到CSV文件
final_results_df.to_csv("D:/Desktop/fama_macbeth_results.csv", index=False)

print("Fama-Macbeth回归结果已保存到 'D:/Desktop/fama_macbeth_results.csv'")



















