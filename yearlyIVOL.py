import pandas as pd
import numpy as np
import statsmodels.api as sm

# 读取FF3因子&Carhart4因子数据，并筛选出MarkettypeID为“P9706”的数据
fama_french_factors = pd.read_csv("D:\\Desktop\\月度数据\\STK_MKT_CARHARTFOURFACTORS.csv",
                                  parse_dates=['Date'],
                                  usecols=['Date', 'MarkettypeID', 'RiskPremium1', 'SMB1', 'HML1', 'UMD1'])
fama_french_factors = fama_french_factors[fama_french_factors['MarkettypeID'] == 'P9706']

# 读取无风险收益率数据，并按月聚合
risk_free_rate = pd.read_csv("D:\\Desktop\\月度数据\\TRD_Nrratemonth.csv",
                             parse_dates=['Date'],
                             usecols=['Date', 'Nrrmtdt'])

# 按月聚合无风险收益率，取每月的平均值
risk_free_rate['Month'] = risk_free_rate['Date'].dt.to_period('M')
monthly_risk_free_rate = risk_free_rate.groupby('Month').mean().reset_index()
monthly_risk_free_rate['Date'] = monthly_risk_free_rate['Month'].dt.to_timestamp()

# 读取个股收益率数据，并筛选Markettype为1和4的数据
stock_returns = pd.read_csv("D:\\Desktop\\月度数据\\TRD_Mnth.csv",
                            parse_dates=['Date'],
                            usecols=['Date', 'Stkcd', 'Mretwd', 'Markettype'])
stock_returns = stock_returns[(stock_returns['Markettype'] == 1) | (stock_returns['Markettype'] == 4)]

# 将日期列转换为月度周期
stock_returns['Month'] = stock_returns['Date'].dt.to_period('M')
fama_french_factors['Month'] = fama_french_factors['Date'].dt.to_period('M')

# 合并无风险收益率数据到股票收益率数据中
stock_returns = stock_returns.merge(monthly_risk_free_rate[['Month', 'Nrrmtdt']], on='Month', how='left')

# 计算每只股票的超额收益率
stock_returns['ExcessReturn'] = stock_returns['Mretwd'] - stock_returns['Nrrmtdt']

# 合并Fama-French因子和动量因子数据
stock_returns = stock_returns.merge(fama_french_factors[['Month', 'RiskPremium1', 'SMB1', 'HML1', 'UMD1']], on='Month', how='left')

# 创建一个DataFrame存储特质波动率
idiosyncratic_volatility_list = []

# 回归并计算特质波动率
unique_stocks = stock_returns['Stkcd'].unique()
for stock in unique_stocks:
    # 准备回归数据
    stock_data = stock_returns[stock_returns['Stkcd'] == stock].copy()
    stock_data.loc[:, 'Year'] = stock_data['Date'].dt.year

    for year, data in stock_data.groupby('Year'):
        # 使用前一年的数据进行回归
        past_data = stock_data[stock_data['Year'] == year - 1]

        if not past_data.empty:
            y = past_data['ExcessReturn']
            X_ff3 = past_data[['RiskPremium1', 'SMB1', 'HML1']]

            # 检查并处理缺失值
            if y.isnull().any() or X_ff3.isnull().any().any():
                past_data = past_data.dropna(subset=['ExcessReturn', 'RiskPremium1', 'SMB1', 'HML1'])
                y = past_data['ExcessReturn']
                X_ff3 = past_data[['RiskPremium1', 'SMB1', 'HML1']]

            # 如果数据点的数量大于 3，则进行回归计算
            if len(past_data) > 3:
                # 添加常数项
                X_ff3 = sm.add_constant(X_ff3)

                # 进行OLS回归
                model_ff3 = sm.OLS(y, X_ff3).fit()

                # 获取回归残差
                residuals = model_ff3.resid

                # 计算特质波动率（残差的标准差），按照公式进行计算
                rse = np.sqrt(np.sum(residuals ** 2) / (len(past_data) - 3))
                trading_days_sqrt = np.sqrt(len(data))
                volatility = 100 * rse * trading_days_sqrt

                # 存储特质波动率
                idiosyncratic_volatility_list.append({
                    'Stkcd': stock,
                    'Year': year,
                    'Volatility': volatility
                })

# 转换为DataFrame
idiosyncratic_volatility = pd.DataFrame(idiosyncratic_volatility_list)

# 输出特质波动率到CSV文件
output_file = "D:\\Desktop\\idiosyncratic_volatility_yearlyt_1.csv"
idiosyncratic_volatility.to_csv(output_file, index=False)

print("特质波动率已保存到:", output_file)




