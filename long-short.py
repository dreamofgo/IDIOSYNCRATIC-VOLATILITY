import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# 读取月度收益率数据集
monthly_returns_df = pd.read_csv(r"D:\Desktop\月度数据\TRD_Mnth.csv")
# 打印列名
print(monthly_returns_df.columns)

# 转换日期格式，并提取年份和月份
monthly_returns_df['Date'] = pd.to_datetime(monthly_returns_df['Date'], format='%Y/%m/%d')
monthly_returns_df['Year'] = monthly_returns_df['Date'].dt.year
monthly_returns_df['Month'] = monthly_returns_df['Date'].dt.month

# 读取组合股票代码数据集
portfolio_stkcds = pd.read_csv(r"D:\Desktop\portfolio_stkcds.csv")

# 解析 Stkcd_List 列
portfolio_stkcds['Stkcd_List'] = portfolio_stkcds['Stkcd_List'].apply(ast.literal_eval)

# 初始化 DataFrame 存储 5-1 多空组合的月度收益率（等权重）
long_short_returns_ew = pd.DataFrame()

# 按年份循环
for year in range(2001, 2020):
    # 获取当前年份的组合股票代码
    portfolio_5_stocks_ew = portfolio_stkcds[
        (portfolio_stkcds['Date'] == year) & (portfolio_stkcds['Volatility_Quantile'] == 5)
        ]['Stkcd_List'].values[0]

    portfolio_1_stocks_ew = portfolio_stkcds[
        (portfolio_stkcds['Date'] == year) & (portfolio_stkcds['Volatility_Quantile'] == 1)
        ]['Stkcd_List'].values[0]

    # 获取当前年份及下一年份的月度数据
    year_monthly_data = monthly_returns_df[
        ((monthly_returns_df['Year'] == year) & (monthly_returns_df['Month'] >= 7)) |
        ((monthly_returns_df['Year'] == year + 1) & (monthly_returns_df['Month'] <= 6))
        ]

    # 筛选组合 5 和 组合 1 的月度收益率（等权重）
    portfolio_5_returns_ew = year_monthly_data[year_monthly_data['Stkcd'].isin(portfolio_5_stocks_ew)]
    portfolio_1_returns_ew = year_monthly_data[year_monthly_data['Stkcd'].isin(portfolio_1_stocks_ew)]

    # 计算等权重月度平均收益率
    portfolio_5_avg_returns_ew = portfolio_5_returns_ew.groupby('Date')['Mretwd'].mean()
    portfolio_1_avg_returns_ew = portfolio_1_returns_ew.groupby('Date')['Mretwd'].mean()

    # 计算5-1多空组合的月度收益率（等权重）
    long_short_monthly_returns_ew = portfolio_5_avg_returns_ew - portfolio_1_avg_returns_ew
    long_short_monthly_returns_ew = long_short_monthly_returns_ew.reset_index()
    long_short_monthly_returns_ew['Year'] = year
    long_short_monthly_returns_ew.rename(columns={0: 'Return'}, inplace=True)

    # 将结果添加到 DataFrame
    long_short_returns_ew = pd.concat([long_short_returns_ew, long_short_monthly_returns_ew])

# 设置显示的最大行数和最大列数
pd.set_option('display.max_rows', 1000)  # 设置最大行数为 1000
pd.set_option('display.max_columns', 100)  # 设置最大列数为 100

# 打印结果
print("\n5-1 Long-Short Portfolio Monthly Returns (Equally Weighted)")
print(long_short_returns_ew)

# 保存结果到 CSV 文件
long_short_returns_ew.to_csv(r'D:\Desktop\long_short_portfolio_monthly_returns_ew.csv', index=False)

# 转换日期格式
long_short_returns_ew['Date'] = pd.to_datetime(long_short_returns_ew['Date'])

# 绘制等权重的5-1多空组合月度收益率折线图
plt.figure(figsize=(14, 7))
plt.plot(long_short_returns_ew['Date'], long_short_returns_ew['Mretwd'], label='5-1 Long-Short Portfolio (Equally Weighted)')
plt.xlabel('Date')
plt.ylabel('Monthly Return')
plt.title('5-1 Long-Short Portfolio Monthly Returns (Equally Weighted)')
plt.legend()
plt.grid(True)
plt.show()















