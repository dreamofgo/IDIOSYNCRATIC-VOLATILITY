import pandas as pd
import numpy as np

# 加载数据
file_path =r'D:\Study\Investment\result_2008_2023EW.xlsx'
data = pd.read_excel(file_path, sheet_name='Full Data', parse_dates=['Trdmnt'])

# 将 'market_value' 列转换为数值类型
data['market_value'] = pd.to_numeric(data['market_value'], errors='coerce')

# 定义函数来计算单个市场每月的因子
def calculate_factors_for_market(data):
    # 获取唯一的日期
    unique_dates = data['Trdmnt'].dt.to_period('Y').unique()
    
    # 初始化列表以存储因子值
    smb_values = []
    hml_values = []
    mkt_values = []
    risk_free_values = []
    dates = []
    mretwd_values = []
    smb_ch3_values = []
    vmg_values = []

    # 对每个月的数据进行计算
    for date in unique_dates:
        month_data = data[data['Trdmnt'].dt.to_period('Y') == date]
        
        # 计算SMB、HML、MKT、risk free rate和Mretwd
        smb = calculate_smb(month_data)
        hml = calculate_hml(month_data)
        mkt = month_data['MKT'].mean()
        risk_free = month_data['risk_free_rate'].mean()
        mretwd = month_data['Mretwd'].mean()
        
        # 计算SMB-CH3和VMG因子
        smb_ch3 = calculate_smb_ch3(month_data)
        vmg = calculate_vmg(month_data)
        
        # 将因子值附加到列表中
        smb_values.append(smb)
        hml_values.append(hml)
        mkt_values.append(mkt)
        risk_free_values.append(risk_free)
        dates.append(date.to_timestamp())
        mretwd_values.append(mretwd)
        smb_ch3_values.append(smb_ch3)
        vmg_values.append(vmg)

    # 创建因子的DataFrame
    factors = pd.DataFrame({
        'Trdmnt': dates,
        'SMB': smb_values,
        'HML': hml_values,
        'MKT': mkt_values,
        'risk_free_rate': risk_free_values,
        'Mretwd': mretwd_values,
        'SMB-CH3': smb_ch3_values,
        'VMG': vmg_values
    })
    
    # 将 'Trdmnt' 列转换为仅包含年月
    factors['Trdmnt'] = factors['Trdmnt'].dt.to_period('Y').astype(str)
    
    return factors

# 定义计算SMB的函数
def calculate_smb(data):
    try:
        # 计算断点
        bm_30 = data[data['Bm'] > 0]['Bm'].quantile(0.3)
        bm_70 = data[data['Bm'] > 0]['Bm'].quantile(0.7)
        market_value_median = data[data['market_value'] > 0]['market_value'].median()
        
        # 分组
        low_bm_low_mv = data[(data['Bm'] <= bm_30) & (data['market_value'] <= market_value_median)]['Mretwd'].mean()
        mid_bm_low_mv = data[(data['Bm'] > bm_30) & (data['Bm'] <= bm_70) & (data['market_value'] <= market_value_median)]['Mretwd'].mean()
        high_bm_low_mv = data[(data['Bm'] > bm_70) & (data['market_value'] <= market_value_median)]['Mretwd'].mean()
        
        low_bm_high_mv = data[(data['Bm'] <= bm_30) & (data['market_value'] > market_value_median)]['Mretwd'].mean()
        mid_bm_high_mv = data[(data['Bm'] > bm_30) & (data['Bm'] <= bm_70) & (data['market_value'] > market_value_median)]['Mretwd'].mean()
        high_bm_high_mv = data[(data['Bm'] > bm_70) & (data['market_value'] > market_value_median)]['Mretwd'].mean()
        
        # 计算SMB
        smb = (low_bm_low_mv + mid_bm_low_mv + high_bm_low_mv) / 3 - (low_bm_high_mv + mid_bm_high_mv + high_bm_high_mv) / 3
    except Exception as e:
        print(f"Error calculating SMB: {e}")
        smb = np.nan
    
    return smb

# 定义计算HML的函数
def calculate_hml(data):
    try:
        # 计算断点
        bm_30 = data[data['Bm'] > 0]['Bm'].quantile(0.3)
        bm_70 = data[data['Bm'] > 0]['Bm'].quantile(0.7)
        market_value_median = data[data['market_value'] > 0]['market_value'].median()
        
        # 分组
        low_bm_low_mv = data[(data['Bm'] <= bm_30) & (data['market_value'] <= market_value_median)]['Mretwd'].mean()
        high_bm_low_mv = data[(data['Bm'] > bm_70) & (data['market_value'] <= market_value_median)]['Mretwd'].mean()
        
        low_bm_high_mv = data[(data['Bm'] <= bm_30) & (data['market_value'] > market_value_median)]['Mretwd'].mean()
        high_bm_high_mv = data[(data['Bm'] > bm_70) & (data['market_value'] > market_value_median)]['Mretwd'].mean()
        
        # 计算HML
        hml = (high_bm_low_mv + high_bm_high_mv) / 2 - (low_bm_low_mv + low_bm_high_mv) / 2
    except Exception as e:
        print(f"Error calculating HML: {e}")
        hml = np.nan
    
    return hml

# 定义计算SMB-CH3的函数
def calculate_smb_ch3(data):
    try:
        # 去除市值最小的30%股票
        market_value_70th = data['market_value'].quantile(0.3)
        filtered_data = data[data['market_value'] > market_value_70th]
        
        # 计算断点
        market_value_median = filtered_data['market_value'].median()
        
        # 使用 1/PE 进行分组
        inv_pe = 1 / filtered_data['PE']
        inv_pe_50 = inv_pe.median()
        
        # 计算每个分组的市值和1/PE的平均回报率
        low_inv_pe_low_mv = filtered_data[(inv_pe <= inv_pe_50) & (filtered_data['market_value'] <= market_value_median)]['Mretwd'].mean()
        mid_inv_pe_low_mv = filtered_data[(inv_pe > inv_pe_50) & (inv_pe <= inv_pe_50 * 2) & (filtered_data['market_value'] <= market_value_median)]['Mretwd'].mean()
        high_inv_pe_low_mv = filtered_data[(inv_pe > inv_pe_50 * 2) & (filtered_data['market_value'] <= market_value_median)]['Mretwd'].mean()
        
        low_inv_pe_high_mv = filtered_data[(inv_pe <= inv_pe_50) & (filtered_data['market_value'] > market_value_median)]['Mretwd'].mean()
        mid_inv_pe_high_mv = filtered_data[(inv_pe > inv_pe_50) & (inv_pe <= inv_pe_50 * 2) & (filtered_data['market_value'] > market_value_median)]['Mretwd'].mean()
        high_inv_pe_high_mv = filtered_data[(inv_pe > inv_pe_50 * 2) & (filtered_data['market_value'] > market_value_median)]['Mretwd'].mean()
        
        # 计算SMB-CH3
        smb_ch3 = (low_inv_pe_low_mv + mid_inv_pe_low_mv + high_inv_pe_low_mv) / 3 - (low_inv_pe_high_mv + mid_inv_pe_high_mv + high_inv_pe_high_mv) / 3
    except Exception as e:
        print(f"Error calculating SMB-CH3: {e}")
        smb_ch3 = np.nan
    
    return smb_ch3

# 定义计算VMG的函数
def calculate_vmg(data):
    try:
        # 去除市值最小的30%股票
        market_value_70th = data['market_value'].quantile(0.3)
        filtered_data = data[data['market_value'] > market_value_70th]
        
        # 计算断点
        inv_pe = 1 / filtered_data['PE']
        inv_pe_30 = inv_pe.quantile(0.3)
        inv_pe_70 = inv_pe.quantile(0.7)
        market_value_median = filtered_data['market_value'].median()
        
        # 分组
        low_inv_pe_low_mv = filtered_data[(inv_pe <= inv_pe_30) & (filtered_data['market_value'] <= market_value_median)]['Mretwd'].mean()
        high_inv_pe_low_mv = filtered_data[(inv_pe > inv_pe_70) & (filtered_data['market_value'] <= market_value_median)]['Mretwd'].mean()
        
        low_inv_pe_high_mv = filtered_data[(inv_pe <= inv_pe_30) & (filtered_data['market_value'] > market_value_median)]['Mretwd'].mean()
        high_inv_pe_high_mv = filtered_data[(inv_pe > inv_pe_70) & (filtered_data['market_value'] > market_value_median)]['Mretwd'].mean()
        
        # 计算VMG
        vmg = (high_inv_pe_low_mv + high_inv_pe_high_mv) / 2 - (low_inv_pe_low_mv + low_inv_pe_high_mv) / 2
    except Exception as e:
        print(f"Error calculating VMG: {e}")
        vmg = np.nan
    
    return vmg

# 计算因子
factors = calculate_factors_for_market(data)

# 将因子数据写入Excel文件
output_file_path = r'D:\Study\Investment\factors_adjusted2008_2023EW.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    factors.to_excel(writer, sheet_name='Factors', index=False)

print(f"数据已成功保存至 {output_file_path}")
