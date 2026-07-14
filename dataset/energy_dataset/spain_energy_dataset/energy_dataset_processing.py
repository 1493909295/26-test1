# 本文件用于对西班牙能源数据集进行分析、预处理

#------------ 源数据集内容列内容注释 ---------------#

# ---------------------------------------------------------------------------------
#| 列名                                            |    解释                          |
#| ---------------------------------------------------------------------------------
#| time                                           |    时间戳，+1代表与格林威治时间偏差  |
#| generation fossil brown coal/lignite           |    褐煤/压燃煤发电量               |
#| generation fossil gas                          |    天然气发电量                   |
#| generation fossil coal-derived gas             |    煤气发电量                     |
#| generation fossil hard coal                    |    硬煤/无烟煤发电量               |
#| generation fossil oil                          |    石油发电量                     |
#| generation fossil oil shale                    |    页岩油发电量                   |
#| generation fossil peat                         |    泥煤发电量                     |
#| generation nuclear                             |    核电发电量                     |
#| generation waste                               |    垃圾焚烧发电量                  |
#| generation other                               |    其他来源发电量                  |
#| generation biomass                             |    生物质能发电量                  |
#| generation geothermal                          |    地热能发电量                    |
#| generation hydro run-of-river and poundage     |    径流式及蓄水式水力发电量          |
#| generation hydro water reservoir               |    水库水力发电量                  |
#| generation marine                              |    海洋能发电量                    |
#| generation solar                               |    太阳能发电量                    |
#| generation wind offshore                       |    海上风电发电量                  |
#| generation wind onshore                        |    陆上风电发电量                  |
#| generation other renewable                     |    其他可再生发电量                |
#| generation hydro pumped storage consumption    |    抽水蓄能电站消耗的电量           |
#| forecast solar day ahead                       |    日前预测的太阳能发电量           |
#| forecast wind onshore day ahead                |    日前预测的陆上风电发电量         |
#| total load forecast                            |    日前预测的总电力负荷             |
#| total load actual                              |    实际测量的总电力负荷             |
#| price day ahead                                |    日前市场电价                   |
#| price actual                                   |    实时市场实际结算电价            |
#-----------------------------------------------------------------------------------


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# 使用平均数补全数据集
def fill_missing_with_mean(df):
    """
    遍历数据集，使用每一列的平均数填补缺项部分。
    （仅对数值类型的列进行操作，避开时间戳等非数值列）
    """
    print("开始使用均值填补缺失项...")
    # 获取所有数值类型的列
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # 检查该列是否有缺失值
        if df[col].isnull().any():
            # 计算平均数，fillna 会自动跳过 NaN 计算均值
            mean_val = df[col].mean()
            # 填补缺失值
            df[col] = df[col].fillna(mean_val)
    return df

# 计算不可再生能源与可再生能源所占比例
def calculate_rates(df):
    """
    分别计算每一行不可再生能源与可再生能源占总发电量的百分比，
    并记录在新的列 “Non_renewable_rate” 与 “renewable_rate” 中。
    """
    print("开始计算可再生能源与不可再生能源占比...")

    # 不可再生能源列名
    non_renewable_cols = [
        'generation fossil brown coal/lignite',
        'generation fossil gas',
        'generation fossil coal-derived gas',
        'generation fossil hard coal',
        'generation fossil oil',
        'generation fossil oil shale',
        'generation fossil peat',
        'generation nuclear',
        'generation waste',
        'generation other'
    ]

    # 可再生能源列名
    renewable_cols = [
        'generation biomass',
        'generation geothermal',
        'generation hydro run-of-river and poundage',
        'generation hydro water reservoir',
        'generation marine',
        'generation solar',
        'generation wind offshore',
        'generation wind onshore',
        'generation other renewable'
    ]

    # 按行求和：计算各自的总发电量
    total_non_renewable = df[non_renewable_cols].sum(axis=1)
    total_renewable = df[renewable_cols].sum(axis=1)

    # 计算当前行的总发电量
    total_generation = total_non_renewable + total_renewable

    # 计算百分比并创建新列 (避免除以 0 的极小概率情况，当总发电量为0时，占比设为0)
    df['Non_renewable_rate'] = (total_non_renewable / total_generation).fillna(0)
    df['renewable_rate'] = (total_renewable / total_generation).fillna(0)

    return df

# 将可再生能源与不可再生能源占比化为折线图
def plot_rates_over_time(csv_file="new_enery_dataset.csv", output_image="rates_over_time.png"):
    """
    读取数据集，将时间列作为横坐标，可再生与不可再生能源比例作为纵坐标绘制折线图
    """
    print(f"正在读取数据：{csv_file} ...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"找不到文件：{csv_file}，请确保路径正确。")
        return
    matplotlib.use('TkAgg')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 1. 提取时间特征并将时间序列作为索引，方便后续按时间维度绘图
    # 加上 utc=True 以正确解析带有时区偏移量 "+01:00" 的时间字符串
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time')
    df.set_index('time', inplace=True)

    # 2. 由于原数据是按小时记录的 (3.5万行)，直接绘制折线图会形成一团乱麻
    # 这里我们采用“按天聚合(Resample to Daily)”计算每日平均占比，使趋势更加平滑直观
    # 如果您仍想看原始的小时级波动，可以将下一行的 'D' 改为 'H'，或者直接删除这两行
    df_daily = df[['Non_renewable_rate', 'renewable_rate']].resample('D').mean()

    # 3. 开始创建并绘制折线图
    # 设置画布大小：宽15寸，高6寸，以便完整铺开几年的时间线
    plt.figure(figsize=(15, 6))

    # 绘制不可再生能源占比 (红色线)
    plt.plot(df_daily.index, df_daily['Non_renewable_rate'],
             label='Non-Renewable Rate (不可再生)', color='#1078bf', linewidth=1.5, alpha=0.8)

    # 绘制可再生能源占比 (绿色线)
    plt.plot(df_daily.index, df_daily['renewable_rate'],
             label='Renewable Rate (可再生)', color='#76a700', linewidth=1.5, alpha=0.8)

    # 4. 图表排版与美化装饰
    plt.title('Daily Average Generation Rates: Renewable vs Non-renewable (2015-2018)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Proportion of Total Generation (0.0 - 1.0)', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')

    # 添加网格线以帮助读数
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 5. 将生成的折线图保存到当前路径下
    plt.savefig(output_image)
    print(f"折线图绘制成功！已保存为图片：{output_image}")

# 调用 fill_missing_with_mean(df) 和 calculate_rates(df)
def main():
    # 假设输入文件名为 source_energy_dataset.csv，请根据实际情况确认
    input_file = "source_energy_dataset.csv"
    output_file = "new_enery_dataset.csv"

    if not os.path.exists(input_file):
        print(f"错误：在当前路径下找不到文件 '{input_file}'，请检查文件名或路径。")
        return

    # 读取数据集
    print(f"正在读取数据集：{input_file}")
    df = pd.read_csv(input_file)

    # 1. 填补缺失值
    df = fill_missing_with_mean(df)

    # 2. 计算占比
    df = calculate_rates(df)

    # 将处理后的数据集保存为新文件
    df.to_csv(output_file, index=False)

    print(f"处理完成！新的数据集已成功保存至：{os.path.abspath(output_file)}")
    print("\n新生成列的数据预览：")
    print(df[['time', 'Non_renewable_rate', 'renewable_rate']].head())


# 将可再生能源与不可再生能源占比化为折线图
def plot_rates_solar(csv_file="new_enery_dataset.csv", output_image="rates_solar.png"):
    """
    读取数据集，将时间列作为横坐标，可再生与不可再生能源比例作为纵坐标绘制折线图
    """
    print(f"正在读取数据：{csv_file} ...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"找不到文件：{csv_file}，请确保路径正确。")
        return
    matplotlib.use('TkAgg')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 1. 提取时间特征并将时间序列作为索引，方便后续按时间维度绘图
    # 加上 utc=True 以正确解析带有时区偏移量 "+01:00" 的时间字符串
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time')
    df.set_index('time', inplace=True)

    # 2. 由于原数据是按小时记录的 (3.5万行)，直接绘制折线图会形成一团乱麻
    # 这里我们采用“按天聚合(Resample to Daily)”计算每日平均占比，使趋势更加平滑直观
    # 如果您仍想看原始的小时级波动，可以将下一行的 'D' 改为 'H'，或者直接删除这两行
    df_daily = df[['Non_renewable_rate', 'renewable_rate']].resample('D').mean()

    # 3. 开始创建并绘制折线图
    # 设置画布大小：宽15寸，高6寸，以便完整铺开几年的时间线
    plt.figure(figsize=(15, 6))

    # 绘制不可再生能源占比 (红色线)
    plt.plot(df_daily.index, df_daily['Non_renewable_rate'],
             label='Non-Renewable Rate (不可再生)', color='#1078bf', linewidth=1.5, alpha=0.8)

    # 绘制可再生能源占比 (绿色线)
    plt.plot(df_daily.index, df_daily['renewable_rate'],
             label='Renewable Rate (可再生)', color='#76a700', linewidth=1.5, alpha=0.8)

    # 4. 图表排版与美化装饰
    plt.title('Daily Average Generation Rates: Renewable vs Non-renewable (2015-2018)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Proportion of Total Generation (0.0 - 1.0)', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')

    # 添加网格线以帮助读数
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 5. 将生成的折线图保存到当前路径下
    plt.savefig(output_image)
    print(f"折线图绘制成功！已保存为图片：{output_image}")



#------------------计算每一种可再生能源的占比-----------------------#

def calculate_specific_renewable_rates(file_path="new_enery_dataset.csv"):
    """
    计算特定可再生能源相对于 (所有不可再生能源 + 该特定可再生能源) 的占比，
    并将新列写入原数据集中。
    """
    print(f"正在读取文件：{file_path} ...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}，请确保路径或文件名正确。")
        return

    # 定义不可再生能源列（作为公式分母的基础部分）
    non_renewable_cols = [
        'generation fossil brown coal/lignite',
        'generation fossil gas',
        'generation fossil coal-derived gas',
        'generation fossil hard coal',
        'generation fossil oil',
        'generation fossil oil shale',
        'generation fossil peat',
        'generation nuclear',
        'generation waste',
        'generation other'
    ]

    # 预先计算出每一行不可再生能源的总发电量
    # 这样在后续循环计算中效率更高，不需要重复求和
    non_renewable_sum = df[non_renewable_cols].sum(axis=1)

    # 定义需要计算的可再生能源及其对应的新列名映射表
    renewable_mapping = {
        'generation biomass': 'biomass_rate',
        'generation geothermal': 'generation_geothermal_rate',
        'generation hydro run-of-river and poundage': 'generation_hydro_run-of-river_and_poundage_rate',
        'generation hydro water reservoir': 'generation_hydro_water_reservoir_rate',
        'generation marine': 'generation_marine_rate',
        'generation solar': 'generation_solar_rate',
        'generation wind offshore': 'generation_wind_offshore_rate',
        'generation wind onshore': 'generation_wind_onshore_rate',
        'generation other renewable': 'generation_other_renewable_rate'
    }

    print("开始计算各项可再生能源占比...")

    # 遍历每一种可再生能源进行计算
    for source_col, rate_col in renewable_mapping.items():
        # 分母 = 不可再生能源总和 + 当前的这项可再生能源
        denominator = non_renewable_sum + df[source_col]

        # 计算比例。为了防止出现除以 0 的情况（比如某些时段停机无发电），使用 fillna(0) 兜底
        df[rate_col] = (df[source_col] / denominator).fillna(0)

        print(f"已生成新列: {rate_col}")

    # 将计算后的结果覆盖保存回原文件
    df.to_csv(file_path, index=False)
    print(f"\n所有比例计算完成！更新后的数据已保存至：{os.path.abspath(file_path)}")

    # 打印部分结果以供预览
    preview_cols = ['time', 'biomass_rate', 'generation_solar_rate', 'generation_wind_onshore_rate']
    print("\n新数据集数据预览（部分列）：")
    print(df[preview_cols].head())

#-------------------给四种可再生能源占比画图-----------------------#

def plot_all_renewable_rates(file_path="new_enery_dataset.csv", output_image="detailed_renewable_rates.png"):
    """
    读取包含各项可再生能源占比的数据集，按天进行重采样并绘制折线图。
    """
    print(f"正在读取文件：{file_path} ...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}，请确保前一步已正确生成该文件。")
        return
    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 1. 解析时间列，并设置为索引
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time')
    df.set_index('time', inplace=True)

    # 2. 定义需要绘制的列名
    rate_columns = [
        'generation_hydro_run-of-river_and_poundage_rate',
        'generation_hydro_water_reservoir_rate',
        'generation_solar_rate',
        'generation_wind_onshore_rate'
    ]

    # 检查列是否都存在于数据集中
    missing_cols = [col for col in rate_columns if col not in df.columns]
    if missing_cols:
        print(f"警告：以下列在数据集中缺失，将跳过绘制：{missing_cols}")
        rate_columns = [col for col in rate_columns if col in df.columns]

    # 3. 按天 (D) 对数据进行重采样并求均值
    print("正在按天聚合数据...")
    df_daily = df[rate_columns].resample('D').mean()

    # 4. 绘制折线图
    print("开始绘制折线图...")
    # 设置大尺寸画布以容纳多条折线
    plt.figure(figsize=(16, 8))

    # 循环绘制每一条折线，并稍微调整线宽和透明度以免严重遮挡
    for col in rate_columns:
        # 为了让图例更简洁，去掉前缀的 'generation_' 和后缀的 '_rate'
        label_name = col.replace('generation_', '').replace('_rate', '').replace('_', ' ').title()
        plt.plot(df_daily.index, df_daily[col], label=label_name, linewidth=1.2, alpha=0.8)

    # 5. 图表装饰与排版
    plt.title('Daily Average Rates of Specific Renewable Energy Sources', fontsize=16)
    plt.xlabel('Time (Year)', fontsize=12)
    plt.ylabel('Proportion (Source / (Non-renewable Total + Source))', fontsize=12)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.5)

    # 将图例放置在图表右侧外部，防止遮挡折线
    plt.legend(title="Renewable Sources", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    # 6. 保存图表
    plt.savefig(output_image, dpi=300)
    print(f"绘制成功！折线图已保存为：{output_image}")


if __name__ == "__main__":
    plot_all_renewable_rates()



