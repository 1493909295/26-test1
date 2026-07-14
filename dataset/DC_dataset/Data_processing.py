import pandas as pd
import os
import matplotlib
import config as conf
matplotlib.use('Agg')

#-----------对于 job 数据集的分析---------
#   CPU                                     GPU
# 共有 76 CPU请求                            共有 18 种GPU请求
# 平均数：24.54                              平均数：2.02
# 众数：8                                    众数：1.0
# 8.00: 157,239 个                           1.00: 314,922 个
# 4.00: 77,704 个                            8.00: 61,343 个
# 2.00: 52,733 个                            0.50: 26,372 个
# 12.00: 33,962 个                           2.00: 22,985 个
# 112.00: 19,160 个                          4.00: 18,023 个
# 14.00: 14,281 个                           0.30: 11,996 个
# 184.00: 13,818 个                          0.20: 8,354 个
# 16.00: 10,929 个                           0.10: 2,428 个
# 20.00: 9,690 个                            6.00: 230 个
# 32.00: 7,647 个                            0.40: 74 个

# submit_time 的时间范围从 0 到 15,902,470 ，跨度太大，不能直接用
# 相邻时间的平均跨度是 34.06 秒，中位数跨度是 16秒

# duration（执行时间） 的平均数为 185,391.64 ，中位数是 1216.0
# 分布表现为右偏长尾分布
# 50%的任务在 1216 以内完成，75% 的任务在 4584 内完成，但排名前 1% 的极少数任务执行时间甚至超过了 5,663,026。

# ----------------------------------------------------------------------------------


######### 用于统计 阿里巴巴26 数据集任务文件中gpu_request种类  ########
def analyze_gpu_data(file_path):
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件，请检查路径是否正确：{file_path}")
        return
    try:
        # 2. 读取 CSV 文件
        # 如果文件很大，我们只读取需要的 gpu_request 这一列以节省内存
        df = pd.read_csv(file_path, usecols=['gpu_request'])

        # 3. 统计 gpu_request 列
        # value_counts 会统计每个唯一值出现的次数
        # dropna=False 会把缺失值 (NaN) 也统计进去
        counts = df['gpu_request'].value_counts(dropna=False)
        # 4. 获取唯一值的种类数量
        unique_types_count = len(counts)
        # 5. 打印结果
        print("="*40)
        print(f"数据集路径: {file_path}")
        print(f"gpu_request 共有 {unique_types_count} 种不同的取值")
        print("="*40)
        print("具体统计结果 (取值 : 出现次数):")
        print(counts.sort_index()) # 按 GPU 数量排序打印，方便查看
        print("="*40)
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
    # --- 执行脚本 ---
    # 使用 r 前缀处理 Windows 路径
    # target_path = r'C:\DuYan\experiment\26-MAPPO\dataset\job_info_df.csv'
    # analyze_gpu_data(target_path)

    # --- 统计结果 ---
    # gpu_request共有 18 种不同的取值
    # 具体统计结果(取值: 出现次数):
    # 0.01 2
    # 0.10 2428
    # 0.20 8354
    # 0.25 1
    # 0.30 11996
    # 0.40 74
    # 0.50 26372
    # 0.67 6
    # 0.80 2
    # 0.85 3
    # 1.00 314922
    # 2.00 22985
    # 3.00 46
    # 4.00 18023
    # 5.00 56
    # 6.00 230
    # 7.00 24
    # 8.00 61343

######### 用于统计 阿里巴巴26 数据集节点文件中gpu_capacity_numt的种类  ########
def analyze_node_gpu_capacity(file_path):
    """
    读取节点信息数据集并统计 gpu_capacity_num 列
    """
    # 1. 检查路径是否存在
    if not os.path.exists(file_path):
        print(f"错误：未找到文件，请确认路径是否正确: {file_path}")
        return

    try:
        # 2. 读取 CSV 文件 (仅加载需要的列以提高速度)
        # 目标列: gpu_capacity_num (节点拥有的GPU总数)
        df = pd.read_csv(file_path, usecols=['gpu_capacity_num'])

        # 3. 执行统计
        # value_counts 会统计每种 GPU 数量的节点出现了多少次
        stats = df['gpu_capacity_num'].value_counts(dropna=False)

        # 4. 计算种类数量
        unique_kinds = len(stats)

        # 5. 输出结果
        print("-" * 50)
        print(f"文件位置: {file_path}")
        print(f"gpu_capacity_num 共有 {unique_kinds} 种不同的取值。")
        print("-" * 50)
        print("具体分布如下 (GPU单机容量 : 节点数量):")

        # sort_index() 按照 GPU 数量从小到大排序显示
        print(stats.sort_index())
        print("-" * 50)

    except Exception as e:
        print(f"程序运行出错: {e}")

    # --- 执行部分 ---
    # 使用 r 前缀处理 Windows 路径中的反斜杠
    # csv_path = r'C:\DuYan\experiment\26-MAPPO\dataset\node_info_df.csv'
    # analyze_node_gpu_capacity(csv_path)

    # --- 统计结果 ---
    # gpu_capacity_num共有 3 种不同的取值。
    # 具体分布如下(GPU单机容量: 节点数量):
    # 1 3396
    # 4 10
    # 8 872

######### 用于统计 阿里巴巴26 数据集节点文件中cpu_num的种类  ########
def analyze_node_cpu_count(file_path):
    """
    读取节点信息数据集并统计 cpu_num 列（单机CPU核心数）的种类及分布
    """
    # 1. 检查文件路径是否存在
    if not os.path.exists(file_path):
        print(f"【错误】找不到文件，请确认路径是否正确: {file_path}")
        print("提示：请检查文件名中 'node_info_' 后面是否真的有一个空格。")
        return

    try:
        # 2. 读取 CSV 文件
        # 为了节省内存和提高速度，只加载 'cpu_num' 这一列
        df = pd.read_csv(file_path, usecols=['cpu_num'])

        # 3. 统计 cpu_num 列
        # value_counts 会统计每种 CPU 核心数对应的节点数量
        # dropna=False 用于统计是否存在空值
        cpu_stats = df['cpu_num'].value_counts(dropna=False)

        # 4. 获取唯一值的种类数量
        unique_kinds = len(cpu_stats)

        # 5. 打印分析报告
        print("=" * 55)
        print(f"分析文件: {os.path.basename(file_path)}")
        print(f"cpu_num 共有 {unique_kinds} 种不同的取值（硬件规格）")
        print("-" * 55)
        print("具体分布如下 (单机CPU核心数 : 节点台数):")

        # sort_index() 按照 CPU 核心数从小到大排序打印，方便查看硬件梯度
        print(cpu_stats.sort_index())
        print("=" * 55)

    except Exception as e:
        print(f"【程序运行出错】: {e}")
    # --- 执行脚本 ---
    # target_path = r'C:\DuYan\experiment\26-MAPPO\dataset\node_info_df.csv'
    # analyze_node_cpu_count(target_path)

    # cpu_num共有 3 种不同的取值（硬件规格）
    # 具体分布如下(单机CPU核心数: 节点台数):
    # cpu_num
    # 126 2
    # 128 2947
    # 192 1329



import matplotlib.pyplot as plt


def plot_duration_histogram(file_path):
    # 1. 读取数据
    df = pd.read_csv(file_path)

    print("=== 排名前 10 的最大 Duration 值 ===")
    top_10_durations = df['duration'].nlargest(10)
    print(top_10_durations)
    print("===================================\n")
    # 2. 设置画布大小
    plt.figure(figsize=(12, 6))

    # 3. 绘制直方图并启用对数刻度 (Y轴)
    plt.hist(df['duration'], bins=100, color='#4C72B0', edgecolor='black', log=True)

    # 4. 设置标题与坐标轴标签
    plt.title('Distribution of Job Duration (Log Scale for Frequency)', fontsize=14)
    plt.xlabel('Duration', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.ticklabel_format(style='plain', axis='x')

    # 防止数字太长互相重叠，将横轴标签倾斜15度
    plt.xticks(rotation=15)

    # 6. 保存图表
    output_filename = 'small_duration_distribution_plain.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"✅ 图表已成功生成并保存为: {output_filename}")

    # 关闭画布释放内存
    plt.close()


def extract_small_jobs(input_file, output_file, max_duration):
    """
    从原始数据集中提取 duration 在指定范围内的数据，并保存为新的 CSV 文件。

    参数:
    input_file (str): 原始数据集路径
    output_file (str): 新数据集保存路径
    max_duration (float/int): 允许的最大 duration 值
    """
    print(f"正在读取文件: {input_file} ...")
    # 1. 读取原始数据集
    df = pd.read_csv(input_file)

    # 2. 过滤数据：挑选出 duration <= max_duration 的行
    small_df = df[df['duration'] <= max_duration]

    # 3. 将过滤后的数据保存为新的 CSV 文件 (index=False 表示不保存行索引)
    small_df.to_csv(output_file, index=False)

    # 4. 打印结果对比，方便验证
    print("✅ 提取完成！")
    print(f" - 原始数据量: {len(df)} 行")
    print(f" - 提取数据量: {len(small_df)} 行 (占比约 {len(small_df) / len(df) * 100:.1f}%)")
    print(f" - 新文件已保存至当前路径: {output_file}")


# ========== 调用示例 ==========
# 将 duration 在 4584 内的数据提取出来
# extract_small_jobs(
#     input_file='job_info_df.csv',
#     output_file='small_job_info_df.csv',
#     max_duration=4584
# )
# ========== 调用示例 ==========

# if __name__ == "__main__":

from pathlib import Path




def process_gpu_request() -> None:
    """处理当前目录下的任务数据，并生成新的 CSV 文件。"""

    # 当前路径下的输入文件和输出文件。
    input_path = Path.cwd() / "small_job_info_df.csv"
    output_path = Path.cwd() / "new_small_job_info_df.csv"

    # 检查输入文件是否存在。
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件：{input_path}")

    # 读取 CSV 文件。
    df = pd.read_csv(input_path)

    # 检查 gpu_request 列是否存在。
    if "gpu_request" not in df.columns:
        raise KeyError(
            "CSV 文件中不存在 gpu_request 列。"
            f"当前列名为：{df.columns.tolist()}"
        )

    # 将 gpu_request 转换为数值类型。
    # 如果存在无法转换的非数值内容，会直接报错，避免静默修改数据。
    df["gpu_request"] = pd.to_numeric(
        df["gpu_request"],
        errors="raise",
    )

    # 找到 gpu_request 大于 1 的行。
    mask = df["gpu_request"] > 1

    # 将大于 1 的 GPU 需求除以 10。
    # 小于或等于 1 的值保持不变。
    df.loc[mask, "gpu_request"] = (
        df.loc[mask, "gpu_request"] / 10
    )

    # 保存为新的 CSV 文件，不保存 DataFrame 的行索引。
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("处理完成。")
    print(f"共修改 {int(mask.sum())} 条记录。")
    print(f"新文件保存位置：{output_path}")


if __name__ == "__main__":
    process_gpu_request()


