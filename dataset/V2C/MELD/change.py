import pandas as pd

def normalize_text(text):
    # 替换弯引号和其他常见的智能符号
    translation_table = str.maketrans({
        "\u2018": "'",  # ‘ -> '
        "\u2019": "'",  # ’ -> '
        "\u201c": '"',  # “ -> "
        "\u201d": '"',  # ” -> "
    })
    return text.translate(translation_table)

# 2. 读取 CSV 文件
input_file = '/data2/ruixin/datasets/MELD_clips/metadata.csv'   # 输入文件名
output_file = '/data2/ruixin/datasets/MELD_clips/metadata2.csv' # 输出文件名
target_column = 'Utterances'    # 你想处理的那一列的列头名称

# 读取文件
df = pd.read_csv(input_file)

# 3. 核心步骤：对指定列应用函数
# .apply() 会自动把 normalize_text 应用到这一列的每一行
if target_column in df.columns:
    df[target_column] = df[target_column].apply(normalize_text)
    print(f"列 '{target_column}' 处理完成。")
else:
    print(f"错误：找不到列名 '{target_column}'")

# 4. 保存结果
# index=False 表示不保存行号，encoding='utf-8-sig' 可以防止 Excel 打开中文乱码
df.to_csv(output_file, index=False, encoding='utf-8-sig')