from sklearn.ensemble import RandomForestClassifier
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
import pandas as pd
import numpy as np
import argparse
#python=3.12

def add_sequence_labels(input_file, output_file):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            sequence_number = 1
            for line in infile:
                # 去除行末的换行符，并跳过空行
                line = line.strip()
                if line:
                    # 写入序号行
                    outfile.write(f"> {sequence_number}_{line}\n")
                    # 写入原始序列行
                    outfile.write(f"{line}\n")
                    sequence_number += 1
        print(f"Processed file saved as: {output_file}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
input_file = "text.fasta"  # 替换为你的输入文件名
output_file = "text2.fasta"  # 你想保存的输出文件名
add_sequence_labels(input_file, output_file)


# 训练随机森林模型
def train_model(training_file):
    # 加载数据（假设数据格式与R代码一致）
    data = pd.read_csv(training_file)
    X = data.drop(columns=["Class"])
    y = data["Class"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model


# 提取AAC和DPC特征
def extract_features(seq):
    analyzed_seq = ProteinAnalysis(str(seq))
    aac = analyzed_seq.get_amino_acids_percent()

    # 确保AAC的特征按固定顺序返回
    aac_values = [aac.get(aa, 0) for aa in "ACDEFGHIKLMNPQRSTVWY"]

    # 计算DPC（根据实际情况定义该函数）
    dpc = calculate_dpc(seq)

    # 确保DPC的特征按固定顺序返回
    dpc_values = [dpc.get(aa, 0) for aa in sorted(dpc.keys())]

    return aac_values + dpc_values


def calculate_dpc(seq):
    # 示例函数：计算二肽组成（Dipeptide Composition）
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    dipeptides = [a1 + a2 for a1 in amino_acids for a2 in amino_acids]

    total = len(seq) - 1
    dpc_counts = {dipeptide: 0 for dipeptide in dipeptides}

    for i in range(len(seq) - 1):
        dipeptide = seq[i:i + 2]
        if dipeptide in dpc_counts:
            dpc_counts[dipeptide] += 1

    return {k: v / total for k, v in dpc_counts.items() if total > 0}


# 处理FASTA文件
def process_fasta(file_path):
    sequences = []
    ids = []
    for record in SeqIO.parse(file_path, "fasta"):
        features = extract_features(record.seq)
        sequences.append(features)
        ids.append(record.id)
    return np.array(sequences, dtype=object), ids


# 主函数
def main(training_file, input_fasta):
    # 训练模型
    print("Training the model...")
    model = train_model(training_file)

    # 处理输入FASTA文件
    print("Processing input FASTA file...")
    features, ids = process_fasta(input_fasta)

    # 转换特征为DataFrame，防止特征不一致
    features_df = pd.DataFrame(features)

    # 预测
    print("Predicting...")
    predictions = model.predict(features_df)

    # 输出结果
    results = pd.DataFrame({"Protein": ids, "Prediction": predictions})
    print("Results:")
    print(results)
    results.to_csv("predicted_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify antifreeze proteins from non-antifreeze proteins")
    parser.add_argument("training_file", type=str, help="Path to the training CSV file")
    parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file")
    args = parser.parse_args()

    main(args.training_file, args.input_fasta)

