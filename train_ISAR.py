import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import math
from scipy.optimize import fsolve
from npeet import entropy_estimators as ee
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
print("=================================================================")
print("      ISAR 图像识别网络：暗室 + 外场合并后的严格 10 折交叉验证")
print("=================================================================")

# ================= 1. 将暗室和外场图像一次性载入内存 =================
print("\n>>> 正在从文件夹读取暗室与外场 ISAR 图像数据进入内存...")

# 你需要根据自己的真实路径修改这两个路径
anechoic_data_dir = '/data/image/image_0303'
outdoor_data_dir  = '/data/image/image_0609'

datagen = ImageDataGenerator(rescale=1./255)

# ---------- 读取暗室图像 ----------
print("\n>>> 正在读取暗室 ISAR 图像数据...")

anechoic_generator = datagen.flow_from_directory(
    anechoic_data_dir,
    target_size=(256, 256),
    batch_size=200,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

X_anechoic, Y_anechoic = next(anechoic_generator)
L_anechoic = np.argmax(Y_anechoic, axis=1)

print(f"暗室数据载入完成！X_anechoic shape: {X_anechoic.shape}, Y_anechoic shape: {Y_anechoic.shape}")
print("暗室类别顺序:", anechoic_generator.class_indices)

# ---------- 读取外场图像 ----------
print("\n>>> 正在读取外场 ISAR 图像数据...")

outdoor_generator = datagen.flow_from_directory(
    outdoor_data_dir,
    target_size=(256, 256),
    batch_size=200,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

X_outdoor, Y_outdoor = next(outdoor_generator)
L_outdoor = np.argmax(Y_outdoor, axis=1)

print(f"外场数据载入完成！X_outdoor shape: {X_outdoor.shape}, Y_outdoor shape: {Y_outdoor.shape}")
print("外场类别顺序:", outdoor_generator.class_indices)

# ---------- 检查暗室和外场类别顺序是否一致 ----------
if anechoic_generator.class_indices != outdoor_generator.class_indices:
    raise ValueError("暗室和外场数据的类别文件夹顺序不一致，请检查文件夹名称和类别顺序！")

# ---------- 合并暗室和外场图像 ----------
X_all = np.vstack((X_anechoic, X_outdoor))
Y_all = np.vstack((Y_anechoic, Y_outdoor))

# 真实类别标签：0,1,2,3
L_all = np.concatenate((L_anechoic, L_outdoor))

# 域标签：0表示暗室，1表示外场
D_anechoic = np.zeros(len(X_anechoic), dtype=np.int64)
D_outdoor  = np.ones(len(X_outdoor), dtype=np.int64)
D_all = np.concatenate((D_anechoic, D_outdoor))

# 用于分层10折的标签：类别 + 域
# 0~3：暗室四类；4~7：外场四类
split_labels = L_all + 4 * D_all

print("\n>>> 暗室 + 外场 ISAR 图像数据合并完成")
print(f"X_all shape: {X_all.shape}, Y_all shape: {Y_all.shape}")
print(f"L_all shape: {L_all.shape}, D_all shape: {D_all.shape}")

print("合并后类别分布:", np.bincount(L_all, minlength=4))
print("合并后域分布: 暗室 =", np.sum(D_all == 0), ", 外场 =", np.sum(D_all == 1))
print("分层标签分布:", np.bincount(split_labels, minlength=8))


# ================= 2. 参数与保存路径设置 =================
nb_epoch = 50
batchsize = 16
loca = str(time.strftime('%Y-%m-%d-%H-%M-%S'))

base_save_dir = f'NN_10折训练/ISAR-MixedCV-{loca}'
os.makedirs(base_save_dir, exist_ok=True)


# ================= 3. 封装 2D CNN 模型 =================
def create_isar_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(192, (3, 3), activation='relu'),
        layers.MaxPooling2D((8, 8)),

        layers.Flatten(),
        layers.Dense(128, activation='relu', name='feature_layer'),
        layers.Dense(15, activation='relu', name='fc1'),
        layers.Dense(4, activation='softmax', name='output')
    ])

    adam = Adam(learning_rate=1e-3)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ================= 3.5 互信息与 Fano-derived reference 计算函数 =================
def extract_feature_fc1(model, X):
    """
    提取 ISAR 网络的15维深层特征。
    对当前网络结构：
    Dense(128, name='feature_layer') -> Dense(15, name='fc1') -> Dense(4, softmax)
    这里提取 fc1 层输出，即15维特征。
    """
    _ = model.predict(X[:1], verbose=0)

    feat_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('fc1').output
    )

    F = feat_extractor.predict(X, verbose=0)
    return F


def calc_micd_bits(F, L, k=3, standardize=True):
    """
    计算连续深层特征 F 与离散类别标签 L 之间的互信息 I(L;f)。
    注意：npeet 的 ee.micd 默认 base=2，因此输出单位已经是 bits。
    """
    L = np.asarray(L).reshape(-1, 1)

    if standardize:
        F = StandardScaler().fit_transform(F)

    I_bits = ee.micd(F, L, k=k)

    # 互信息不应为负
    I_bits = max(0.0, I_bits)

    return I_bits


def strict_fano_bound(I_bits, num_classes=4):
    """
    根据 I(L;f) 计算 Fano-derived error reference。
    返回错误率参考下界 Pe_bound。

    四分类任务中 H(L)=log2(4)=2 bits，因此 I(L;f) 理论上不应超过2 bits。
    KSG有限样本估计可能略超过理论上限，所以这里截断到合法范围。
    """
    H_L = math.log2(num_classes)

    # 截断到合法互信息范围
    I_bits = min(max(I_bits, 0.0), H_L)

    H_L_given_f = H_L - I_bits

    if H_L_given_f <= 1e-6:
        return 0.0

    def fano_eq(pe):
        if pe <= 0 or pe >= 1:
            return 1e5

        h_pe = -pe * math.log2(pe) - (1 - pe) * math.log2(1 - pe)
        return h_pe + pe * math.log2(num_classes - 1) - H_L_given_f

    try:
        pe_bound = fsolve(fano_eq, 0.01)[0]
        pe_bound = max(0.0, min(1.0, pe_bound))
    except Exception:
        pe_bound = np.nan

    return pe_bound

# ================= 4. 执行严格 10 折交叉验证 =================
print("\n>>> 开始执行暗室 + 外场合并后的严格 10 折交叉验证...")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_train_acc, all_val_acc = [], []
all_train_loss, all_val_loss = [], []

test_accuracies, test_losses = [], []
test_acc_anechoic_list, test_acc_outdoor_list = [], []

# ================== 新增：每折 ISAR 特征互信息与 Fano reference ==================
mi_feature_list = []
fano_error_bound_list = []
fano_acc_reference_list = []

# ================== 新增：保存每折验证集预测结果，用于混淆矩阵 ==================
all_y_true = []
all_y_pred = []
all_fold_ids = []
all_confusion_matrices = []

# 根据 flow_from_directory 的类别顺序自动生成类别名称
# 这样能保证混淆矩阵的类别顺序和模型标签一致
class_names = [None] * len(anechoic_generator.class_indices)
for class_name, class_idx in anechoic_generator.class_indices.items():
    class_names[class_idx] = class_name

print("混淆矩阵类别顺序:", class_names)

fold_no = 1

# 注意：
# 这里 split 的是一张张 ISAR 图像。
# 对ISAR而言，一张图像就是一个group，因此这里是 image-level group-wise 10-fold。
# split_labels 同时考虑类别和采集域，保证每折测试集同时包含暗室和外场的四类图像。
for train_index, test_index in skf.split(X_all, split_labels):

    print(f"\n==================================================")
    print(f"   ISAR Training Fold {fold_no} / 10")
    print(f"==================================================")

    X_train, X_test = X_all[train_index], X_all[test_index]
    Y_train, Y_test = Y_all[train_index], Y_all[test_index]

    L_train, L_test = L_all[train_index], L_all[test_index]
    D_train, D_test = D_all[train_index], D_all[test_index]

    print(f"-> 图像级划分: 训练图像 {len(X_train)} 张, 测试图像 {len(X_test)} 张")
    print(f"-> 训练集类别分布: {np.bincount(L_train, minlength=4)}")
    print(f"-> 测试集类别分布: {np.bincount(L_test, minlength=4)}")
    print(f"-> 训练集域分布: 暗室 {np.sum(D_train == 0)} 张, 外场 {np.sum(D_train == 1)} 张")
    print(f"-> 测试集域分布: 暗室 {np.sum(D_test == 0)} 张, 外场 {np.sum(D_test == 1)} 张")

    model = create_isar_model()

    history = model.fit(
        X_train, Y_train,
        epochs=nb_epoch,
        batch_size=batchsize,
        verbose=1,
        validation_data=(X_test, Y_test)
    )

    # ---------- 整体测试集评估 ----------
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(f">> Fold {fold_no} 总测试集 -> Loss: {scores[0]:.4f}, Accuracy: {scores[1]*100:.2f}%")

    test_losses.append(scores[0])
    test_accuracies.append(scores[1] * 100)
    
        # ---------- 新增：保存该折验证集预测结果，并计算该折混淆矩阵 ----------
    Y_pred_prob = model.predict(X_test, verbose=0)

    y_true_fold = np.argmax(Y_test, axis=1)
    y_pred_fold = np.argmax(Y_pred_prob, axis=1)

    # 保存该折预测结果，后面用于10折合并总混淆矩阵
    all_y_true.append(y_true_fold)
    all_y_pred.append(y_pred_fold)
    all_fold_ids.append(np.ones_like(y_true_fold) * fold_no)

    # 计算该折混淆矩阵
    cm_fold = confusion_matrix(y_true_fold, y_pred_fold, labels=[0, 1, 2, 3])
    all_confusion_matrices.append(cm_fold)

    print(f">> Fold {fold_no} confusion matrix counts:")
    print(cm_fold)

    # 保存该折混淆矩阵计数版
    pd.DataFrame(
        cm_fold,
        index=[f"True_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names]
    ).to_csv(
        os.path.join(base_save_dir, f"Confusion_Matrix_Fold_{fold_no}_Counts.csv")
    )

    # 保存该折混淆矩阵归一化版
    cm_fold_norm = cm_fold.astype(float) / np.maximum(cm_fold.sum(axis=1, keepdims=True), 1)

    pd.DataFrame(
        cm_fold_norm,
        index=[f"True_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names]
    ).to_csv(
        os.path.join(base_save_dir, f"Confusion_Matrix_Fold_{fold_no}_Normalized.csv")
    )
    
        # ---------- 估计该折 held-out 测试集上的 I(L;f) 和 Fano reference ----------
    # 注意：这里仅使用该折 X_test 提取特征，避免训练集样本进入MI估计
    L_test_mi = np.argmax(Y_all, axis=1).reshape(-1, 1)

    # 提取该折测试集的15维深层特征
    F_test = extract_feature_fc1(model, X_all)

    # KSG估计 I(L;f)，单位 bits
    # ISAR每折测试图像数量较少，KSG会有一定波动，这是正常的
    I_L_f_bits = calc_micd_bits(F_test, L_test_mi, k=3, standardize=True)

    # 用于Fano计算的互信息限制在 [0, log2(4)]
    I_L_f_bits_for_fano = min(max(I_L_f_bits, 0.0), math.log2(4))

    # Fano-derived error reference
    Pe_bound = strict_fano_bound(I_L_f_bits_for_fano, num_classes=4)

    # Fano-derived accuracy reference
    Acc_reference = 1.0 - Pe_bound

    print(f">> Fold {fold_no} ISAR特征互信息 I(L;f): {I_L_f_bits:.4f} bits")
    print(f">> Fold {fold_no} Fano-derived error reference: {Pe_bound*100:.2f}%")
    print(f">> Fold {fold_no} Fano-derived accuracy reference: {Acc_reference*100:.2f}%")

    mi_feature_list.append(I_L_f_bits)
    fano_error_bound_list.append(Pe_bound * 100)
    fano_acc_reference_list.append(Acc_reference * 100)

    # ---------- 分域测试精度：暗室测试图像 / 外场测试图像 ----------
    idx_test_anechoic = np.where(D_test == 0)[0]
    idx_test_outdoor  = np.where(D_test == 1)[0]

    scores_anechoic = model.evaluate(
        X_test[idx_test_anechoic],
        Y_test[idx_test_anechoic],
        verbose=0
    )

    scores_outdoor = model.evaluate(
        X_test[idx_test_outdoor],
        Y_test[idx_test_outdoor],
        verbose=0
    )

    acc_anechoic = scores_anechoic[1] * 100
    acc_outdoor = scores_outdoor[1] * 100

    print(f">> Fold {fold_no} 暗室测试子集 -> Accuracy: {acc_anechoic:.2f}%")
    print(f">> Fold {fold_no} 外场测试子集 -> Accuracy: {acc_outdoor:.2f}%")

    test_acc_anechoic_list.append(acc_anechoic)
    test_acc_outdoor_list.append(acc_outdoor)

    # ---------- 保存历史曲线数据 ----------
    all_train_acc.append(history.history['accuracy'])
    all_val_acc.append(history.history['val_accuracy'])
    all_train_loss.append(history.history['loss'])
    all_val_loss.append(history.history['val_loss'])

    # ---------- 保存当前折模型 ----------
    model.save(os.path.join(base_save_dir, f'ISAR_model_fold{fold_no}.h5'))

    fold_no += 1

# ================= 4.5 计算并保存10折合并后的总混淆矩阵 =================
print("\n==================================================")
print("     计算10折合并后的 ISAR held-out confusion matrix")
print("==================================================")

# 合并10折所有held-out预测结果
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
fold_ids_all = np.concatenate(all_fold_ids)

print(f"10折合并后验证样本数: {len(y_true_all)}")
print(f"真实类别分布: {np.bincount(y_true_all, minlength=4)}")
print(f"预测类别分布: {np.bincount(y_pred_all, minlength=4)}")

# 总混淆矩阵：计数版
cm_total = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3])

# 总混淆矩阵：按真实类别行归一化
cm_total_norm = cm_total.astype(float) / np.maximum(cm_total.sum(axis=1, keepdims=True), 1)

print("10折合并 confusion matrix counts:")
print(cm_total)

print("10折合并 confusion matrix normalized:")
print(np.round(cm_total_norm * 100, 2))

# 保存每个样本的真实标签和预测标签，方便后续复查
pd.DataFrame({
    "Fold": fold_ids_all,
    "True_Label": y_true_all,
    "Pred_Label": y_pred_all
}).to_csv(
    os.path.join(base_save_dir, "OOF_Predictions_For_Confusion_Matrix.csv"),
    index=False
)

# 保存总混淆矩阵计数版
pd.DataFrame(
    cm_total,
    index=[f"True_{name}" for name in class_names],
    columns=[f"Pred_{name}" for name in class_names]
).to_csv(
    os.path.join(base_save_dir, "Confusion_Matrix_Total_Counts.csv")
)

# 保存总混淆矩阵归一化版
pd.DataFrame(
    cm_total_norm,
    index=[f"True_{name}" for name in class_names],
    columns=[f"Pred_{name}" for name in class_names]
).to_csv(
    os.path.join(base_save_dir, "Confusion_Matrix_Total_Normalized.csv")
)

print("已保存：")
print("1. OOF_Predictions_For_Confusion_Matrix.csv")
print("2. Confusion_Matrix_Total_Counts.csv")
print("3. Confusion_Matrix_Total_Normalized.csv")
print("4. Confusion_Matrix_Fold_*_Counts.csv")
print("5. Confusion_Matrix_Fold_*_Normalized.csv")
print("==================================================")

# ================= 4.6 绘制10折合并后的总混淆矩阵图 =================

plt.figure(figsize=(5, 4))
im = plt.imshow(cm_total_norm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(np.arange(4), class_names, rotation=45, ha="right")
plt.yticks(np.arange(4), class_names)

plt.xlabel("Predicted label")
plt.ylabel("True label")

for i in range(4):
    for j in range(4):
        value_percent = cm_total_norm[i, j] * 100
        value_count = cm_total[i, j]
        text_color = "white" if cm_total_norm[i, j] > 0.5 else "black"

        plt.text(
            j, i,
            f"{value_percent:.1f}%\n({value_count})",
            ha="center",
            va="center",
            color=text_color,
            fontsize=9
        )

plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, "Confusion_Matrix_Total_Normalized.png"), dpi=600)
plt.savefig(os.path.join(base_save_dir, "Confusion_Matrix_Total_Normalized.pdf"))
plt.close()


# 计数版总混淆矩阵
plt.figure(figsize=(5, 4))
im = plt.imshow(cm_total, interpolation="nearest", cmap="Blues")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(np.arange(4), class_names, rotation=45, ha="right")
plt.yticks(np.arange(4), class_names)

plt.xlabel("Predicted label")
plt.ylabel("True label")

for i in range(4):
    for j in range(4):
        text_color = "white" if cm_total[i, j] > cm_total.max() / 2 else "black"

        plt.text(
            j, i,
            f"{cm_total[i, j]}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=10
        )

plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, "Confusion_Matrix_Total_Counts.png"), dpi=600)
plt.savefig(os.path.join(base_save_dir, "Confusion_Matrix_Total_Counts.pdf"))
plt.close()

print("已保存总混淆矩阵图：")
print("1. Confusion_Matrix_Total_Normalized.png / .pdf")
print("2. Confusion_Matrix_Total_Counts.png / .pdf")
# ================= 5. 保存汇总矩阵与评估报告 =================
print("\n==================================================")
print("          ISAR 暗室 + 外场合并数据 10 折交叉验证报告")
print("==================================================")

mean_acc = np.mean(test_accuracies)
std_acc = np.std(test_accuracies)
mean_loss = np.mean(test_losses)

mean_acc_anechoic = np.mean(test_acc_anechoic_list)
std_acc_anechoic = np.std(test_acc_anechoic_list)

mean_acc_outdoor = np.mean(test_acc_outdoor_list)
std_acc_outdoor = np.std(test_acc_outdoor_list)

# ================== 新增：MI与Fano统计 ==================
mean_mi = np.mean(mi_feature_list)
std_mi = np.std(mi_feature_list)

mean_fano_error = np.mean(fano_error_bound_list)
std_fano_error = np.std(fano_error_bound_list)

mean_fano_acc_reference = np.mean(fano_acc_reference_list)
std_fano_acc_reference = np.std(fano_acc_reference_list)

print(f"总测试集各折准确率: {[f'{acc:.2f}%' for acc in test_accuracies]}")
print(f"总测试集平均准确率: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
print(f"总测试集平均 Loss : {mean_loss:.4f}")

print(f"暗室测试子集各折准确率: {[f'{acc:.2f}%' for acc in test_acc_anechoic_list]}")
print(f"暗室测试子集平均准确率: {mean_acc_anechoic:.2f}% (+/- {std_acc_anechoic:.2f}%)")

print(f"外场测试子集各折准确率: {[f'{acc:.2f}%' for acc in test_acc_outdoor_list]}")
print(f"外场测试子集平均准确率: {mean_acc_outdoor:.2f}% (+/- {std_acc_outdoor:.2f}%)")
print("==================================================")
print(f"各折特征互信息 I(L;f): {[f'{mi:.4f}' for mi in mi_feature_list]}")
print(f"平均特征互信息 I(L;f): {mean_mi:.4f} bits (+/- {std_mi:.4f})")

print(f"各折 Fano-derived 错误率参考值: {[f'{pe:.2f}%' for pe in fano_error_bound_list]}")
print(f"平均 Fano-derived 错误率参考值: {mean_fano_error:.2f}% (+/- {std_fano_error:.2f}%)")

print(f"各折 Fano-derived 准确率参考值: {[f'{acc:.2f}%' for acc in fano_acc_reference_list]}")
print(f"平均 Fano-derived 准确率参考值: {mean_fano_acc_reference:.2f}% (+/- {std_fano_acc_reference:.2f}%)")

# 保存 Test 结果
with open(os.path.join(base_save_dir, "ISAR_Mixed_Anechoic_Outdoor_10Fold_Results.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Fold",
        "Test Accuracy (%)",
        "Test Loss",
        "Anechoic Test Accuracy (%)",
        "Outdoor Test Accuracy (%)",
        "I(L;f) bits",
        "Fano Error Reference (%)",
        "Fano Accuracy Reference (%)"
    ])

    for i in range(10):
        writer.writerow([
            i+1,
            test_accuracies[i],
            test_losses[i],
            test_acc_anechoic_list[i],
            test_acc_outdoor_list[i],
            mi_feature_list[i],
            fano_error_bound_list[i],
            fano_acc_reference_list[i]
        ])

    writer.writerow([
        "Mean",
        mean_acc,
        mean_loss,
        mean_acc_anechoic,
        mean_acc_outdoor,
        mean_mi,
        mean_fano_error,
        mean_fano_acc_reference
    ])

    writer.writerow([
        "Std",
        std_acc,
        "N/A",
        std_acc_anechoic,
        std_acc_outdoor,
        std_mi,
        std_fano_error,
        std_fano_acc_reference
    ])

# 保存给 MATLAB 绘图的汇总矩阵
pd.DataFrame(
    np.array(all_train_acc).T,
    columns=[f'Fold_{i+1}' for i in range(10)]
).to_csv(os.path.join(base_save_dir, 'CV_Matrix_Train_Acc.csv'), index=False)

pd.DataFrame(
    np.array(all_val_acc).T,
    columns=[f'Fold_{i+1}' for i in range(10)]
).to_csv(os.path.join(base_save_dir, 'CV_Matrix_Val_Acc.csv'), index=False)

pd.DataFrame(
    np.array(all_train_loss).T,
    columns=[f'Fold_{i+1}' for i in range(10)]
).to_csv(os.path.join(base_save_dir, 'CV_Matrix_Train_Loss.csv'), index=False)

pd.DataFrame(
    np.array(all_val_loss).T,
    columns=[f'Fold_{i+1}' for i in range(10)]
).to_csv(os.path.join(base_save_dir, 'CV_Matrix_Val_Loss.csv'), index=False)

# 保存每折 ISAR MI + Fano 结果
pd.DataFrame({
    "Fold": np.arange(1, 11),
    "Test_Accuracy_percent": test_accuracies,
    "Test_Loss": test_losses,
    "Anechoic_Test_Accuracy_percent": test_acc_anechoic_list,
    "Outdoor_Test_Accuracy_percent": test_acc_outdoor_list,
    "I_L_f_bits": mi_feature_list,
    "Fano_Error_Reference_percent": fano_error_bound_list,
    "Fano_Accuracy_Reference_percent": fano_acc_reference_list
}).to_csv(os.path.join(base_save_dir, 'ISAR_MI_Fano_PerFold_Results.csv'), index=False)

print(f"所有 ISAR 模型与图表矩阵已全部保存至: {base_save_dir}")