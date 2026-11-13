#########################################   NN ##################################################
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import csv
import os
from tensorflow.keras import regularizers
from sklearn.manifold import TSNE
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Activation, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import keras
import math
from scipy.optimize import fsolve
from npeet import entropy_estimators as ee
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

print('开始读取数据')

type='jump'
data='500'
bg='nobg'
test='True'
num_total=500

# =========================
# 1. 路径设置
dev_path = f'/data/physical_waveforms_lfm'

# =============================================================================
# 一、读取数据，作为 X_all / Y_all
# =============================================================================
print('开始读取数据')

data_target1_pad=np.zeros([num_total,1,100000])
data_target1=pd.read_csv(f'{dev_path}/j16-{bg}-{data}.csv', header=None)
data_target1=np.mat(data_target1)
data_target1=np.array(data_target1)
a=data_target1.max(axis=1)
a = a.reshape(data_target1.shape[0], 1)
a[a == 0] = 1
data_target1 = data_target1 / a
data_target1=data_target1.reshape(num_total,80000,1)
print(data_target1.shape)

data_target2_pad=np.zeros([num_total,1,100000])
data_target2=pd.read_csv(f'{dev_path}/j20-{bg}-{data}.csv', header=None)
data_target2=np.mat(data_target2)
data_target2=np.array(data_target2)
a=data_target2.max(axis=1)
a = a.reshape(data_target2.shape[0], 1)
a[a == 0] = 1
data_target2 = data_target2 / a
data_target2=data_target2.reshape(num_total,80000,1)
print(data_target2.shape)

data_target3_pad=np.zeros([num_total,1,100000])
data_target3=pd.read_csv(f'{dev_path}/plane-{bg}-{data}.csv', header=None)
data_target3=np.mat(data_target3)
data_target3=np.array(data_target3)
a=data_target3.max(axis=1)
a = a.reshape(data_target3.shape[0], 1)
a[a == 0] = 1
data_target3 = data_target3 / a
data_target3=data_target3.reshape(num_total,80000,1)
print(data_target3.shape)

data_target4_pad=np.zeros([num_total,1,100000])
data_target4=pd.read_csv(f'{dev_path}/uav-{bg}-{data}.csv', header=None)
data_target4=np.mat(data_target4)
data_target4=np.array(data_target4)
a=data_target4.max(axis=1)
a = a.reshape(data_target4.shape[0], 1)
a[a == 0] = 1
data_target4 = data_target4 / a
data_target4=data_target4.reshape(num_total,80000,1)
print(data_target4.shape)


X_all = np.vstack((data_target1[::1,:,:], data_target2[::1,:,:], data_target3[::1,:,:], data_target4[::1,:,:]))

Y_target1 = np.array([[1, 0, 0, 0]] * 500)
Y_target2 = np.array([[0, 1, 0, 0]] * 500)
Y_target3 = np.array([[0, 0, 1, 0]] * 500)
Y_target4 = np.array([[0, 0, 0, 1]] * 500)

Y_all = np.vstack((Y_target1, Y_target2, Y_target3, Y_target4))

pulses_per_target = 5

X_all_mix = X_all
Y_all_mix = Y_all

num_targets = 400
target_ids = np.arange(num_targets)

# 真实目标类别标签：仍然是四分类，用于统计类别
target_labels = np.array(
    [0]*50 + [1]*50 + [2]*50 + [3]*50 +
    [0]*50 + [1]*50 + [2]*50 + [3]*50
)

# 域标签
target_domain_labels = np.array([0]*200 + [1]*200)

# 用于StratifiedKFold的分层标签
# 这样可以保证每一折验证集中，四类目标都比较均衡
target_split_labels = target_labels + 4 * target_domain_labels

# 定义映射函数：输入 target ID 列表，输出它们对应的所有 echo 索引
def get_pulse_indices(selected_target_ids, pulses_per_target=10):
    indices = []
    for tid in selected_target_ids:
        indices.extend(range(tid * pulses_per_target, (tid + 1) * pulses_per_target))
    return np.array(indices)
# ================= 3. 参数与路径设置 =================
nb_epoch = 50
batchsize = 32
loca = str(time.strftime('%Y-%m-%d-%H-%M-%S'))

base_save_dir = f'NN_10折训练/NN-{type}-{loca}-abshrrp-out15-mix-9train'
os.makedirs(base_save_dir, exist_ok=True)

# ================= 3.5 互信息与 Fano-derived lower bound 计算函数 =================
def extract_feature_fc1(model, X):
    """
    提取倒数第二层特征。
    对你的网络结构：
    Dense(128) -> relu(fc2) -> Dense(15) -> relu(fc1) -> Dense(4) -> softmax
    这里提取 fc1 层输出，即 15维深层特征。
    """
    _ = model.predict(X[:1], verbose=0)

    feat_extractor = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('fc1').output
    )

    F = feat_extractor.predict(X, verbose=0)
    return F


def calc_micd_bits(F, L, k=3, standardize=True):
    """
    计算连续深层特征 F 与离散类别标签 L 之间的互信息 I(L; f)。
    ee.micd 输出单位是 nats，这里转换为 bits。
    """
    L = np.asarray(L).reshape(-1, 1)

    if standardize:
        F = StandardScaler().fit_transform(F)

    I_bits = ee.micd(F, L, k=k)

    # 互信息不应为负
    I_bits = max(0.0, I_bits)
    I_bits=min(I_bits,2)

    return I_bits


def strict_fano_bound(I_bits, num_classes=4):
    """
    根据 I(L;f) 计算 Fano-derived error lower bound。
    返回的是错误率下界 Pe_bound。

    对四分类任务：
    H(L) = log2(4) = 2 bits
    因此 I(L;f) 理论上不应超过 2 bits。
    KSG 有有限样本估计误差，所以这里做截断。
    """
    H_L = math.log2(num_classes)

    # 防止 KSG 估计值超过理论上限
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
        pe_bound = fsolve(fano_eq, 0.001)[0]
        pe_bound = max(0.0, min(1.0, pe_bound))
    except Exception:
        pe_bound = np.nan

    return pe_bound

# ================= 4. 封装 1D CNN 模型 =================
def create_phy_model():
    model = Sequential()
    model.add(Conv1D(filters=4, kernel_size=3, padding='same', strides=2, 
                            input_shape=(80000,1), data_format="channels_last", name='conv2d', use_bias=True))
    model.add(Activation('relu', name='conv1'))
    # model.add(LeakyReLU(alpha=0.1, name='conv1_leaky'))
    model.add(Conv1D(filters=4, kernel_size=3, padding='same', strides=2, 
                            data_format="channels_last", name='conv2d_4', use_bias=True))
    model.add(Activation('relu', name='conv4'))
    # model.add(LeakyReLU(alpha=0.1, name='conv2_leaky'))
    model.add(MaxPooling1D(pool_size=8, data_format="channels_last", padding="same", name='pool2'))
    model.add(Flatten(data_format="channels_last"))
    model.add(Dense(128))
    model.add(Activation('relu', name='fc2'))
    # model.add(LeakyReLU(alpha=0.1, name='conv3_leaky'))
    model.add(Dense(15))
    model.add(Activation('relu', name='fc1'))
    # model.add(LeakyReLU(alpha=0.1, name='conv4_leaky'))
    model.add(Dense(4, name='fc3'))
    model.add(Activation('softmax', name='output'))
    
    # adam = Adam(learning_rate=1e-4,clipnorm=1.0)
    adam = Adam(learning_rate=1e-3)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ================= 5. 执行 group-wise 10 折交叉验证 =================

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_train_acc, all_val_acc = [], []
all_train_loss, all_val_loss = [], []

test_accuracies, test_losses = [], []
mi_feature_list = []
fano_error_bound_list = []
fano_acc_upper_bound_list = []

all_y_true = []
all_y_pred = []
all_fold_ids = []
all_confusion_matrices = []

fold_no = 1

# 注意：
# 这里 split 的是 400 个 target/group，而不是 4000 条 echo。
for train_target_indices, test_target_indices in skf.split(target_ids, target_split_labels):
# for  test_target_indices,train_target_indices in skf.split(target_ids, target_split_labels):
    print(f"\n==================================================")
    print(f"   Training Fold {fold_no} / 10")
    print(f"==================================================")
    
    # 1. 训练 / 测试 group ID
    train_ids = target_ids[train_target_indices]
    test_ids  = target_ids[test_target_indices]
    
    print(f"-> group级划分: 训练 group {len(train_ids)} 个, 测试 group {len(test_ids)} 个")
    
    train_domain = target_domain_labels[train_ids]
    test_domain  = target_domain_labels[test_ids]

    
    # 打印类别分布，方便检查
    print(f"-> 训练集类别分布: {np.bincount(target_labels[train_ids], minlength=4)}")
    print(f"-> 测试集类别分布: {np.bincount(target_labels[test_ids], minlength=4)}")
    
    # 2. 将 target ID 映射为实际 echo 索引
    idx_train = get_pulse_indices(train_ids, pulses_per_target)
    idx_test  = get_pulse_indices(test_ids, pulses_per_target)
    
    # 3. 根据索引获取真实数据
    X_train, Y_train = X_all_mix[idx_train], Y_all_mix[idx_train]
    X_test, Y_test   = X_all_mix[idx_test],  Y_all_mix[idx_test]
    
    print(f"-> echo级数据: 训练 echo {len(X_train)} 个, 测试 echo {len(X_test)} 个")
    
    # ================== 回调函数，可以按你原来设置选择是否打开 ==================
    # reduce_lr = ReduceLROnPlateau(
    #     monitor='val_loss', 
    #     factor=0.5, 
    #     patience=5, 
    #     min_lr=1e-6, 
    #     verbose=1
    # )
    
    # early_stop = EarlyStopping(
    #     monitor='val_loss', 
    #     patience=15, 
    #     restore_best_weights=True, 
    #     verbose=1
    # )
    # ====================================================================
    
    # 4. 开始训练
    model = create_phy_model()
    
    history = model.fit(
        X_train, Y_train,
        epochs=nb_epoch,
        batch_size=batchsize,
        verbose=1,
        validation_data=(X_test, Y_test),
        # callbacks=[reduce_lr, early_stop]
    )
    
    # 5. 测试集评估
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(f">> Fold {fold_no} 测试集 -> Loss: {scores[0]:.4f}, Accuracy: {scores[1]*100:.2f}%")
    
    test_losses.append(scores[0])
    test_accuracies.append(scores[1] * 100)
    # ================== 5.1 保存该折held-out预测结果，用于最终合并混淆矩阵 ==================
    Y_pred_prob = model.predict(X_test, verbose=0)

    y_true_fold = np.argmax(Y_test, axis=1)
    y_pred_fold = np.argmax(Y_pred_prob, axis=1)

    all_y_true.append(y_true_fold)
    all_y_pred.append(y_pred_fold)
    all_fold_ids.append(np.ones_like(y_true_fold) * fold_no)

    # 保存该折自己的混淆矩阵，后续可选用
#     cm_fold = confusion_matrix(y_true_fold, y_pred_fold, labels=[0, 1, 2, 3])
#     all_confusion_matrices.append(cm_fold)

#     print(f">> Fold {fold_no} confusion matrix:")
#     print(cm_fold)
        # ================== 5.5 计算该折验证/测试集上的 I(L; f) 和 Fano 下界 ==================
    # 真实类别标签 L
    
    # L_test = np.argmax(Y_test, axis=1).reshape(-1, 1)
    L_test = np.argmax(Y_all_mix, axis=1).reshape(-1, 1)

    # 提取该折测试集的倒数第二层深层特征 f
    # F_test = extract_feature_fc1(model, X_test)
    F_test = extract_feature_fc1(model, X_all_mix)
    
    # 计算 I(L; f)，单位 bits
    I_L_f_bits = calc_micd_bits(F_test, L_test, k=3, standardize=False)

    # 用于 Fano 计算的互信息需要限制在 [0, log2(4)]
    I_L_f_bits_for_fano = min(max(I_L_f_bits, 0.0), math.log2(4))

    # Fano-derived error lower bound
    Pe_bound = strict_fano_bound(I_L_f_bits_for_fano, num_classes=4)

    # 对应的准确率参考上界
    Acc_upper_bound = 1.0 - Pe_bound

    print(f">> Fold {fold_no} 特征互信息 I(L;f): {I_L_f_bits:.4f} bits")
    print(f">> Fold {fold_no} Fano-derived error lower bound: {Pe_bound*100:.2f}%")
    print(f">> Fold {fold_no} Fano-derived accuracy upper bound: {Acc_upper_bound*100:.2f}%")

    mi_feature_list.append(I_L_f_bits)
    fano_error_bound_list.append(Pe_bound * 100)
    fano_acc_upper_bound_list.append(Acc_upper_bound * 100)
    # 6. 保存该折的训练历史
    all_train_acc.append(history.history['accuracy'])
    all_val_acc.append(history.history['val_accuracy'])
    all_train_loss.append(history.history['loss'])
    all_val_loss.append(history.history['val_loss'])
    
    # 7. 保存模型
    model.save(os.path.join(base_save_dir, f'Phy_model_fold{fold_no}.h5'))
    
    fold_no += 1

 # ================= 5.9 计算并保存10折合并后的held-out混淆矩阵 =================
print("\n==================================================")
print("     计算10折合并后的 held-out confusion matrix")
print("==================================================")

# 合并10折所有held-out预测
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
fold_ids_all = np.concatenate(all_fold_ids)

print(f"合并后样本数: {len(y_true_all)}")
print(f"真实类别分布: {np.bincount(y_true_all, minlength=4)}")
print(f"预测类别分布: {np.bincount(y_pred_all, minlength=4)}")

# 原始计数混淆矩阵
cm_total = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3])

# # 行归一化混淆矩阵：每一行除以该真实类别样本总数
# cm_total_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

print("10折合并 confusion matrix 计数:")
print(cm_total)

# print("10折合并 confusion matrix 行归一化:")
# print(cm_total_norm)

# 保存原始预测标签，方便复查
pd.DataFrame({
    "Fold": fold_ids_all,
    "True_Label": y_true_all,
    "Pred_Label": y_pred_all
}).to_csv(os.path.join(base_save_dir, "OOF_Predictions_For_Confusion_Matrix.csv"), index=False)

# 保存混淆矩阵
class_names = ["Class_0", "Class_1", "Class_2", "Class_3"]

pd.DataFrame(
    cm_total,
    index=[f"True_{name}" for name in class_names],
    columns=[f"Pred_{name}" for name in class_names]
).to_csv(os.path.join(base_save_dir, "Confusion_Matrix_Counts.csv"))

# pd.DataFrame(
#     cm_total_norm,
#     index=[f"True_{name}" for name in class_names],
#     columns=[f"Pred_{name}" for name in class_names]
# ).to_csv(os.path.join(base_save_dir, "Confusion_Matrix_Normalized.csv"))

# 保存每折混淆矩阵
# for i, cm in enumerate(all_confusion_matrices):
#     pd.DataFrame(
#         cm,
#         index=[f"True_{name}" for name in class_names],
#         columns=[f"Pred_{name}" for name in class_names]
#     ).to_csv(os.path.join(base_save_dir, f"Confusion_Matrix_Fold_{i+1}.csv"))

print("已保存：")
print("1. OOF_Predictions_For_Confusion_Matrix.csv")
print("2. Confusion_Matrix_Counts.csv")
print("3. Confusion_Matrix_Normalized.csv")
print("4. Confusion_Matrix_Fold_1.csv 到 Fold_10.csv")
print("==================================================")
# ================= 5.10 绘制论文用混淆矩阵图 =================

# 这里你可以改成真实类别名称
class_names = ["Drone", "J16", "Plane", "J20"]

# plt.figure(figsize=(5, 4))
# im = plt.imshow(cm_total_norm, interpolation="nearest", cmap="Blues")
# plt.colorbar(im, fraction=0.046, pad=0.04)

# plt.xticks(np.arange(4), class_names, rotation=45, ha="right")
# plt.yticks(np.arange(4), class_names)

# plt.xlabel("Predicted label")
# plt.ylabel("True label")

# # 在格子里标注百分比和样本数
# for i in range(4):
#     for j in range(4):
#         value_percent = cm_total_norm[i, j] * 100
#         value_count = cm_total[i, j]
#         text_color = "white" if cm_total_norm[i, j] > 0.5 else "black"
#         plt.text(
#             j, i,
#             f"{value_percent:.1f}%\n({value_count})",
#             ha="center",
#             va="center",
#             color=text_color,
#             fontsize=9
#         )

# plt.tight_layout()
# plt.savefig(os.path.join(base_save_dir, "Confusion_Matrix_Normalized.png"), dpi=600)
# plt.savefig(os.path.join(base_save_dir, "Confusion_Matrix_Normalized.pdf"))
# plt.close()

# 如果还想画计数版
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
plt.savefig(os.path.join(base_save_dir, "Confusion_Matrix_Counts.png"), dpi=600)
# plt.savefig(os.path.join(base_save_dir, "Confusion_Matrix_Counts.pdf"))
plt.close()

print("已保存论文用混淆矩阵图：")
print("1. Confusion_Matrix_Normalized.png / .pdf")
print("2. Confusion_Matrix_Counts.png / .pdf")

# ================= 5.9 保存每一折训练/验证 accuracy 和 loss 曲线 =================
print("\n==================================================")
print("     保存每一折训练/验证 Accuracy 和 Loss 曲线")
print("==================================================")

# 转成 numpy array，shape = (10, nb_epoch)
all_train_acc = np.array(all_train_acc)
all_val_acc = np.array(all_val_acc)
all_train_loss = np.array(all_train_loss)
all_val_loss = np.array(all_val_loss)

epochs = np.arange(1, nb_epoch + 1)

# 1. 保存为矩阵格式：每一列是一折，方便后续 MATLAB / Origin / Python 画图
train_acc_df = pd.DataFrame({"Epoch": epochs})
val_acc_df = pd.DataFrame({"Epoch": epochs})
train_loss_df = pd.DataFrame({"Epoch": epochs})
val_loss_df = pd.DataFrame({"Epoch": epochs})

for i in range(all_train_acc.shape[0]):
    train_acc_df[f"Fold_{i+1}"] = all_train_acc[i]
    val_acc_df[f"Fold_{i+1}"] = all_val_acc[i]
    train_loss_df[f"Fold_{i+1}"] = all_train_loss[i]
    val_loss_df[f"Fold_{i+1}"] = all_val_loss[i]

train_acc_df.to_csv(os.path.join(base_save_dir, "CV_Matrix_Train_Acc.csv"), index=False)
val_acc_df.to_csv(os.path.join(base_save_dir, "CV_Matrix_Val_Acc.csv"), index=False)
train_loss_df.to_csv(os.path.join(base_save_dir, "CV_Matrix_Train_Loss.csv"), index=False)
val_loss_df.to_csv(os.path.join(base_save_dir, "CV_Matrix_Val_Loss.csv"), index=False)

# 2. 保存为 long format：每一行是 Fold-Epoch，方便 seaborn / pandas / Origin 分组画图
history_records = []

for fold_idx in range(all_train_acc.shape[0]):
    for epoch_idx in range(nb_epoch):
        history_records.append({
            "Fold": fold_idx + 1,
            "Epoch": epoch_idx + 1,
            "Train_Accuracy": all_train_acc[fold_idx, epoch_idx],
            "Val_Accuracy": all_val_acc[fold_idx, epoch_idx],
            "Train_Loss": all_train_loss[fold_idx, epoch_idx],
            "Val_Loss": all_val_loss[fold_idx, epoch_idx]
        })

history_df = pd.DataFrame(history_records)
history_df.to_csv(os.path.join(base_save_dir, "CV_History_Long_Format.csv"), index=False)

# 3. 保存平均曲线和标准差，论文图最常用
mean_train_acc = np.mean(all_train_acc, axis=0)
std_train_acc = np.std(all_train_acc, axis=0)

mean_val_acc = np.mean(all_val_acc, axis=0)
std_val_acc = np.std(all_val_acc, axis=0)

mean_train_loss = np.mean(all_train_loss, axis=0)
std_train_loss = np.std(all_train_loss, axis=0)

mean_val_loss = np.mean(all_val_loss, axis=0)
std_val_loss = np.std(all_val_loss, axis=0)

summary_curve_df = pd.DataFrame({
    "Epoch": epochs,
    "Train_Accuracy_Mean": mean_train_acc,
    "Train_Accuracy_Std": std_train_acc,
    "Val_Accuracy_Mean": mean_val_acc,
    "Val_Accuracy_Std": std_val_acc,
    "Train_Loss_Mean": mean_train_loss,
    "Train_Loss_Std": std_train_loss,
    "Val_Loss_Mean": mean_val_loss,
    "Val_Loss_Std": std_val_loss
})

summary_curve_df.to_csv(os.path.join(base_save_dir, "CV_History_Mean_Std.csv"), index=False)

# 4. 同时保存 npz，方便之后直接 Python 读取
np.savez(
    os.path.join(base_save_dir, "CV_History_All_Folds.npz"),
    epochs=epochs,
    all_train_acc=all_train_acc,
    all_val_acc=all_val_acc,
    all_train_loss=all_train_loss,
    all_val_loss=all_val_loss,
    mean_train_acc=mean_train_acc,
    std_train_acc=std_train_acc,
    mean_val_acc=mean_val_acc,
    std_val_acc=std_val_acc,
    mean_train_loss=mean_train_loss,
    std_train_loss=std_train_loss,
    mean_val_loss=mean_val_loss,
    std_val_loss=std_val_loss
)

print("已保存每折训练曲线：")
print("1. CV_Matrix_Train_Acc.csv")
print("2. CV_Matrix_Val_Acc.csv")
print("3. CV_Matrix_Train_Loss.csv")
print("4. CV_Matrix_Val_Loss.csv")
print("5. CV_History_Long_Format.csv")
print("6. CV_History_Mean_Std.csv")
print("7. CV_History_All_Folds.npz")
print("==================================================")
# ================= 6. 保存汇总矩阵与评估报告 =================

mean_acc = np.mean(test_accuracies)
std_acc = np.std(test_accuracies)
mean_loss = np.mean(test_losses)
mean_mi = np.mean(mi_feature_list)
std_mi = np.std(mi_feature_list)

mean_fano_error = np.mean(fano_error_bound_list)
std_fano_error = np.std(fano_error_bound_list)

mean_fano_acc_upper = np.mean(fano_acc_upper_bound_list)
std_fano_acc_upper = np.std(fano_acc_upper_bound_list)
print(f"各折测试集准确率: {[f'{acc:.2f}%' for acc in test_accuracies]}")
print(f"平均测试集准确率: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
print(f"平均测试集 Loss : {mean_loss:.4f}")

print(f"各折特征互信息 I(L;f): {[f'{mi:.4f}' for mi in mi_feature_list]}")
print(f"平均特征互信息 I(L;f): {mean_mi:.4f} bits (+/- {std_mi:.4f})")

print(f"各折 Fano-derived 错误率下界: {[f'{pe:.2f}%' for pe in fano_error_bound_list]}")
print(f"平均 Fano-derived 错误率下界: {mean_fano_error:.2f}% (+/- {std_fano_error:.2f}%)")

print(f"各折 Fano-derived 准确率参考上界: {[f'{acc:.2f}%' for acc in fano_acc_upper_bound_list]}")
print(f"平均 Fano-derived 准确率参考上界: {mean_fano_acc_upper:.2f}% (+/- {std_fano_acc_upper:.2f}%)")
print("==================================================")

# 保存 Test + MI + Fano 结果
with open(os.path.join(base_save_dir, "Mixed_Anechoic_Outdoor_GroupWise_10Fold_Results.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Fold", 
        "Test Accuracy (%)", 
        "Test Loss",
        "I(L;f) bits",
        "Fano Error Lower Bound (%)",
        "Fano Accuracy Upper Bound (%)"
    ])
    
    for i in range(10):
        writer.writerow([
            i+1, 
            test_accuracies[i], 
            test_losses[i],
            mi_feature_list[i],
            fano_error_bound_list[i],
            fano_acc_upper_bound_list[i]
        ])
    
    writer.writerow([
        "Mean", 
        mean_acc, 
        mean_loss,
        mean_mi,
        mean_fano_error,
        mean_fano_acc_upper
    ])

    writer.writerow([
        "Std", 
        std_acc, 
        "N/A",
        std_mi,
        std_fano_error,
        std_fano_acc_upper
    ])

pd.DataFrame({
    "Fold": np.arange(1, 11),
    "Test_Accuracy_percent": test_accuracies,
    "Test_Loss": test_losses,
    "I_L_f_bits": mi_feature_list,
    "Fano_Error_Lower_Bound_percent": fano_error_bound_list,
    "Fano_Accuracy_Upper_Bound_percent": fano_acc_upper_bound_list
}).to_csv(os.path.join(base_save_dir, 'MI_Fano_PerFold_Results.csv'), index=False)