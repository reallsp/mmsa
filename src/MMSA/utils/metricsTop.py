import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
                'SIMS': self.__eval_sims_regression,
                'SIMSV2': self.__eval_sims_regression,
                'CUSTOM': self.__eval_mosi_regression,  # 使用MOSI的回归指标
                'TRAIN_12_16': self.__eval_mosi_regression,  # 使用MOSI的回归指标
                'COPA_1231': self.__eval_mosi_regression,  # 使用MOSI的回归指标
            }
        else:
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_classification,
                'MOSEI': self.__eval_mosei_classification,
                'SIMS': self.__eval_sims_classification,
                'SIMSV2': self.__eval_sims_classification,
                'CUSTOM': self.__eval_mosi_classification,  # 使用MOSI的分类指标
                'TRAIN_12_16': self.__eval_mosi_classification,  # 使用MOSI的分类指标
                'COPA_1231': self.__eval_mosi_classification,  # 使用MOSI的分类指标
            }

    def __eval_mosi_classification(self, y_pred, y_true):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2   
        }
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        # three classes
        y_pred_3 = np.argmax(y_pred, axis=1)
        Mult_acc_3 = accuracy_score(y_pred_3, y_true)
        F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
        # two classes 
        y_pred = np.array([[v[0], v[2]] for v in y_pred])
        # with 0 (<= 0 or > 0)
        y_pred_2 = np.argmax(y_pred, axis=1)
        y_true_2 = []
        for v in y_true:
            y_true_2.append(0 if v <= 1 else 1)
        y_true_2 = np.array(y_true_2)
        Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
        # without 0 (< 0 or > 0)
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
        y_pred_2 = y_pred[non_zeros]
        y_pred_2 = np.argmax(y_pred_2, axis=1)
        y_true_2 = y_true[non_zeros]
        Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

        eval_results = {
            "Has0_acc_2":  round(Has0_acc_2, 4),
            "Has0_F1_score": round(Has0_F1_score, 4),
            "Non0_acc_2":  round(Non0_acc_2, 4),
            "Non0_F1_score": round(Non0_F1_score, 4),
            "Acc_3": round(Mult_acc_3, 4),
            "F1_score_3": round(F1_score_3, 4)
        }
        return eval_results
    
    def __eval_mosei_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __eval_sims_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)


        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        # 转换为0/1标签（False->0, True->1）
        binary_truth_int = binary_truth.astype(int)
        binary_preds_int = binary_preds.astype(int)
        acc2 = accuracy_score(binary_preds_int, binary_truth_int)
        f_score = f1_score(binary_truth_int, binary_preds_int, average='weighted')
        
        # 计算每个类别的精确率、召回率和F1分数（标签0和1）
        precision_per_class = precision_score(binary_truth_int, binary_preds_int, average=None, zero_division=0)
        recall_per_class = recall_score(binary_truth_int, binary_preds_int, average=None, zero_division=0)
        f1_per_class = f1_score(binary_truth_int, binary_preds_int, average=None, zero_division=0)
        
        eval_results = {
            "Has0_acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
            # 标签0的指标
            "Label0_Precision": round(precision_per_class[0], 4) if len(precision_per_class) > 0 else 0.0,
            "Label0_Recall": round(recall_per_class[0], 4) if len(recall_per_class) > 0 else 0.0,
            "Label0_F1": round(f1_per_class[0], 4) if len(f1_per_class) > 0 else 0.0,
            # 标签1的指标
            "Label1_Precision": round(precision_per_class[1], 4) if len(precision_per_class) > 1 else 0.0,
            "Label1_Recall": round(recall_per_class[1], 4) if len(recall_per_class) > 1 else 0.0,
            "Label1_F1": round(f1_per_class[1], 4) if len(f1_per_class) > 1 else 0.0,
        }
        return eval_results


    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": round(mult_a2, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4), # Correlation Coefficient
        }
        return eval_results
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
    
    def eval_copa_paradigm_accuracy(self, y_pred, y_true, sample_indices=None, group_type='i1'):
        """
        评估COPA心理范式的准确率
        
        Args:
            y_pred: 预测值 (tensor or numpy array), shape: (N,)
            y_true: 真实值 (tensor or numpy array), shape: (N,)
            sample_indices: 样本索引列表，用于确定每个样本属于哪个范式
            group_type: 群体类型 ('i1': 男犯, 'i2': 女犯, 'i3': 未成年)
        
        Returns:
            dict: COPA范式评估结果
        """
        # 转换为numpy数组
        if hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu().detach().numpy()
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().detach().numpy()
        
        # 如果是多维，展平
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        
        # 将回归值转换为分类值 (0或1)
        # 假设回归值范围是[-1, 1]，需要转换为[0, 1]
        # 如果y_pred已经是0/1，则不需要转换
        if y_pred.min() < 0 or y_pred.max() > 1:
            # 将[-1, 1]映射到[0, 1]
            y_pred_class = ((y_pred + 1) / 2).round().astype(int)
            y_pred_class = np.clip(y_pred_class, 0, 1)
        else:
            y_pred_class = y_pred.round().astype(int)
        
        if y_true.min() < 0 or y_true.max() > 1:
            y_true_class = ((y_true + 1) / 2).round().astype(int)
            y_true_class = np.clip(y_true_class, 0, 1)
        else:
            y_true_class = y_true.round().astype(int)
        
        # COPA范式定义
        paradigms = {
            "P1": [0, 16, 32, 48, 64, 80, 96, 109],
            "P2": [1, 17, 33, 49, 65, 81, 97, 110],
            "P3": [3, 19, 35, 51, 67, 83, 99, 112],
            "P4": [5, 21, 37, 53, 69, 85, 100, 113],
            "P5": [23, 39, 55, 71, 87, 89, 102, 115],
            "P6": [9, 25, 41, 57, 73, 74, 86, 117],
            "P7": [11, 27, 43, 59, 75, 91, 105, 118],
            "P8": [13, 29, 45, 61, 77, 93, 107, 120],
            "P9": [7, 15, 31, 47, 63, 79, 95, 108],
            "P10": [2, 18, 26, 34, 42, 44, 50, 52, 58, 66, 82, 90],
            "P11": [4, 12, 20, 28, 36, 60, 68, 76, 84, 92, 104, 121],
            "P12": [6, 10, 14, 22, 30, 38, 46, 54, 62, 70, 78, 94]
        }
        
        # 常模数据
        norm_data = {
            "P1": {"i1": {"mean": 5.1285, "std": 2.3828}, "i2": {"mean": 4.6214, "std": 2.4538}, "i3": {"mean": 5.4530, "std": 2.1273}},
            "P2": {"i1": {"mean": 4.3345, "std": 2.4344}, "i2": {"mean": 4.2658, "std": 2.4801}, "i3": {"mean": 4.2108, "std": 2.4044}},
            "P3": {"i1": {"mean": 6.9572, "std": 1.4691}, "i2": {"mean": 7.2494, "std": 1.2188}, "i3": {"mean": 6.4582, "std": 1.7123}},
            "P4": {"i1": {"mean": 2.1710, "std": 1.9742}, "i2": {"mean": 2.3374, "std": 2.0306}, "i3": {"mean": 2.6899, "std": 2.0704}},
            "P5": {"i1": {"mean": 4.1637, "std": 2.5371}, "i2": {"mean": 4.0531, "std": 2.5925}, "i3": {"mean": 4.9487, "std": 2.3277}},
            "P6": {"i1": {"mean": 3.7783, "std": 2.2979}, "i2": {"mean": 3.3951, "std": 2.2326}, "i3": {"mean": 4.2770, "std": 2.2951}},
            "P7": {"i1": {"mean": 3.1612, "std": 2.3235}, "i2": {"mean": 2.8749, "std": 2.1610}, "i3": {"mean": 3.1681, "std": 2.2884}},
            "P8": {"i1": {"mean": 3.0347, "std": 2.3788}, "i2": {"mean": 3.0872, "std": 2.4887}, "i3": {"mean": 3.2021, "std": 2.3921}},
            "P9": {"i1": {"mean": 3.9035, "std": 2.3100}, "i2": {"mean": 3.7218, "std": 2.4190}, "i3": {"mean": 4.0192, "std": 2.1929}},
            "P10": {"i1": {"mean": 3.8228, "std": 3.3359}, "i2": {"mean": 2.3228, "std": 2.3648}, "i3": {"mean": 4.7190, "std": 3.4341}},
            "P11": {"i1": {"mean": 1.5697, "std": 1.8542}, "i2": {"mean": 1.1868, "std": 1.6012}, "i3": {"mean": 2.2030, "std": 2.0456}},
            "P12": {"i1": {"mean": 4.9915, "std": 3.1602}, "i2": {"mean": 3.9119, "std": 2.7829}, "i3": {"mean": 5.6280, "std": 2.9183}}
        }
        
        # T分数范围
        t_ranges = {
            "低分": {"min": 0, "max": 35},
            "较低分": {"min": 36, "max": 45},
            "中等分": {"min": 46, "max": 54},
            "较高": {"min": 55, "max": 64},
            "高分": {"min": 65, "max": 100}
        }
        
        def get_level(T):
            if T <= 35:
                return "低分"
            elif T <= 45:
                return "较低分"
            elif T < 55:
                return "中等分"
            elif T < 65:
                return "较高"
            else:
                return "高分"
        
        # 如果没有提供样本索引，假设样本按顺序排列，每组122个样本
        if sample_indices is None:
            sample_indices = np.arange(len(y_pred))
        
        # 按组处理（每组122个样本）
        group_size = 122
        num_groups = len(y_pred) // group_size
        if num_groups == 0:
            num_groups = 1
        
        # 统计每个范式的准确率
        paradigm_stats = {p: {"correct": 0, "total": 0} for p in paradigms}
        overall_correct = 0
        overall_total = 0
        
        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, len(y_pred))
            
            group_pred = y_pred_class[start_idx:end_idx]
            group_true = y_true_class[start_idx:end_idx]
            group_indices = sample_indices[start_idx:end_idx]
            
            # 处理每个范式
            for paradigm, indices in paradigms.items():
                # 检查索引是否在组内
                valid_indices = [i for i in indices if i < len(group_pred)]
                if len(valid_indices) == 0:
                    continue
                
                # 检查是否有缺失数据
                if any(i >= len(group_pred) for i in valid_indices):
                    continue
                
                # 计算范式聚合分数（求和）
                pred_xj = group_pred[valid_indices].sum()
                label_xj = group_true[valid_indices].sum()
                
                # 获取常模数据
                norm = norm_data[paradigm][group_type]
                
                # 计算Z分数
                pred_Z = (pred_xj - norm["mean"]) / norm["std"] if norm["std"] > 0 else 0
                label_Z = (label_xj - norm["mean"]) / norm["std"] if norm["std"] > 0 else 0
                
                # 计算T分数
                pred_T = int(50 + 10 * pred_Z)
                label_T = int(50 + 10 * label_Z)
                
                # 获取等级
                pred_level = get_level(pred_T)
                label_level = get_level(label_T)
                
                # 判断是否在范围内
                in_range = t_ranges[label_level]["min"] <= pred_T <= t_ranges[label_level]["max"]
                
                paradigm_stats[paradigm]["total"] += 1
                overall_total += 1
                
                if in_range:
                    paradigm_stats[paradigm]["correct"] += 1
                    overall_correct += 1
        
        # 计算每个范式的准确率
        paradigm_accuracies = {}
        for p in paradigms:
            total = paradigm_stats[p]["total"]
            correct = paradigm_stats[p]["correct"]
            acc = correct / total if total > 0 else 0.0
            paradigm_accuracies[f"COPA_{p}_acc"] = round(acc, 4)
        
        # 整体准确率
        overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
        
        result = {
            "COPA_overall_acc": round(overall_acc, 4),
            **paradigm_accuracies
        }
        
        return result