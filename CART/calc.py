import numpy as np
from typing import Dict, List


class Feature:
    def __init__(self, id: str, name_to_value: Dict[str, float], ground_truth):
        self.id = id
        self.name_to_value = name_to_value
        self.ground_truth = ground_truth


class OneFeatureAllSamples:
    def __init__(self, feature_name: str, ids: List[int], values: List[float], ground_truths: List[float],
                 condition=None):
        self.feature_name = feature_name
        self.ids = ids
        self.values = np.array(values)
        self.ground_truths = ground_truths
        self.condition = condition

    def find_best_splits(self):
        feature_level = np.unique(np.sort(self.values))
        thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0
        min_mse = None
        best = None
        print("find split for {}".format(self.feature_name))
        print("all split points: {}".format(",".join([str(t) for t in thresholds])))
        for threshold in thresholds:
            print("##-----------------------------split point: {:.2f}--------------------------------------".format(
                threshold))

            l_condition = self.values <= threshold
            l_tree = self.generate_split_by_condition(l_condition)
            l_mse, l_calculation = l_tree.calculate_mse()
            print("left branch: split data (id, feature: {feature_name}, result) pair: {pair}".format(
                feature_name=self.feature_name, pair=l_tree.get_show_pair()))
            print("left branch: mse: {:.2f}".format(l_mse))
            print("left branch: calculation: {}".format(l_calculation))

            r_condition = self.values > threshold
            r_tree = self.generate_split_by_condition(r_condition)
            r_mse, r_calculation = r_tree.calculate_mse()
            print("right branch: split data (id, feature: {feature_name}, result) pair: {pair}".format(
                feature_name=self.feature_name, pair=r_tree.get_show_pair()))
            print("right branch: mse: {:.2f}".format(r_mse))
            print("right branch: calculation: {}".format(r_calculation))

            sum_mse = l_tree.values.size / self.values.size * l_mse + r_tree.values.size / self.values.size * r_mse
            print("sum mse of two branches: ({:.2f} + {:.2f}) / {:.2f} = {:.2f}".format(l_tree.values.size * l_mse,
                                                                                        r_tree.values.size * r_mse,
                                                                                        self.values.size, sum_mse))
            gain = self.calculate_mse()[0] - (sum_mse)
            if min_mse is None or min_mse < gain:
                best = BestSplit(threshold, gain, self, l_tree, r_tree)
                min_mse = gain
            print("##--------------------------threshold: {:.2f} end--------------------------\n\n".format(threshold))

        return best

    def generate_split_by_condition(self, condition):
        ground_truths = self.ground_truths[condition]
        ids = self.ids[condition]
        values = self.values[condition]
        tree = OneFeatureAllSamples(self.feature_name, ids, values, ground_truths, condition)
        return tree

    def split_by_condition(self, condition):
        l_ground_truths = self.ground_truths[condition]
        l_ids = self.ids[condition]
        l_values = self.values[condition]
        return l_ground_truths, l_ids, l_values

    def calculate_mse(self):
        result = ""
        samples_ground_truth = self.ground_truths
        samples_ground_truth_str = ["{:.2f}".format(ground_truth) for ground_truth in self.ground_truths]
        ground_truth = np.array(samples_ground_truth)
        average = np.sum(ground_truth) / len(ground_truth)
        mse = np.sum((ground_truth - average) ** 2) / len(ground_truth)
        sum = "+".join(samples_ground_truth_str)
        average_calculation = "average = ({}) / {} = {:.2f}\n".format(sum, len(self.ground_truths), average)

        mse_calculation = "se = ({}) = {:.2f}\n".format(
            "+\n".join(["({}-{:.2f})^2".format(g, average) for g in samples_ground_truth_str]),
            mse.real * len(ground_truth))
        result += average_calculation + mse_calculation
        return mse, result

    def get_show_pair(self):
        return " ".join(
            ["({},{},{})".format(self.ids[i], self.values[i], self.ground_truths[i]) for i in range(self.values.size)])


class BestSplit:
    def __init__(self, threshold, gain, feature_for_samples, l_tree: OneFeatureAllSamples,
                 r_tree: OneFeatureAllSamples):
        self.threshold = threshold
        self.feature_for_samples = feature_for_samples
        self.gain = gain
        self.l_tree = l_tree
        self.r_tree = r_tree
        self.best_split_l = None
        self.best_split_r = None

    def display_for_sub_tree(self, tree):
        if not tree:
            return ""
        return """
        ids: {ids}
        """.format(ids=tree.ids)

    def __str__(self):
        return """
        -----
        ids: {ids}
        split feature: {feature}
        split point: {threshold}
        -----
        
        left: {left}
        
        ---
        
        right: {right}
        
        """.format(ids=self.feature_for_samples.ids,
                   feature=self.feature_for_samples.feature_name,
                   threshold=self.threshold,
                   left=self.best_split_l.__str__() if self.best_split_l else self.display_for_sub_tree(self.l_tree),
                   right=self.best_split_r.__str__() if self.best_split_r else self.display_for_sub_tree(self.r_tree)
        )


# mse, result = ExampleGenerator([Feature(0, {}, i.__float__()) for i in all]).calculate_mse()

ground_truth = np.array(np.array([100, 80, 70, 70, 20, 0, 20]))

features = {
    "看过的动漫数": [10, 2, 10, 0, 0, 0, 0],
    "性别": [1, 1, 0, 1, 1, 0, 1, ],
    "年龄": [20, 12, 12, 15, 80, 80, 8, ]
}

features_samples = [
    OneFeatureAllSamples(feature_name, np.array(range(len(ground_truth))), np.array(values), ground_truth) for
    feature_name, values in features.items()]


def split_once(features_samples):
    print(
        "-------------------------------------------------split start {}-------------------------------------------------".format(
            ",".join(map(lambda x: str(x), features_samples[0].ids))))
    splits = [feature.find_best_splits() for feature in features_samples]
    best_split = None
    for split in splits:
        if split is not None:
            left_count = split.l_tree.values.size
            left_mse = split.l_tree.calculate_mse()[0]
            right_count = split.r_tree.values.size
            right_mse = split.r_tree.calculate_mse()[0]
            mse = split.feature_for_samples.calculate_mse()[0]
            print(
                """best split for feature[{feature_name}]: 
split point is {split_point}
mse for left tree is {l_mse:.2f}
mse for right tree is {r_mse:.2f}
gain is {mse:.2f} - {sum_mse:.2f} = {gain:.2f}
left count is {left_count}
right count is {right_count} """.format(
                    split_point=split.threshold,
                    feature_name=split.feature_for_samples.feature_name,
                    all_count=split.feature_for_samples.values.size,
                    left_count=left_count,
                    sum_mse=mse - split.gain,
                    l_mse=left_mse,
                    r_se=right_mse * right_count,
                    right_count=right_count,
                    r_mse=right_mse,
                    mse=mse,
                    gain=split.gain))
            if best_split is None or best_split.gain <= split.gain:
                best_split = split
    print("final feature is {} and split point is {}".format(best_split.feature_for_samples.feature_name,
                                                             best_split.threshold))
    print(
        "-------------------------------------------------split end-------------------------------------------------\n\n\n")
    return best_split


def split_for(new_l_features):
    best_split_l = split_once(new_l_features)
    new_l_l_features = [feature.generate_split_by_condition(best_split_l.l_tree.condition) for feature in
                        new_l_features]
    new_l_r_features = [feature.generate_split_by_condition(best_split_l.r_tree.condition) for feature in
                        new_l_features]

    if len(new_l_l_features[0].ids) <= 1 or len(new_l_r_features[0].ids) <= 1:
        return None
    else:
        best_split_l.best_split_l = split_for(new_l_l_features)
        best_split_l.best_split_r = split_for(new_l_r_features)
        return best_split_l


best_split = split_for(features_samples)

print(best_split.__str__())
