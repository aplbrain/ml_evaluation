# Copyright 2019 The Johns Hopkins University Applied Physics Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats


class Metrics:
    def __init__(self, episode_classes, raw_classes):
        self.episode_classes = sorted(episode_classes)
        self.raw_classes = sorted(raw_classes)
        self.new_episode_class = max(episode_classes)

        self.num_trials = 0
        self.num_trials_by_episode_class = defaultdict(int)
        self.num_trials_by_raw_class = defaultdict(int)

        self.correctness_overall = {True: 0, False: 0}
        self.correctness_by_episode_class = defaultdict(lambda: {True: 0, False: 0})
        self.correctness_by_raw_class = defaultdict(lambda: {True: 0, False: 0})
        self.correctness_by_raw_extrinsic = defaultdict(lambda: {True: 0, False: 0})

        self.correctness_by_alg = defaultdict(lambda: {True: 0, False: 0})
        self.correctness_by_extrinsic = defaultdict(lambda: {True: 0, False: 0})

        self.episode_confusion = defaultdict(lambda: defaultdict(int))
        self.raw_confusion = defaultdict(lambda: defaultdict(int))
        self.new_class_confusion = defaultdict(lambda: defaultdict(int))

        self.trial_results = []
        self.new_class_results = []
        self.original_class_results = []

        self.new_class_scores = []
        self.old_class_scores = []
        self.correct_is_new = []
        self.class_labels = []
        self.chosen_labels = []
        self.thresholds = None
        self.fpr = None
        self.fnr = None
        self.tpr = None
        self.tnr = None
        self.precision = None
        self.new_class_precision = None
        self.new_class_recall = None
        self.new_class_f1 = None
        self.new_class_acc = None
        self.entropy = []

    def add_trial_result(self, correct_alg, correct_extrinsic,
                         correct_episode_class, correct_raw_class, correct_raw_extrinsic,
                         score_by_episode_class, raw_class_by_episode_class):
        # sort the candidates by their distances & compute rankings for each candidate
        sorted_episode_classes = sorted(score_by_episode_class, key=score_by_episode_class.get, reverse=True)

        # compute the candidate that the alg chose & the candidate that was correct
        chosen_episode_class = sorted_episode_classes[0]

        is_correct = chosen_episode_class == correct_episode_class

        self.correctness_overall[is_correct] += 1
        self.correctness_by_episode_class[correct_episode_class][is_correct] += 1
        self.correctness_by_raw_class[correct_raw_class][is_correct] += 1
        self.correctness_by_raw_extrinsic[correct_raw_extrinsic][is_correct] += 1

        self.correctness_by_alg[correct_alg][is_correct] += 1
        self.correctness_by_extrinsic[correct_extrinsic][is_correct] += 1

        correct_is_new_class = correct_episode_class == self.new_episode_class
        chosen_is_new_class = chosen_episode_class == self.new_episode_class
        self.new_class_confusion[correct_is_new_class][chosen_is_new_class] += 1
        self.episode_confusion[correct_episode_class][chosen_episode_class] += 1

        chosen_raw_class = raw_class_by_episode_class[chosen_episode_class]
        self.raw_confusion[correct_raw_class][chosen_raw_class] += 1

        self.num_trials += 1
        self.num_trials_by_episode_class[correct_episode_class] += 1
        self.num_trials_by_raw_class[correct_raw_class] += 1

        self.trial_results.append(is_correct)
        if correct_is_new_class:
            self.new_class_results.append(chosen_is_new_class)
        else:
            self.original_class_results.append(is_correct)

        softmax_scores = self.get_softmax_scores(score_by_episode_class)
        self.entropy.append(entropy(list(softmax_scores.values())))
        new_class_score = softmax_scores[self.new_episode_class]
        softmax_scores.pop(self.new_episode_class)
        old_class_score = list(softmax_scores.values())
        self.new_class_scores.append(new_class_score)
        self.old_class_scores.append(old_class_score)
        self.correct_is_new.append(correct_is_new_class)
        self.class_labels.append(correct_episode_class)
        self.chosen_labels.append(chosen_episode_class)
        # self.original_class_results.append(is_correct)

    @staticmethod
    def get_softmax_scores(class_scores):
        all_scores = np.array(list(class_scores.values()))
        denom = np.sum(np.exp(all_scores))

        return {k: np.exp(v) / denom for k, v in class_scores.items()}

    def confidence_intervals(self, do_new_class_ci=False, alpha=0.05):
        # calculate bootstrap estimates for the mean and standard deviation
        ci_obj = bs.bootstrap(np.array(self.trial_results), stat_func=bs_stats.mean, alpha=alpha)
        m = (ci_obj.value, ci_obj.lower_bound, ci_obj.upper_bound)

        if do_new_class_ci:
            nc_obj = bs.bootstrap(np.array(self.new_class_results), stat_func=bs_stats.mean, alpha=alpha)
            nc = (nc_obj.value, nc_obj.lower_bound, nc_obj.upper_bound)
        else:
            nc = (0, 0, 0)

        return m, nc

    def build_confusion_matrix(self, n_thresholds=100, mode='naive'):
        self.new_class_scores = np.array(self.new_class_scores)
        self.old_class_scores = np.array(self.old_class_scores)
        self.thresholds = np.linspace(0, 1., n_thresholds)
        self.tpr = np.zeros(self.thresholds.shape)
        self.tnr = np.zeros(self.thresholds.shape)
        self.fpr = np.zeros(self.thresholds.shape)
        self.fnr = np.zeros(self.thresholds.shape)
        self.precision = np.zeros(self.thresholds.shape)
        idx = self.correct_is_new

        new_class_tp = np.sum(self.new_class_scores[idx] > self.old_class_scores[idx].max(axis=1))
        new_class_fp = np.sum(
            self.new_class_scores[np.logical_not(idx)] > self.old_class_scores[np.logical_not(idx)].max(axis=1))
        new_class_fn = np.sum(self.new_class_scores[idx] < self.old_class_scores[idx].max(axis=1))
        new_class_tn = np.sum(
            self.new_class_scores[np.logical_not(idx)] < self.old_class_scores[np.logical_not(idx)].max(axis=1))
        self.new_class_precision = new_class_tp / (new_class_tp + new_class_fp)
        self.new_class_recall = new_class_tp / (new_class_tp + new_class_fn)
        self.new_class_f1 = (2 * new_class_tp) / (2 * new_class_tp + new_class_fp + new_class_fn)
        self.new_class_acc = (new_class_tn + new_class_tp) / len(idx)  # over all trials

        for i, t in enumerate(self.thresholds):
            if mode == 'naive':
                # Each trial is one test example (the threshold is one-sided and applied only to the new class scores)
                tp = np.sum(self.new_class_scores[idx] > t)
                fp = np.sum(self.new_class_scores[np.logical_not(idx)] > t)
            elif mode == 'micro':
                # Each label is treated as an exemplar
                w_new_class = t * self.new_class_scores
                w_old_class = (1 - t) * self.old_class_scores
                # Note that the division below is by the number of classes but when combined with the calculation of TPR
                # it fully accounts for the total number of trials in this setting
                tp = np.sum(w_new_class[idx, np.newaxis] > w_old_class[idx]) / w_old_class.shape[1]
                fp = np.sum(w_new_class[np.logical_not(idx), np.newaxis] > w_old_class[np.logical_not(idx)]) / \
                     w_old_class.shape[1]
            elif mode == 'ratio':
                # Each trial is one test example; ratio test uses weighting to compare the two classes
                old_class_scores = self.old_class_scores.max(axis=1)
                w_new_class = t * self.new_class_scores
                w_old_class = (1 - t) * old_class_scores
                tp = np.sum(w_new_class[idx] > w_old_class[idx])
                fp = np.sum(w_new_class[np.logical_not(idx)] > w_old_class[np.logical_not(idx)])

            self.tpr[i] = tp / np.sum(idx)
            self.fpr[i] = fp / np.sum(np.logical_not(idx))
            self.tnr[i] = 1 - self.fpr[i]
            self.fnr[i] = 1 - self.tpr[i]
            self.precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0


def print_accuracy(metrics, algorithm, experiment):
    correct = metrics.correctness_overall[True]
    incorrect = metrics.correctness_overall[False]
    total = correct + incorrect
    print('Overall Accuracy: {} / {} ({:.2f}%)'.format(correct, total, 100 * correct / total))


def print_ci(metrics):
    m, nc = metrics.confidence_intervals(do_new_class_ci=True)
    print(f'{m[0]:.4f},{m[0] - m[1]:.4f},{m[2] - m[0]:.4f}')
    print(f'{nc[0]:.4f},{nc[0] - nc[1]:.4f},{nc[2] - nc[0]:.4f}')


def confidence_intervals(metrics, algorithm, experiment, alpha=0.05):
    m, nc = metrics.confidence_intervals(do_new_class_ci=False, alpha=alpha)
    with open('{}_{}_confidence-intervals.csv'.format(algorithm, experiment), 'w') as fp:
        fp.write('category, mean, lower, upper\n')
        fp.write(f'overall,{m[0]:.4f},{m[0] - m[1]:.4f},{m[2] - m[0]:.4f}\n')
        fp.write(f'new_vs_old,{nc[0]:.4f},{nc[0] - nc[1]:.4f},{nc[2] - nc[0]:.4f}')


def accuracy(metrics, algorithm, experiment):
    with open('{}_{}_overall-accuracy.csv'.format(algorithm, experiment), 'w') as fp:
        fp.write('num_correct,num_incorrect\n')
        fp.write('{},{}\n'.format(metrics.correctness_overall[True], metrics.correctness_overall[False]))


def class_accuracy(metrics, algorithm, experiment):
    with open('{}_{}_class-accuracy.csv'.format(algorithm, experiment), 'w') as fp:
        fp.write('class,num_correct,num_incorrect\n')
        for cls in sorted(metrics.correctness_by_raw_class):
            fp.write('{},{},{}\n'.format(cls, metrics.correctness_by_raw_class[cls][True],
                                         metrics.correctness_by_raw_class[cls][False]))


def episode_class_accuracy(metrics, algorithm, experiment):
    with open('{}_{}_episode-class-accuracy.csv'.format(algorithm, experiment), 'w') as fp:
        fp.write('episode_class,num_correct,num_incorrect\n')
        for cls in sorted(metrics.episode_classes):
            fp.write('{},{},{}\n'.format(cls, metrics.correctness_by_episode_class[cls][True],
                                         metrics.correctness_by_episode_class[cls][False]))


def extrinsic_accuracy(metrics, algorithm, experiment):
    with open('{}_{}_extrinsic-accuracy.csv'.format(algorithm, experiment), 'w') as fp:
        fp.write('extrinsic,num_correct,num_incorrect\n')
        for extrinsic in sorted(metrics.correctness_by_raw_extrinsic):
            fp.write('{},{},{}\n'.format(extrinsic, metrics.correctness_by_raw_extrinsic[extrinsic][True],
                                         metrics.correctness_by_raw_extrinsic[extrinsic][False]))


def plot_alg_accuracy(metrics, algorithm, experiment):
    plt.figure(dpi=300)
    plt.title('{} on {} - Algorithm Class Accuracy (Percent)'.format(algorithm, experiment))
    algs = sorted(metrics.correctness_by_alg)
    accuracies = [metrics.correctness_by_alg[alg][True] / sum(metrics.correctness_by_alg[alg].values()) for alg in algs]
    sns.barplot(x=algs, y=accuracies, color='C0')
    plt.xticks(rotation=45)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylabel('Accuracy')
    plt.xlabel('Algorithm Class')
    plt.tight_layout()
    plt.savefig('{}_{}_alg-accuracy.png'.format(algorithm, experiment))


def plot_extrinsic_accuracy(metrics, algorithm, experiment):
    plt.figure(dpi=300)
    plt.title('{} on {} - Extrinsic Class Accuracy (Percent)'.format(algorithm, experiment))
    extrs = sorted(metrics.correctness_by_extrinsic)
    accuracies = [metrics.correctness_by_extrinsic[extr][True] / sum(metrics.correctness_by_extrinsic[extr].values())
                  for extr in extrs]
    sns.barplot(x=extrs, y=accuracies, color='C0')
    plt.xticks(rotation=45)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylabel('Accuracy')
    plt.xlabel('Extrinsic Class')
    plt.tight_layout()
    plt.savefig('{}_{}_extrinsic-accuracy.png'.format(algorithm, experiment))


def confusion(metrics, algorithm, experiment):
    plt.figure(dpi=300)
    plt.title('{} on {} - Confusion Matrix (Count)'.format(algorithm, experiment))
    sns.heatmap(
        [[metrics.raw_confusion[correct_class][chosen_class] for chosen_class in metrics.raw_classes]
         for correct_class in metrics.raw_classes],
        cmap='Greens', annot=True, fmt='d', square=True, xticklabels=metrics.raw_classes,
        yticklabels=metrics.raw_classes, annot_kws={'weight': 'heavy'})
    plt.ylabel('Correct Class')
    plt.xlabel('Chosen Class')
    plt.savefig('{}_{}_confusion.png'.format(algorithm, experiment))


def episode_class_confusion(metrics, algorithm, experiment):
    plt.figure(dpi=300)
    plt.title('{} on {} - Confusion Matrix (Count)'.format(algorithm, experiment))
    sns.heatmap(
        [[metrics.episode_confusion[correct_class][chosen_class] for chosen_class in metrics.episode_classes]
         for correct_class in metrics.episode_classes],
        cmap='Greens', annot=True, fmt='.1f', square=True, xticklabels=metrics.episode_classes,
        yticklabels=metrics.episode_classes, annot_kws={'weight': 'heavy'})
    plt.ylabel('Correct Class')
    plt.xlabel('Chosen Class')
    plt.savefig('{}_{}_episode-class-confusion-count.png'.format(algorithm, experiment))

    plt.figure(dpi=300)
    plt.title('{} on {} - Confusion Matrix (Percent Accuracy)'.format(algorithm, experiment))
    sns.heatmap(
        [[100.0 * metrics.episode_confusion[correct_class][chosen_class] / metrics.num_trials_by_episode_class[
            correct_class]
          for chosen_class in metrics.episode_classes]
         for correct_class in metrics.episode_classes],
        cmap='Greens', annot=True, fmt='.1f', square=True, xticklabels=metrics.episode_classes,
        yticklabels=metrics.episode_classes, annot_kws={'weight': 'heavy'})
    plt.ylabel('Correct Class')
    plt.xlabel('Chosen Class')
    plt.savefig('{}_{}_episode-class-confusion-pct.png'.format(algorithm, experiment))


def new_class_confusion(metrics, algorithm, experiment):
    plt.figure(dpi=300)
    plt.title('{} on {} - New Class Confusion Matrix'.format(algorithm, experiment))
    num_new = metrics.new_class_confusion[True][True] + metrics.new_class_confusion[True][False]
    num_old = metrics.new_class_confusion[False][True] + metrics.new_class_confusion[False][False]
    sns.heatmap(
        [[100 * metrics.new_class_confusion[True][True] / num_new,
          100 * metrics.new_class_confusion[True][False] / num_new],
         [100 * metrics.new_class_confusion[False][True] / num_old,
          100 * metrics.new_class_confusion[False][False] / num_old]],
        cmap='Greens', annot=True, fmt='.1f', square=True, xticklabels=['New Class', 'Old Class'],
        yticklabels=['New Class', 'Old Class'], annot_kws={'weight': 'heavy'})
    plt.ylabel('Correct Class')
    plt.xlabel('Chosen Class')
    plt.savefig('{}_{}_new-class-confusion.png'.format(algorithm, experiment))


def roc_curve(metrics, algorithm, experiment):
    metrics.build_confusion_matrix()
    plt.figure(dpi=300)
    plt.title(f'{algorithm} on {experiment} - New Class ROC Curve')
    sns.lineplot(metrics.fpr, metrics.tpr)
    sns.lineplot(metrics.thresholds, metrics.thresholds, dashes=True)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('{}_{}_new-class-roc.png'.format(algorithm, experiment))


def pr_curve(metrics, algorithm, experiment):
    metrics.build_confusion_matrix()
    plt.figure(dpi=300)
    plt.title(f'{algorithm} on {experiment} - New Class PR Curve')
    sns.lineplot(metrics.tpr, metrics.precision)
    plt.ylabel('Precision (PPV)')
    plt.xlabel('Recall (TPR)')
    plt.savefig('{}_{}_new-class-pr.png'.format(algorithm, experiment))


def plot_entropy(metrics, algorithm, experiment):
    plt.figure(dpi=300)
    plt.title(f'{algorithm} on {experiment} - Entropy')
    ent = np.array(metrics.entropy)
    trial_idx = np.arange(len(metrics.trial_results))
    correct = np.array(metrics.trial_results).astype(np.bool)
    sns.scatterplot(trial_idx[correct], ent[correct], markers='.', size=1., color='b', legend=False)
    not_correct = np.logical_not(metrics.trial_results).astype(np.bool)
    sns.scatterplot(trial_idx[not_correct], ent[not_correct], markers='.', size=1., color='r', legend=False)
    plt.ylabel('Softmax Score Entropy')
    plt.xlabel('Trial ID')
    plt.savefig(f'{algorithm}_{experiment}_softmax-score-entropy.png')

    print(f'Correct result avg entropy: {ent[correct].mean()}')
    print(f'Incorrect result avg entropy: {ent[not_correct].mean()}')


def full_summary(metrics, algorithm, experiment):
    metrics.build_confusion_matrix()
    m, nc = metrics.confidence_intervals(do_new_class_ci=True, alpha=0.05)

    oc_obj = bs.bootstrap(np.array(metrics.original_class_results), stat_func=bs_stats.mean, alpha=0.05)
    oc = (oc_obj.value, oc_obj.lower_bound, oc_obj.upper_bound)

    cm = np.array([[metrics.episode_confusion[correct_class][chosen_class] for chosen_class in metrics.episode_classes]
                   for correct_class in metrics.episode_classes])
    mean_recall = np.diag(cm[:-1, :-1]) / np.sum(cm[:-1, :-1], axis=1)
    mean_precision = np.diag(cm[:-1, :-1]) / np.sum(cm[:-1, :-1], axis=0)
    # mean_recall = np.diag(cm) / np.sum(cm, axis=1)
    # mean_precision = np.diag(cm) / np.sum(cm, axis=0)

    with open('{}_{}_summary.csv'.format(algorithm, experiment), 'w') as fp:
        fp.write('category,metric,value,lower_ci,upper_ci\n')
        # Multi-class accuracy over all trials
        fp.write(f'overall,accuracy,{m[0]:.4f},{m[0] - m[1]:.4f},{m[2] - m[0]:.4f}\n')
        # Accuracy only amongst trials where the original class was the correct answer
        fp.write(f'original,accuracy_ci,{oc[0]:.4f},{oc[0] - oc[1]:.4f},{oc[2] - oc[0]:.4f}\n')
        # Accuracy only amongst trials where the new class was the correct answer
        fp.write(f'new,accuracy_ci,{nc[0]:.4f},{nc[0] - nc[1]:.4f},{nc[2] - nc[0]:.4f}\n')

        # Average precision/recall over trials where one of the original classes was the correct answer
        fp.write(f'original,mean_precision,{mean_precision.mean():.3f}\n')
        fp.write(f'original,mean_recall,{mean_recall.mean():.3f}\n')

        # Precision/recall/f1/accuracy over all trials (treated as a new class detection task)
        fp.write(f'new,precision,{metrics.new_class_precision:.3f}\n')
        fp.write(f'new,recall,{metrics.new_class_recall:.3f}\n')
        fp.write(f'new,f1,{metrics.new_class_f1:.3f}\n')
        fp.write(f'new,accuracy,{metrics.new_class_acc:.3f}\n')

        # Average entropy of the similarity scores (to measure the separation of classes)
        ent = np.array(metrics.entropy)
        fp.write(f'overall,class_score_entropy,{ent.mean()}\n')


ALL_METRIC_FNS = [
    accuracy,
    # class_accuracy,
    # episode_class_accuracy,
    # extrinsic_accuracy,
    # episode_class_confusion,
    confidence_intervals,
    # confusion,
    new_class_confusion,
    # # roc_curve,
    # # pr_curve,
    # # plot_entropy,
    full_summary
]
