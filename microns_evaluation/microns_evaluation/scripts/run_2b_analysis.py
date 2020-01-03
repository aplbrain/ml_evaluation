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

import os
import argparse
import csv
import zipfile
import io
from microns_evaluation.analysis.loading import load_episodes, load_outputs, load_outputs_from_zip
from microns_evaluation.analysis.metrics import *

EPISODE_CLASSES = [1, 2, 3, 4, 5, 6]


def main(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs_dir', help='Directory containing all the csvs to use for the experiment.')
    parser.add_argument('results_dir', help='Path to directory containing the result directories for each trial')
    parser.add_argument('--specific-episode', help='Specific episode to look at')
    parser.add_argument('--algorithm', default='',
                        help='Name of the algorithm that produced the results. Used to label plots.')
    parser.add_argument('--experiment', default='',
                        help='Name of the experiment that the algorithm ran on. Used to label plots.')
    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    # load all episodes
    episode_by_tag = defaultdict(dict)
    load_episodes(args.csvs_dir, episode_by_tag,
                  [] if args.specific_episode is None else [tuple(args.specific_episode.split('-'))])
    episode_by_tag = dict(episode_by_tag)

    results_are_zip = args.results_dir.endswith('.zip')

    # load results
    if results_are_zip:
        load_outputs_from_zip(args.results_dir, episode_by_tag)
    else:
        load_outputs(args.results_dir, episode_by_tag)

    results_zip = None
    if results_are_zip:
        results_zip = zipfile.ZipFile(args.results_dir)

    metrics = Metrics(*extract_classes(episode_by_tag))

    for tag in episode_by_tag.keys():
        label_by_trial = {}
        score_by_class_by_trial = {}
        alg_by_trial = {}
        extrinsic_by_trial = {}

        if 'output' not in episode_by_tag[tag] or (
                not results_are_zip and not os.path.exists(episode_by_tag[tag]['output'])):
            print('WARNING: results for {}-{} not found'.format(tag[0], tag[1]))
            continue

        # load the labels
        with open(episode_by_tag[tag]['labels']) as labels_fp:
            for label in labels_fp:
                label_data = parse_label(label)
                label_by_trial[label_data[0]] = label_data[1:]

        with open(episode_by_tag[tag]['test']) as test_fp:
            for trial in test_fp:
                trial_id, ref, c1, c2, c3, c4, c5, c6 = trial.strip().split(',')
                alg, _, extrinsic, _ = os.path.splitext(ref)[0].split('-')
                alg_by_trial[trial_id] = alg
                extrinsic_by_trial[trial_id] = extrinsic

        # load all results
        open_fn = open
        if results_are_zip:
            open_fn = lambda f: io.TextIOWrapper(results_zip.open(f))
        with open_fn(episode_by_tag[tag]['output']) as output_fp:
            reader = csv.DictReader(output_fp, fieldnames=['trial-id', *EPISODE_CLASSES])
            for result in reader:
                trial_id = result['trial-id']
                score_by_class = {}
                for episode_class in EPISODE_CLASSES:
                    if result[episode_class] is None:
                        print(f'No result for class {episode_class}.  Using value of 0.')
                        result[episode_class] = 0.
                    score_by_class[episode_class] = float(result[episode_class])
                score_by_class_by_trial[trial_id] = score_by_class

        # gather metrics
        for trial_id in score_by_class_by_trial:
            correct_episode_class, ref_raw_class, ref_extrinsic, raw_classes, extrinsics = label_by_trial[trial_id]
            raw_class_by_episode_class = {ep_cls: raw_classes[i] for i, ep_cls in enumerate(EPISODE_CLASSES)}
            raw_extrinsic_by_episode_class = {ep_cls: extrinsics[i] for i, ep_cls in enumerate(EPISODE_CLASSES)}
            metrics.add_trial_result(alg_by_trial[trial_id], extrinsic_by_trial[trial_id],
                                     correct_episode_class, ref_raw_class, ref_extrinsic,
                                     score_by_class_by_trial[trial_id], raw_class_by_episode_class)

    if results_are_zip:
        results_zip.close()

    for metric_fn in ALL_METRIC_FNS:
        metric_fn(metrics, args.algorithm, args.experiment)


def extract_classes(episode_by_tag):
    raw_classes = set()
    episode_classes = set()
    for tag in episode_by_tag:
        with open(episode_by_tag[tag]['labels']) as fp:
            for line in fp:
                trial_id, episode_class, ref_class, ref_extrin, choice_classes, choice_extrins = parse_label(line)
                raw_classes.add(ref_class)
                episode_classes.add(episode_class)
    return list(episode_classes), list(raw_classes)


def parse_label(entry):
    parts = entry.strip().split(',')
    trial_id = parts[0]
    label = int(parts[1])
    rc, re, c1, e1, c2, e2, c3, e3, c4, e4, c5, e5, c6, e6 = map(int, parts[2:])
    return trial_id, label, rc, re, [c1, c2, c3, c4, c5, c6], [e1, e2, e3, e4, e5, e6],


if __name__ == '__main__':
    main()
