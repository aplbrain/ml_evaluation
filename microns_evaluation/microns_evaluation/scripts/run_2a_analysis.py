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


def main(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs_dir', help='Directory containing all the csvs to use for the experiment.')
    parser.add_argument('results_dir', help='Path to directory containing the result directories for each trial')
    parser.add_argument('--specific-episode', help='Specific episode to look at. Ex: "easy-0000"')
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
        raw_class_by_episode_class = {}
        raw_extrinsic_by_episode_class = {}
        score_by_class_by_trial = {}
        alg_by_trial = {}
        extrinsic_by_trial = {}

        if 'output' not in episode_by_tag[tag] or (
                not results_are_zip and not os.path.exists(episode_by_tag[tag]['output'])):
            print('WARNING: results for {}-{} not found'.format(tag[0], tag[1]))
            continue

        # load the labels for each trial in the episode
        with open(episode_by_tag[tag]['labels']) as labels_fp:
            for label in labels_fp:
                trial_id, episode_class, raw_class, raw_extrinsic = parse_label(label)
                raw_class_by_episode_class[episode_class] = raw_class
                raw_extrinsic_by_episode_class[episode_class] = raw_extrinsic
                label_by_trial[trial_id] = (episode_class, raw_class, raw_extrinsic)

        with open(episode_by_tag[tag]['test']) as test_fp:
            for trial in test_fp:
                trial_id, ref = trial.strip().split(',')
                alg, _, extrinsic, _ = os.path.splitext(ref)[0].split('-')
                alg_by_trial[trial_id] = alg
                extrinsic_by_trial[trial_id] = extrinsic

        # load the results for trial in the episode
        open_fn = open
        if results_are_zip:
            open_fn = lambda f: io.TextIOWrapper(results_zip.open(f))
        with open_fn(episode_by_tag[tag]['output']) as output_fp:
            for result in csv.DictReader(output_fp):
                score_by_class_by_trial[result['trial-id']] = {}
                for episode_class, raw_class in raw_class_by_episode_class.items():
                    if str(episode_class) in result:
                        score_by_class_by_trial[result['trial-id']][episode_class] = float(result[str(episode_class)])
                    else:
                        print(f'Using default distance (1e-8) for {episode_by_tag[tag]["output"]}')
                        score_by_class_by_trial[result['trial-id']][episode_class] = -1e8

        # gather metrics
        for trial_id in score_by_class_by_trial:
            metrics.add_trial_result(alg_by_trial[trial_id], extrinsic_by_trial[trial_id],
                                     *label_by_trial[trial_id],
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
                trial_id, episode_class, raw_class, raw_extrinsic = parse_label(line)
                raw_classes.add(raw_class)
                episode_classes.add(episode_class)
    return list(episode_classes), list(raw_classes)


def parse_label(entry):
    trial_id, episode_class, raw_class, raw_extrinsic = entry.strip().split(',')
    return trial_id, int(episode_class), int(raw_class), int(raw_extrinsic)


if __name__ == '__main__':
    main()
