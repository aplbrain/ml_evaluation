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
from zipfile import ZipFile


def load_episodes(csvs_dir, episode_by_tag, include):
    for f in os.listdir(csvs_dir):
        full_path = os.path.join(csvs_dir, f)
        if os.path.isdir(full_path):
            load_episodes(full_path, episode_by_tag, include)
        elif os.path.isfile(full_path) and f.endswith('.csv'):
            name, ext = f.split('.')
            data = name.split('-')
            if len(data) == 3:
                experiment_tag, phase, id = data
            else:
                experiment_tag, phase = data
                id = '0000'
            tag = (experiment_tag, id)
            if len(include) == 0 or tag in include:
                episode_by_tag[tag][phase] = os.path.abspath(full_path)


def load_outputs(results_dir, episode_by_tag):
    for f in os.listdir(results_dir):
        full_path = os.path.join(results_dir, f)
        if os.path.isdir(full_path):
            load_outputs(full_path, episode_by_tag)
        elif os.path.isfile(full_path) and f == 'output.csv':
            tag = tuple(os.path.basename(results_dir).split('-'))
            if tag in episode_by_tag:
                episode_by_tag[tag]['output'] = os.path.abspath(full_path)


def load_outputs_from_zip(results_zip_path, episode_by_tag):
    with ZipFile(results_zip_path) as results_zip:
        for f in results_zip.infolist():
            if 'output.csv' in f.filename:
                tag = tuple(os.path.basename(os.path.dirname(f.filename)).split('-'))
                if tag in episode_by_tag:
                    episode_by_tag[tag]['output'] = f.filename
