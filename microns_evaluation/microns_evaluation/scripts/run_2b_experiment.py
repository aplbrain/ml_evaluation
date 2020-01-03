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

import pkg_resources
import os
import subprocess
import argparse
import datetime
import shutil
import boto3
import tempfile
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs_dir', help='Directory containing all the csvs to use for the experiment.')
    parser.add_argument('images_dir', help='Directory containing all the images used')
    parser.add_argument('container', help='The tag of the docker image to use for the experiment')
    parser.add_argument('--initial-params', default='./empty-params', help='Directory containing a pretrained model.')
    parser.add_argument('--user', default='1000:1000',
                        help='The user and group id that the docker image should run as. Defaults to 1000:1000')
    parser.add_argument('--one', default=False, action='store_true', help='Only run one of the csvs found in csvs_dir.')
    parser.add_argument('--use_gpu', default=False, action='store_true',
                        help='Direct the CWL to use the GPU (use nvidia as the docker runtime).')
    parser.add_argument('--verbose', default=False, action='store_true', help='Enable verbose CWL output.')
    parser.add_argument('--s3-zip', default=None,
                        help='The name of the zip file to upload to s3. If not specified, no upload will take place.')
    args = parser.parse_args()

    params_dir = os.path.abspath(args.initial_params)
    cwl_file = pkg_resources.resource_filename('microns_evaluation', 'cwl/task_2b.cwl')
    images_dir = os.path.abspath(args.images_dir)

    episode_by_tag = defaultdict(dict)
    for path in os.listdir(args.csvs_dir):
        name, ext = path.split('.')
        experiment_tag, mode = name.split('-')
        tag = (experiment_tag, '0000')
        episode_by_tag[tag][mode] = os.path.abspath(os.path.join(args.csvs_dir, path))

    container_name = args.container.split('/')[-1]

    results_dir = 'results-{}'.format(container_name)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    os.chdir(results_dir)

    task_dir = 'task_2b'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    os.chdir(task_dir)

    for i, (tag, episode) in enumerate(episode_by_tag.items()):
        print('[{}] {} / {} | {}-{}'.format(datetime.datetime.now(), i + 1, len(episode_by_tag), tag[0], tag[1]))

        experiment_tag, episode_id = tag

        if not os.path.exists(experiment_tag):
            os.mkdir(experiment_tag)
        os.chdir(experiment_tag)

        if os.path.exists('-'.join(tag)):
            os.chdir('..')
            continue

        os.mkdir('-'.join(tag))
        os.chdir('-'.join(tag))

        cmd = ['cwl-runner',
               '--no-container',
               cwl_file,
               '--testData', episode['test'],
               '--dataDirPath', images_dir,
               '--paramsDir', params_dir,
               '--imageName', args.container,
               '--runtime', 'nvidia' if args.use_gpu else '',
               '--user', args.user,
               ]
        if not args.verbose:
            cmd.insert(1, '--quiet')

        output = subprocess.check_output(cmd)

        os.chdir('..')
        os.chdir('..')

        if args.one:
            quit()

    if args.s3_zip is not None:
        s3 = boto3.client('s3')
        with tempfile.NamedTemporaryFile() as fp:
            archive_filename = shutil.make_archive(fp.name, 'zip')
            s3.upload_file(archive_filename, 'microns-phase2-results', args.s3_zip)

    os.chdir('..')


if __name__ == '__main__':
    main()
