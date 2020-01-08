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

cwlVersion: v1.0

class: CommandLineTool

inputs:
  taskInfoFile: File
  trialsFile: File
  dataDirPath: string
  paramsDir: Directory
  imageName: string
  runtime: string
  user: string

outputs:
  results:
    type: File
    outputBinding:
      glob: output.csv

requirements:
  - class: InitialWorkDirRequirement
    listing:
      - entry: ""
        entryname: output.csv
        writable: true
  - class: InlineJavascriptRequirement

baseCommand: [docker, run, -i, --rm]
arguments:
#  - --user=$(inputs.user)
  - valueFrom: $("--runtime=" + inputs.runtime)
  - --volume=$(inputs.taskInfoFile.path):/config/config.json:ro
  - --volume=$(inputs.trialsFile.path):/input/input.csv:ro
  - --volume=$(inputs.dataDirPath):/data:ro
  - --volume=$(inputs.paramsDir.path):/params:ro
  - --volume=$(runtime.outdir):/output
  - $(inputs.imageName)

