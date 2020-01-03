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

class: Workflow

requirements:
  - class: StepInputExpressionRequirement
  - class: InlineJavascriptRequirement

inputs:
  testInfo: {type: File, default: {class: File, path: ../config/2b_test.json}}
  testData: File
  dataDirPath: string
  paramsDir: Directory
  imageName: string
  runtime: string
  user: string

outputs:
  results:
    type: File
    outputSource: test/results

steps:
  test:
    run: test_algorithm.cwl
    in:
      taskInfoFile: testInfo
      trialsFile: testData
      dataDirPath: dataDirPath
      paramsDir: paramsDir
      imageName: imageName
      runtime: runtime
      user: user
    out: [results]
