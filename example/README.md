# MICrONS Evaluation Example

## Create the docker container

```bash
> cd example/random-performer
> docker build -t microns/random-performer .
```

Or

```bash
> cd example/random-performer
> ./build.sh
```

## Run one of the example experiments

### 2A

```bash
> cd example/example_2a
> microns-2a-run ./csvs ./images microns/random-performer
> microns-2a-analyze ./csvs ./results-random-performer/task_2a/example_2a
```

### 2B

```bash
> cd example/example_2b
> microns-2b-run ./csvs ./images microns/random-performer
> microns-2b-analyze ./csvs ./results-random-performer/task_2b/example_2b
```

## Modifying the experiments

1. Images

To add or edit images, simply add new images to the images directory,
and edit the csvs to include your new images.

The images need to have a name according to this scheme:

    <descriptor>-<class id>-<modifier>-<modifier id>.png

The example images are named:

    mnist-<class id>-none-0.png
    
Because they are from the mnist data set, and have no modifiers.

2. Episodes

An episode is a set of csvs under the csvs/ directory. For Task 2A,
it is a set of 4 csvs including one for training, online training,
testing, and labels for the tests. For Task 2B it is a set of 2 csvs
including one for testing and labels for the tests.

Episodes are identified by their 4 digit identifier at the end of the
filename. For example:

    example_2a-labels-0000.csv

Is episode 0000 for example_2a.

3. Trials

An episode is comprised of multiple trials, each one on its own line
in the episode CSV file. Trials are identified by a tag like:

    trial-<4 digit id>

To add a new trial add a new line to one set of CSV for an episode.