# Setup

## 3rd party libraries

```
> pip install -r requirements.txt
```

## Baselines

```
> git clone https://bitbucket.xrcs.jhuapl.edu/scm/mmae/baselines.git
> cd baselines
> ./build-all.sh
```

## Images

```
> python download_images.py
```

# Running an image on an experiment

## Task 2a

```
> python run_2a_experiment.py <directory containing csv files> <directory containing images> <docker image>
```

example:
```
> python run_2a_experiment.py hst_experiment/easy/ images/ microns/random-performer
```

## Task 2b

```
> python run_2b_experiment.py <csv file> <directory containing images> <docker image>
```

example:
```
> python run_2b_experiment.py easy-2b.csv images/ microns/random-performer
```

