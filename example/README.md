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
> microns-2a-analyze TODO
```

### 2B

```bash
> cd example/example_2b
> microns-2b-run ./csvs ./images microns/random-performer
> microns-2b-analyze TODO
```