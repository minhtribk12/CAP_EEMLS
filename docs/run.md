# Instructions for Running EEMLS

## Overview

- Parts 1 to 4: done once.
- Parts 5 to 9: tests on the development machine.
- Parts 10 to 12: runs on edge devices.

| Part | Purpose |
|---|---|
| 1. Prerequisites | Prepare the software and the machines. |
| 2. Setup | Install the package and create the local configuration files. |
| Quick start | Run the whole system once on this machine, with public models only. |
| 3. Dataset | Download the images and build the image store. |
| 4. Models | Download the backbones, tune the model variations, and measure their profiles. |
| 5. Quick test run on local machine | Run the whole system on one machine. |
| 6. Container images | Build the service images. |
| 7. Run configuration | What a run is made of, and how to set it: settings, overrides, scenarios, experiment plans. |
| 8. Local test with Docker | Run the system on one machine with Docker. |
| 9. Local test with minikube | Publish the images, then run the system on a local Kubernetes cluster. |
| 10. Kubernetes cluster | Install k3s on the devices and prepare each node. |
| 11. Runs | Measure the devices, then run a scenario. |
| 12. Shutdown | Stop the system and delete the cluster. |
| 13. Results | Where the results are stored. |
| 14. Known errors | Errors seen so far, and how to fix them. |

- Commands run on the development machine, in the repository root, unless a step says otherwise.
- `<...>` marks a value to fill in, such as `<host>`.

## 1. Prerequisites

### 1.1 Development machine

An x86 Linux machine. It builds the images and drives every run. It needs:

- Python 3.11 or 3.12 and Poetry
- Docker with the buildx plugin
- kubectl
- rsync
- minikube, a local Kubernetes, for Part 9 only

### 1.2 Devices

Edge devices that run a Kubernetes-like environment (k3s), reachable over SSH with sudo rights.
A device with an NVIDIA GPU also needs the NVIDIA Container Toolkit.
A Jetson board that builds the Jetson image (6.4) also needs Docker with the buildx plugin.

### 1.3 Database password

- The metrics database password comes from the environment, never from a file.
- Set it in each shell:

```bash
export EEMLS_DATABASE_PASSWORD=<password>
```

Example: check it is set, without printing it.

```bash
export EEMLS_DATABASE_PASSWORD=db_password
echo "${EEMLS_DATABASE_PASSWORD:+set}"   # prints: set
```

## 2. Setup

### 2.1 Install the package

Install the framework, every service and the tools, into one environment.

```bash
make install
```

- The framework is [`src/eemls/`](../src/eemls/); each service is a folder of [`application/`](../application/); the tools are [`tools/`](../tools/).
- Each service and tool gives its own command, such as `eemls-config`, `eemls-exp`, `eemls-warm`, `eemls-collect` and `eemls-load`.

Example: check the install.

```bash
poetry run python3 -c "import eemls; print('ok')"   # prints: ok
```

### 2.2 Create files for model and image build

The system needs two files in [`config/local/`](../config/local/).

| File | Holds |
|---|---|
| [`config/local/models.yaml`](../config/local/models.yaml) | - each model id's repository, file and SHA-256<br>- to download and check each model |
| [`config/local/images.yaml`](../config/local/images.yaml) | - the registry and the image version<br>- to push and pull the images |

- [`models.yaml`](../config/local/models.yaml):
  - read when models are downloaded (4.1)
  - written when variations are tuned (4.2)
  - read by each inference service to find its file
- [`images.yaml`](../config/local/images.yaml): read when the images are published (9.1) and pulled.

**Step 1. Create both files.** Choose one way.

- **A. From the examples** in [`examples/local/`](../examples/local/): eight public backbones as `b1` to
  `b8`. Every later step runs with them.

  ```bash
  mkdir -p config/local
  cp examples/local/models.yaml examples/local/images.yaml config/local/
  ```

- **B. Empty**, with every id of [`config/models/`](../config/models/) listed and its fields
  blank:

  ```bash
  poetry run python3 tools/new_local.py
  # prints: <repository>/config/local: fill in models.yaml and images.yaml
  ```

  | Option | Meaning |
  |---|---|
  | `--config <folder>` | write into `<folder>/local/` of another configuration tree; default [`config/`](../config/) |

Either way, the files are in [`config/local/`](../config/local/). Every tool reads them there.

The services read a copy of [`models.yaml`](../config/local/models.yaml) on each node (8, 9.4, 10.5).

**Step 2. Fill in [`models.yaml`](../config/local/models.yaml).** Fields of each backbone (`base_models`) and model (`models`):

| Field | Meaning | Example |
|---|---|---|
| `name` | backbones only: the backbone's own name | `mnasnet_100` |
| `hugging_face_id` | the Hugging Face repository the file is downloaded from | `onnxmodelzoo/mnasnet_100_Opset17` |
| `onnx_file` | the ONNX file in that repository | `mnasnet_100_Opset17.onnx` |
| `sha256` | the file's digest; a download that does not match is refused | `0c3f9bc4...dc2915` |

```yaml
base_models:
  b1:
    name: mnasnet_100
    hugging_face_id: onnxmodelzoo/mnasnet_100_Opset17
    onnx_file: mnasnet_100_Opset17.onnx
    sha256: 0c3f9bc44239843585bebfa9ead904173e7b95766c4b77b432e1afe660dc2915
models: {}
```

`models` holds the tuned variations; 4.2 fills it in.

**Step 3. Fill in [`images.yaml`](../config/local/images.yaml).**

| Field | Meaning | Example |
|---|---|---|
| `registry` | where the images are pushed and pulled | `docker.io/example` |
| `version` | the tag of every image | `latest` |

```yaml
registry: docker.io/example
version: latest
```

Filled examples: [`examples/local/`](../examples/local/).

Quick example, after way A: download the two backbones.

```bash
poetry run python3 tools/fetch_models.py b1 b2
```

To add a model:

1. Add a backbone to [`config/models/variations.yaml`](../config/models/variations.yaml), under `base_models` and `specialisms`.
2. Give it its source under `base_models` in [`config/local/models.yaml`](../config/local/models.yaml).
3. Download it (4.1), build its variations (4.2) and measure them (4.3).

## Quick start

**Purpose.** Run the whole system once on this machine, with public models only. It takes about 30 minutes on a first run; most of it is the first install and image build.

- It serves two public example backbones, `b1` and `b2`, as they are: no variations are tuned.
- Their sources and SHA-256 are the ones in [`examples/local/models.yaml`](../examples/local/models.yaml).
- They answer the 1000 ImageNet classes. [`examples/quick_start/imagenet_class_mapping.yaml`](../examples/quick_start/imagenet_class_mapping.yaml) gives the ImageNet outputs of each of our classes; `pedestrian` has none.
- Each model in [`examples/quick_start/overrides.yaml`](../examples/quick_start/overrides.yaml) names that file by `class_mapping`. `eemls-config render` turns it into the model's `classes`; every other output is `output_<index>`.
- It uses 200 Open Images day images, a 120 s scenario ([`quick_start`](../experiments/scenarios/quick_start.yaml)) and CAP with seed 0 ([`experiments/runs/quick_start.yaml`](../experiments/runs/quick_start.yaml)).
- Needs: 1.1 (no minikube) and 1.3. No edge device, no multispectral dataset.
- Needs the example local files (step 2). With tuned models in [`config/local/models.yaml`](../config/local/models.yaml), step 3 is refused.

| Step | Time measured here |
|---|---|
| 1. Install | 10 s with the packages already cached; minutes on a first install |
| 2. Create the local files | under 1 s |
| 3. Write the quick start tree | 1 s |
| 4. Download the two models | 6 s |
| 5. Download 200 images | 3 min |
| 6. Build the image store | 5 s |
| 7. Measure the profiles | 4 s |
| 8. Run CAP | 5 min with the images already built; a first build takes longer |

**1. Install** (as 2.1).

```bash
make install
```

**2. Create the local files** from the examples (as 2.2, way A).

```bash
mkdir -p config/local
cp examples/local/models.yaml examples/local/images.yaml config/local/
```

**3. Write the quick start tree**: the pool of two services, 5 requests/s per consumer, 20 images per main class and 10 per other class.

```bash
poetry run eemls-config render config --overrides examples/quick_start/overrides.yaml --out runs/quick_start/config
```

**4. Download the two models** to [`models/b1_imagenet/`](../models/b1_imagenet/) and [`models/b2_imagenet/`](../models/b2_imagenet/).

```bash
poetry run python3 tools/fetch_models.py --config runs/quick_start/config b1_imagenet b2_imagenet
```

**5. Download the images** to [`data/open_images/`](../data/open_images/). Only two small Open Images splits are read.

```bash
poetry run python3 tools/fetch_dataset.py --sources runs/quick_start/config/datasets/sources.yaml --splits validation test
```

**6. Build the image store** in [`data/hub/`](../data/hub/). It prints the night and multispectral classes as `missing`; the quick start does not use them.

```bash
poetry run python3 tools/prepare_dataset.py --sources runs/quick_start/config/datasets/sources.yaml
```

**7. Measure the profiles** into [`models/profiles.jsonl`](../models/profiles.jsonl).

```bash
poetry run python3 tools/profile_pool.py --config runs/quick_start/config --images 200
```

**8. Run CAP** for 120 s (as Part 5). It builds the images, runs, collects the rows, and removes the containers.

```bash
export EEMLS_DATABASE_PASSWORD=<password>   # as in 1.3
PLAN=experiments/runs/quick_start.yaml OVERRIDES=examples/quick_start/overrides.yaml make experiment
```

The rows are in [`runs/local_<time>_quick_start/quick_start/cap/seed-0/rows/`](../runs/). Here: 1797 queries, 932 of them with feedback, 357 decisions.

## 3. Dataset

### 3.1 Download the Open Images subset

**Purpose.** Download the images of the consumers' classes.

Check the counts, then download to [`data/open_images/`](../data/open_images/).

```bash
poetry run python3 tools/fetch_dataset.py --count
poetry run python3 tools/fetch_dataset.py
```

- The full download is about 17,800 images and 5.4 GB.
- It takes several hours: about 1 image per second.
- It prints `N downloaded, N already there, N failed`. Any failure ends it with an error; run it again to retry.
- Images in a class folder that it does not pick move to [`data/open_images/unpicked/`](../data/open_images/unpicked/)
  and are not used.
- Parts 5 and 8 need every class by day, and the multispectral classes by night (3.2).

| Parameter | Meaning |
|---|---|
| `--count` | only print the available and wanted count per class; download nothing |
| `--classes <class> ...` | download only these classes |
| `--splits <split> ...` | the Open Images splits to read; default `train validation test` |
| `--seed <n>` | the seed that chooses which images are taken; default 20261016 |
| `--into <folder>` | where the images are written; default [`data/open_images/`](../data/open_images/) |
| `--sources <file>` | the dataset sources file; default [`config/datasets/sources.yaml`](../config/datasets/sources.yaml) |

| Purpose | Command |
|---|---|
| see what is available | `fetch_dataset.py --count` |
| add one class later | `fetch_dataset.py --classes cyclist` |
| a different sample of images | `fetch_dataset.py --seed 7 --into data/open_images_7` |

### 3.2 Download the multispectral dataset

**Purpose.** Night-time images.

1. Download it manually from the page in [`config/datasets/sources.yaml`](../config/datasets/sources.yaml) into
   [`data/multispectral/`](../data/multispectral/). It holds [`images/`](../data/multispectral/images/),
   [`labels/`](../data/multispectral/labels/), [`train.txt`](../data/multispectral/train.txt),
   [`val.txt`](../data/multispectral/val.txt), [`test.txt`](../data/multispectral/test.txt) and
   [`black_list.txt`](../data/multispectral/black_list.txt).
2. Build its 224-pixel crops in [`data/mfnet_224/<day|night>/<class>/`](../data/mfnet_224/):

   ```bash
   poetry run python3 tools/prepare_multispectral.py
   ```

| Parameter | Meaning |
|---|---|
| `--multispectral <folder>` | the downloaded dataset; default [`data/multispectral/`](../data/multispectral/) |
| `--sources <file>` | the dataset sources file; default [`config/datasets/sources.yaml`](../config/datasets/sources.yaml) |
| `--into <folder>` | where the crops are written; default [`data/mfnet_224/`](../data/mfnet_224/) |
| `--size <n>` | the crop size in pixels; default `image_size` in [`sources.yaml`](../config/datasets/sources.yaml) |

- Each image takes the class covering most of it; an image where that class covers under 0.5% is left out.
- Images on the black list or with an empty mask are left out.

Example: [`data/mfnet_224/night/car/`](../data/mfnet_224/night/car/) holds the night crops of `car`.

### 3.3 Build the image store

**Purpose.** Build the image store the data hub serves, in [`data/hub/`](../data/hub/). Each image gets an
id, a class, a condition (day or night) and a quality (clear or blurred).

Check, then build:

```bash
poetry run python3 tools/prepare_dataset.py --check
poetry run python3 tools/prepare_dataset.py
```

| Parameter | Meaning |
|---|---|
| `--check` | print what would be built, and write nothing |
| `--open-images <folder>` | the Open Images download; default [`data/open_images/`](../data/open_images/) |
| `--imagenet <folder>` | (optional) |
| `--multispectral <folder>` | the multispectral crops; default [`data/mfnet_224/`](../data/mfnet_224/) |
| `--into <folder>` | the image store written; default [`data/hub/`](../data/hub/) |
| `--sources <file>` | the dataset sources file; default [`config/datasets/sources.yaml`](../config/datasets/sources.yaml) |

Example: count the images and read the first.

```bash
wc -l data/hub/manifest.jsonl
head -1 data/hub/manifest.jsonl
```

An id is source, condition, quality and source file name: `open_images-day-clear-ba366d620b2c3914`.

## 4. Models

### 4.1 Download the backbones

**Purpose.** Download the correct backbone for each id, to [`models/base/<id>.onnx`](../models/base/).

```bash
poetry run python3 tools/fetch_models.py b1
poetry run python3 tools/fetch_models.py
```

| Parameter | Meaning |
|---|---|
| `<id> ...` | the backbones to download, e.g. `b1`; all of them when none is given |
| `--config <folder>` | the configuration tree that names the models; default [`config/`](../config/) |
| `--into <folder>` | where the files are written; default [`models/`](../models/) |

Example: `fetch_models.py b1` writes [`models/base/b1.onnx`](../models/base/b1.onnx). A correct file is kept.

### 4.2 Tune the model variations

**Purpose.** Build the generalists and specialists of [`config/models/variations.yaml`](../config/models/variations.yaml) from the
backbones. A specialist weighs some classes more. Needs [`data/hub/`](../data/hub/).

- `--probe`: tune two, print the accuracy per class, write nothing.
- `--build`: tune all; write each to [`models/<variation>/`](../models/) and its source to
  [`config/local/models.yaml`](../config/local/models.yaml); rewrite
  [`model_repository.yaml`](../config/models/model_repository.yaml) and [`variation_profiles.yaml`](../config/models/variation_profiles.yaml);
  list the training images in [`data/hub/trained.txt`](../data/hub/trained.txt).
- Edit only [`variations.yaml`](../config/models/variations.yaml). The other files are written here.

```bash
poetry run python3 tools/build_variations.py --probe
poetry run python3 tools/build_variations.py --build
```

| Parameter | Meaning |
|---|---|
| `--probe` | tune two variations and print what they reach; write nothing |
| `--build` | tune every variation of the recipe and write it |
| `--backbones <id> ...` | for `--probe`: the backbones tried; default `b3 b6` |
| `--specialism <class> ...` | for `--probe`: the classes the specialist weighs more; default `cyclist motorcycle` |
| `--weight <w>` | for `--probe`: how much more those classes weigh; default 8 |
| `--per-class <n>` | for `--probe`: images per class used for training; default 120 |
| `--seed <n>` | for `--probe`: the seed of the training sample; default 0 |

`--build` takes both from [`variations.yaml`](../config/models/variations.yaml): `train_images_per_class` and `seed`.

| Purpose | Command |
|---|---|
| check a backbone is worth using | `build_variations.py --probe --backbones b1` |
| try a stronger specialist | `build_variations.py --probe --specialism pedestrian --weight 16` |
| a faster, rougher build | lower `train_images_per_class` in [`variations.yaml`](../config/models/variations.yaml), then `--build` |

### 4.3 Benchmarking - profiling

**Purpose.** Measure each service's accuracy, time and cost on this machine, into
[`models/profiles.jsonl`](../models/profiles.jsonl). Copies are sized from it; policies start from it. Cluster profiles: 11.2.

```bash
poetry run python3 tools/profile_pool.py --images 200
poetry run python3 tools/profile_pool.py
```

| Parameter | Meaning |
|---|---|
| `--images <n>` | clear images measured, spread over the classes, none of them training images; default 10000 (hours on 16 services); 200 takes about five minutes |
| `--seed <n>` | the seed that chooses which images are measured; default 20261016 |
| `--config <folder>` | the configuration tree that names the pool; default [`config/`](../config/) |
| `--into <file>` | where the profiles are written; default [`models/profiles.jsonl`](../models/profiles.jsonl) |
| `--at <service>=<url>` | measure a running service there instead of on this machine (11.2) |

Example: measure into another file, to compare.

```bash
poetry run python3 tools/profile_pool.py --images 200 --into runs/profiles_check.jsonl
```


## 5. Quick test run on local machine

**Purpose.** A short end-to-end check. Needs Parts 1 to 4.

`make experiment` does, on this machine:

1. Builds the x86 images (as 6.1).
2. Starts every service as a container, in a one-node Docker Swarm.
3. Runs each approach and seed (here CAP, 60 s). After each run the control plane stops, so its last rows reach the Observation Service. Then the Observation Service stops. Then the results are collected.
4. Removes the containers and the run's database volume; the results stay in the `OUT` folder.

```bash
export EEMLS_DATABASE_PASSWORD=<password>   # as in 1.3
PLAN=experiments/runs/recurring_ten_days.yaml APPROACHES=cap SEEDS=0 DURATION_S=60 make experiment
```

| Parameter | Meaning |
|---|---|
| `PLAN` | the experiment plan file in [`experiments/runs/`](../experiments/runs/) or [`experiments/archived/runs/`](../experiments/archived/runs/): its scenario, approaches and seeds |
| `SCENARIOS` | the folder of scenario files; default the `scenarios/` folder beside the plan's `runs/` folder |
| `APPROACHES` | the approaches to run, comma separated; all approaches of the plan when unset |
| `SEEDS` | the seeds to run, comma separated; all seeds of the plan when unset |
| `DURATION_S` | the length of each run in seconds; the plan's length when unset |
| `OUT` | the results folder; default [`runs/local_<time>_<plan>`](../runs/) |
| `OVERRIDES` | the overrides every run starts from; default [`deploy/one_host/overrides.yaml`](../deploy/one_host/overrides.yaml) |
| `SERVICE_PROFILES` | the profiles the copies are sized from; default [`models/profiles.jsonl`](../models/profiles.jsonl) |
| `WARM_IMAGES` | images sent to each service before each run; default 50 |

The results are in [`runs/local_<time>_recurring_ten_days/recurring_ten_days/cap/seed-0/rows/`](../runs/). Each run gets a new
folder; a folder given as `OUT` must be new.

## 6. Container images

**Purpose.** Every service runs from an image, on Docker, minikube and k3s. An image holds only its
own code and its dependencies. The config folder (at `/opt/eemls/config`), the scenario files, the
models and the dataset images are mounted at run time, never built in. A change to them needs no
new image.

- Each service image builds from its own folder in [`application/`](../application/), with that folder's
  `Dockerfile`.
- The image build makes the framework wheel itself, from the `framework` build context, and installs
  it only if its hash is the one in that folder's `framework.txt`, such as
  [`application/control-plane/framework.txt`](../application/control-plane/framework.txt).
- The `framework` context is the repository root for an app that pins the root version, and a
  checkout of the git tag `eemls-v<version>` for an older version. `make images` and
  `make images-local` prepare it through [`tools/pin.py`](../tools/pin.py) `--bake`, and refuse an app
  whose tag is missing.
- A build is byte-identical, so the wheel built in the image matches `framework.txt`.
- A change under [`src/eemls/`](../src/eemls/) reaches an app after a pin; a release tags it:

  ```bash
  make pin APP=control-plane VERSION=1.0.0
  make release VERSION=1.0.0
  ```

  | Command | Does |
  |---|---|
  | `make pin APP=<app> VERSION=<version>` | builds the wheel of that version once, writes its pin and hash into [`application/<app>/framework.txt`](../application/), points the app at that version and relocks it; a version other than the root one needs its tag |
  | `make release VERSION=<version>` | tags the commit `eemls-v<version>`; refused unless the version is the one of [`pyproject.toml`](../pyproject.toml) and nothing under [`src/`](../src/) or in [`pyproject.toml`](../pyproject.toml) is uncommitted |

| Image | Runs on | Built on | Build target |
|---|---|---|---|
| `eemls-control-plane`, `eemls-observation-service`, `eemls-data-hub`, `eemls-consumer` | the core node, x86 or ARM | this machine | `control-plane`, `observation-service`, `data-hub`, `consumer` |
| `eemls-inference-service:cpu-amd64` | an x86 CPU | this machine | `inference-service-cpu-amd64` |
| `eemls-inference-service:cpu-arm64` | an ARM CPU, such as a Raspberry Pi | this machine, with emulation | `inference-service-cpu-arm64` |
| `eemls-inference-service:cuda-amd64` | an x86 machine with an NVIDIA GPU | this machine | `inference-service-cuda-amd64` |
| `eemls-inference-service:jetson` | an NVIDIA Jetson GPU | a Jetson board | `inference-service-jetson` |

- Each device type has an execution provider ([`cost/prices.yaml`](../config/cost/prices.yaml)); each provider has an
  image tag (`image_tags` in [`services/deployment.yaml`](../config/services/deployment.yaml)).
- Every inference service image runs ONNX Runtime: the CPU build on CPUs, the CUDA build on GPUs.
- The build targets are in [`docker-bake.hcl`](../docker-bake.hcl).

### 6.1 Build the x86 images for this machine

Build the core images and the x86 CPU inference image into this machine's Docker, for Parts 5
and 8.

```bash
make images-local
```

Example: list the images built.

```bash
docker images | grep eemls-
```

### 6.2 Prepare the ARM build (once)

An x86 machine builds ARM images through emulation, with a builder for both architectures.

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx create --use --name eemls
```

If the builder already exists: `docker buildx use eemls`.

### 6.3 Build one image for one kind of hardware

Build one inference image into this machine's Docker, to test it before publishing.

```bash
REGISTRY=<registry> VERSION=<version> poetry run python tools/pin.py --bake --load <build target>
```

| Parameter | Meaning |
|---|---|
| `REGISTRY`, `VERSION` | the image name prefix and tag; as in [`config/local/images.yaml`](../config/local/images.yaml) |
| `<build target>` | a build target of the table above |

| Hardware | Command |
|---|---|
| ARM CPU | `REGISTRY=docker.io/example VERSION=latest poetry run python tools/pin.py --bake --load inference-service-cpu-arm64` |
| x86 with an NVIDIA GPU | `REGISTRY=docker.io/example VERSION=latest poetry run python tools/pin.py --bake --load inference-service-cuda-amd64` |

Example: the ARM image is `docker.io/example/eemls-inference-service:cpu-arm64-latest`.

### 6.4 Build the Jetson image on a Jetson board

- The Jetson image needs JetPack's CUDA, cuDNN and TensorRT, and an ONNX Runtime built for the board.
- It is built on the board, then brought to this machine.

1. Copy the repository's own files to the board, from the repository folder. They hold the framework
   source the image builds its wheel from, so the inference service must pin the root version. This leaves out `.env*`, [`.git/`](../.git/), [`config/local/`](../config/local/), [`models/`](../models/), [`data/`](../data/), [`runs/`](../runs/) and [`.venv/`](../.venv/).

   The first line removes an older framework source, so the wheel holds only the files of this version.

   ```bash
   ssh <username>@<board> 'rm -rf eemls/src'
   git ls-files -co --exclude-standard | grep -v '^\.env' \
     | rsync -a --files-from=- ./ <username>@<board>:eemls/
   ```

2. Once: build ONNX Runtime for the board into [`application/inference-service/onnxruntime-wheels/`](../application/inference-service/onnxruntime-wheels/). It takes hours on an Orin Nano. Start it
   detached, so it keeps going when SSH drops:

   ```bash
   ssh <username>@<board> 'cd eemls && (nohup setsid docker buildx bake onnxruntime-jetson > ../ort-build.log 2>&1 < /dev/null &)'
   ```

   - Follow it: `ssh <username>@<board> tail -f ort-build.log`.
   - It is done when `eemls/application/inference-service/onnxruntime-wheels/` on the board holds a `.whl` file.
   - Faster on a Jetson AGX Orin: add `--set onnxruntime-jetson.args.PARALLEL=4` after `onnxruntime-jetson`.

3. On the board: build the image.

   ```bash
   cd eemls && REGISTRY=<registry> VERSION=<version> docker buildx bake --load inference-service-jetson
   ```

4. Bring the image to this machine:

   ```bash
   ssh <username>@<board> 'docker save <registry>/eemls-inference-service:jetson-<version> | gzip -1' \
     | gunzip | docker load
   ```

   - This brings only the `jetson-<version>` tag. The later steps use only that tag.

- The board must run JetPack 6 with L4T R36.4.x, the same line as `jetpack_release` in
  [`services/deployment.yaml`](../config/services/deployment.yaml). A board on 36.4.7 works.
- Check the board: `ssh <username>@<board> cat /etc/nv_tegra_release` shows `R36` and `REVISION: 4.x`.
- More detail: [`application/inference-service/README.md`](../application/inference-service/README.md#jetson-tag).

Example: a Jetson Orin, `registry: docker.io/example`, `version: latest`:

```bash
ssh <username>@<board> 'cd eemls && REGISTRY=docker.io/example VERSION=latest docker buildx bake --load inference-service-jetson'
ssh <username>@<board> 'docker save docker.io/example/eemls-inference-service:jetson-latest | gzip -1' | gunzip | docker load
```

## 7. Run configuration

A run is set by six things. Each has its default in [`config/`](../config/); a run changes them
with an overrides file (7.3) or an experiment plan (7.5).

### 7.1 What a run is made of

| What | Sets | Default in | Changed for a run by |
|---|---|---|---|
| Pool | models, device types, copies | [`models/inference_services.yaml`](../config/models/inference_services.yaml) | an overrides file |
| Consumers | contract: CDM weights, cost weight, request rate, requirements, feedback; starting ensemble | [`consumers/<consumer>/`](../config/consumers/) | an overrides file |
| Approach | an approach file, and optionally another contract; the file's `kind` names an installed approach | [`approaches/<name>.yaml`](../config/approaches/) | `approaches:` in an experiment plan, or `APPROACHES` |
| Scenario | how queries change: class mix, day or night, quality drops, surges | [`experiments/scenarios/`](../experiments/scenarios/) | `scenario:` in an experiment plan, or `SCENARIO` |
| Length and seed | how long it runs, and which random draw | `recurring_ten_days`, 18000 s, seed 0 | `duration_s:` and `seeds:` in an experiment plan, or `DURATION_S` (for `make run`) and `SEED` (for `make up`) |
| Profiles | each service's measured speed, accuracy and cost | [`models/profiles.jsonl`](../models/profiles.jsonl) | `SERVICE_PROFILES` |

- Less often: time limits, memory and the decision interval ([`services/*.yaml`](../config/services/)); device
  prices and power ([`cost/prices.yaml`](../config/cost/prices.yaml)).
- Each service's copies must carry all consumers' request rates together, or the run is refused.

### 7.2 Settings

All settings are YAML files under [`config/`](../config/).

| File | Sets |
|---|---|
| [`consumers/<consumer>/cdm.yaml`](../config/consumers/) | the CDMs the consumer declares: how each is measured, its window and unit |
| [`consumers/<consumer>/contract.yaml`](../config/consumers/) | the CDMs a consumer scores, their weights, its cost weight and its request rate |
| [`consumers/<consumer>/contract_<name>.yaml`](../config/consumers/) | another contract of the same consumer, chosen by an approach |
| [`consumers/<consumer>/initial_ensemble.yaml`](../config/consumers/) | the ensemble the consumer starts on |
| [`models/inference_services.yaml`](../config/models/inference_services.yaml) | the pool: the model and device type of each service, and its copies |
| [`approaches/<name>.yaml`](../config/approaches/) | an approach file: the settings of one policy; `kind` names an installed approach (entry-point group `eemls.approaches`) |
| [`services/*.yaml`](../config/services/) | ports, time limits, memory, the policy, probes, profile updates, the broker and the database |
| [`cost/prices.yaml`](../config/cost/prices.yaml) | device prices and power, and the cost unit |

Some settings:

| File | Setting | Meaning |
|---|---|---|
| [`services/control_plane.yaml`](../config/services/control_plane.yaml) | `probe_fraction` | the share of queries also sent to the idle services, so their profiles stay current |
| | `profile_update_interval_s` | seconds between two profile updates |
| [`services/observation_service.yaml`](../config/services/observation_service.yaml) | `url`, `port` | where the other services reach the Observation Service; port 8083 |
| | `request_timeout_ms` | how long a service waits for the Observation Service |
| | `profile_window_s` | the seconds of answers each profile update reads |
| | `profile_min_labeled_per_class` | labeled answers a class needs before its accuracy is updated |
| | `profile_min_labeled_overall` | labeled answers a service needs before its general accuracy is updated |
| [`services/consumer.yaml`](../config/services/consumer.yaml) | `subscription_wait_s` | seconds a consumer waits for its result subscription before it stops |
| [`approaches/cap.yaml`](../config/approaches/cap.yaml) | `night_confidence_drop` | how far the mean confidence must fall below its day level for CAP to read night |
| [`approaches/axiomvision.yaml`](../config/approaches/axiomvision.yaml) | `accuracy_target` | the estimated accuracy at which AxiomVision stops adding members |
| [`approaches/frugalml.yaml`](../config/approaches/frugalml.yaml) | `failing_intervals` | decisions in a row below the quality level before FrugalML changes the ensemble |


### 7.3 Change settings for one run

**Purpose.** Change settings for one run without editing [`config/`](../config/).

**Step 1. Copy an example** from [`examples/overrides/`](../examples/overrides/):

| Example | Changes |
|---|---|
| [`small_pool.yaml`](../examples/overrides/small_pool.yaml) | a small pool on a Raspberry Pi 4 and a Beelink BT3, a lower request rate for one consumer, a longer member time limit |
| [`cluster_gpu.yaml`](../examples/overrides/cluster_gpu.yaml) | a cluster pool on Jetson Orin Nano and Jetson Nano GPUs, Raspberry Pi 4 and Beelink BT3 CPUs, at 7 requests/s a consumer |

```bash
cp examples/overrides/small_pool.yaml my_overrides.yaml
```

**Step 2. Change it.** Each top-level key is a file under [`config/`](../config/); below it are the keys
to replace.

```yaml
models/inference_services.yaml:
  services:
    infer-b1-general-pi4:
      model: b1_general
      device_type: raspberry_pi4
      replicas: 2
    infer-b1-pedestrian-bt3:
      model: b1_pedestrian
      device_type: beelink_bt3
      replicas: 1
consumers/c1_navigation/initial_ensemble.yaml:
  members:
    - infer-b1-general-pi4
    - infer-b1-pedestrian-bt3
consumers/c2_safety/initial_ensemble.yaml:
  members:
    - infer-b1-general-pi4
    - infer-b1-pedestrian-bt3
consumers/c3_surveillance/initial_ensemble.yaml:
  members:
    - infer-b1-general-pi4
    - infer-b1-pedestrian-bt3
consumers/c1_navigation/contract.yaml:
  request_rate_per_s: 7.0
services/control_plane.yaml:
  member_timeout_ms: 2000
```

- Replace a value: edit it, e.g. `replicas: 2` to `replicas: 3`.
- Add a service: add an entry under `services:` with `model`, `device_type` and `replicas`.
- Change another file: add its path as a top-level key, with the keys to replace.
- A named key is replaced whole: `services:` lists every service of the pool.
- An unknown key is refused.

**Step 3. Use it.**

| Where | How |
|---|---|
| a run on Docker | `OVERRIDES=my_overrides.yaml` for `make up` (8) |
| a run on Kubernetes | `OVERRIDES=my_overrides.yaml` for [`up.sh`](../deploy/kube/up.sh) (11.1) |
| an experiment plan on Kubernetes | `OVERRIDES=my_overrides.yaml` for [`experiment.sh`](../deploy/kube/experiment.sh) (11.5) |
| a whole configuration tree | `poetry run eemls-config render config --overrides my_overrides.yaml --out runs/my_tree` |
| inside an experiment plan | its `overrides:` key, applied to every approach |

| Parameter of `eemls-config render` | Meaning |
|---|---|
| `config` | the tree to start from |
| `--overrides <file>` | the keys to replace |
| `--out <folder>` | where the tree is written; it must be empty |

The deployment checks the tree first. It refuses a pool its devices cannot hold, or that cannot
carry the request rates.

### 7.4 Scenarios

A scenario sets how the queries change over time: rate, classes, condition, quality. It is a file
in [`experiments/scenarios/`](../experiments/scenarios/), named without `.yaml`.

- A simulated day is 6 periods. With 300 s periods, it lasts 30 min.
- A scenario lasts `day_length_s` times its days; [`tools/make_scenarios.py`](../tools/make_scenarios.py) sets the days.
- A run longer than its scenario stays in the last period.

| Scenario | Use |
|---|---|
| `smoke` | a short check |
| `quick_start` | the quick start: one 120 s day, Open Images day classes only |
| `recurring_ten_days` | the default of a run: 10 simulated days |
| `recurring` and its variants | contexts that come back |
| `unpredictable` and its seeds | contexts that change at random|
| `only_<change>_seed_<n>` | one kind of change alone |

| Where | How |
|---|---|
| one run | `SCENARIO=<name> DURATION_S=<seconds>` for `make run` (8) or [`run.sh`](../deploy/kube/run.sh) (11.3) |
| an experiment plan | its `scenario:` key |

Example: `SCENARIO=only_class_burst_seed_0 DURATION_S=1800` runs the class-burst scenario for one
simulated day (of its 4).

Scenario files come from seeds, so every run sees the same queries. Check them, or
write them again after a change. The tool writes both [`experiments/scenarios/`](../experiments/scenarios/) and the archived
options in [`experiments/archived/scenarios/`](../experiments/archived/scenarios/).

```bash
poetry run python3 tools/make_scenarios.py --check
poetry run python3 tools/make_scenarios.py
```

| Parameter | Meaning |
|---|---|
| `--check` | fail when a scenario file differs from what the tool writes; write nothing |

Example: after a change to [`tools/make_scenarios.py`](../tools/make_scenarios.py), `--check` fails; the second command
rewrites the files.

### 7.5 Experiment plans

**Purpose.** An experiment plan is one YAML file in [`experiments/runs/`](../experiments/runs/) that says how to run one
scenario: with which approaches, how many seeds, and for how long. It is named after its scenario.

| Field | Meaning |
|---|---|
| `scenario` | the workload the consumers send: how many requests, which classes, and when things change |
| `approaches` | the methods to compare |
| `seeds` | the seeds each method runs with; one run per seed |
| `duration_s` | how long each run lasts, in seconds |
| `description` | one line about the plan |
| `overrides` (optional) | the overrides of the run (7.3) |

An approach has two parts:

- `approach`: the method that picks the ensemble (the set of models that answer together).
- `contract` (optional): the service goals the method must meet. Without it, the method uses the normal contract.

So one method can appear more than once with different contracts. Below, `cap` is CAP with the normal contract, and
`cap_star` is CAP with a contract that checks only general accuracy. Comparing them shows how much the contract changes
CAP's choices.

```yaml
description: CAP against CAP with only the general accuracy.
scenario: recurring
approaches:
  cap:
    approach: cap
  cap_star:
    approach: cap
    contract: only_general_accuracy   # consumers/<consumer>/contract_only_general_accuracy.yaml
seeds: [0, 1, 2, 3]
duration_s: 7200
```

List the runs of an experiment plan:

```bash
poetry run eemls-exp plan experiments/runs/recurring_ten_days.yaml \
  --config config --scenarios experiments/scenarios --approaches cap,cap_star --seeds 0
```

| Parameter | Meaning |
|---|---|
| [`experiments/runs/recurring_ten_days.yaml`](../experiments/runs/recurring_ten_days.yaml) | the experiment plan |
| `--config` | the configuration tree the runs start from |
| `--scenarios` | the folder of scenario files |
| `--approaches`, `--seeds` | the approaches and seeds to plan, comma separated; all when omitted |

Each line: approach, seed, run id, results folder, length, scenario.

Archived options: one experiment plan per scenario in [`experiments/archived/runs/`](../experiments/archived/runs/), e.g.
[`shuffled_days_seed_3.yaml`](../experiments/archived/runs/shuffled_days_seed_3.yaml) runs [`experiments/archived/scenarios/shuffled_days_seed_3.yaml`](../experiments/archived/scenarios/shuffled_days_seed_3.yaml).
Run them like any experiment plan (11.5).

### 7.6 Configuration of one approach

**Purpose.** Make a ready-to-run copy of the configuration for one approach of an experiment plan. Use it to see exactly
what that approach will run with, or to start that approach by hand.

The command writes two things:

| Output | What it is |
|---|---|
| [`runs/recurring-cap/config/`](../runs/recurring-cap/config/) | a full copy of the `--config` tree, with the approach's policy and contract filled in ([`services/control_plane.yaml`](../config/services/control_plane.yaml)), plus the plan's `overrides` |
| [`runs/recurring-cap/manifest.json`](../runs/recurring-cap/manifest.json) | a record of the run: scenario, approach, policy, contract, seeds, length, commit, and a checksum of every file in the copy |

First write the cluster tree, then the approach's copy of it:

```bash
poetry run eemls-config render config --overrides deploy/kube/overrides.yaml --out runs/kube-base
poetry run eemls-exp render experiments/runs/recurring_ten_days.yaml --approach cap \
  --config runs/kube-base --scenarios experiments/scenarios --out runs/recurring-cap
```

| Parameter | Meaning |
|---|---|
| [`experiments/runs/recurring_ten_days.yaml`](../experiments/runs/recurring_ten_days.yaml) | the experiment plan |
| `--approach` | which approach of the plan to write (here `cap`) |
| `--config` | the configuration to copy from; here the cluster tree of the first line |
| `--scenarios` | the folder of scenario files |
| `--out` | where to write the copy and the manifest; the folder must be empty |

To start that approach, point [`up.sh`](../deploy/kube/up.sh) at the copy: `CONFIG_DIR=runs/recurring-cap/config` (11.1).

### 7.7 Common combinations

| Purpose | Settings |
|---|---|
| a quick check of a change | `SCENARIO=smoke DURATION_S=360`: one short simulated day |
| a run of `n` simulated days | `DURATION_S` = `n` x 1800, with 300 s periods |
| one approach only | an experiment plan with `APPROACHES=<approach>` |
| CAP against one baseline | `APPROACHES=cap,scale_reduction SEEDS=0` |
| the variance over seeds | `APPROACHES=cap SEEDS=0,1,2,3` |
| about 1000 queries per approach | `DURATION_S` = 1000 / (sum of the request rates) |
| devices slower than the profiles say | measured profiles (11.2) in [`models/profiles_cluster.jsonl`](../models/profiles_cluster.jsonl), given as `SERVICE_PROFILES` |
| slow devices that cannot carry the load | lower `request_rate_per_s`, or more `replicas`, in an overrides file |

## 8. Local test with Docker

**Purpose.** Test on this machine without Kubernetes: a one-node Docker Swarm with the pool in
[`deploy/one_host/overrides.yaml`](../deploy/one_host/overrides.yaml).

- `make up` builds the x86 images (as 6.1), starts every service from them as a container, loads
  the profiles and warms the inference services.
- `make run` runs the consumers. The results stay in the database (13).
- `make down` stops the consumers, the control plane, the inference services, the Observation Service, the data hub, the broker, then the database; the rows stay in the database volume.

Quick check, one short simulated day (360 s):

```bash
make up
SCENARIO=smoke DURATION_S=360 make run
```

Full run: plain `make run` lasts 18,000 s (5 h) by default.

```bash
make run
```

| Parameter | Meaning |
|---|---|
| `RUN_ID` | for `make up`: the name the rows are stored under: lowercase letters, digits and hyphens, no hyphen at either end; default `one-host-<time>`; refused if its database volume exists |
| `OVERRIDES` | for `make up`: an overrides file of keys to replace; default [`deploy/one_host/overrides.yaml`](../deploy/one_host/overrides.yaml) |
| `CONFIG_DIR` | for `make up`: a whole configuration tree to use instead of `OVERRIDES` |
| `SERVICE_PROFILES` | for `make up`: the profiles the copies are sized from; default [`models/profiles.jsonl`](../models/profiles.jsonl); `make run` uses the same |
| `WARM_IMAGES` | for `make up`: images sent to each service before the run; default 50 |
| `READY_SECONDS` | for `make up`: how long to wait for every service; default 300 |
| `SCENARIO` | for `make run`: a file name in [`experiments/scenarios/`](../experiments/scenarios/), without `.yaml`; default `recurring_ten_days` |
| `SCENARIOS` | for `make run`: the folder of scenario files mounted into the consumers; default [`experiments/scenarios/`](../experiments/scenarios/) |
| `DURATION_S` | for `make run`: the run length in seconds; default 18000, 10 simulated days |
| `SEED` | for `make up`: the seed the control plane, the warm-up and the consumers draw from; default 0; saved in [`runs/one_host/seed`](../runs/one_host/seed); `make run` uses it |

Load test: one consumer at rising rates:

```bash
RATES=10,20,50 STEP_S=30 make load
```

The report is written to [`runs/one_host/load.json`](../runs/one_host/load.json). The load tool reads only the services, consumers and
datasets of the run's configuration.

| Parameter | Meaning |
|---|---|
| `RATES` | the request rates per second, comma separated; default `10,20,50,100` |
| `STEP_S` | the seconds spent at each rate; default 30 |
| `CONSUMER` | the consumer driven; default `c3_surveillance` |

Stop when done:

```bash
make down
```

To delete the run's rows too, remove its database volume (run id from [`runs/one_host/run_id`](../runs/one_host/run_id), read before `make down`):

```bash
docker volume rm eemls_metrics-database-<run id>
```

## 9. Local test with minikube

**Purpose.** Test the Kubernetes files on one machine before a cluster.

- minikube and k3s (Part 10) share [`runs/kube/current`](../runs/kube/current). Run only one at a time.

### 9.1 Publish the images

Push every image to the `registry`, tagged with the `version`, of
[`config/local/images.yaml`](../config/local/images.yaml). Every Kubernetes node pulls from there, minikube too.
Set `registry` to a registry this machine can push to; `docker.io/example` is only an example.

Log in to the registry once:

```bash
docker login <registry>
```

Build both architectures (6.2 first) and push:

```bash
make images
```

Push the Jetson image brought from the board (6.4):

```bash
docker push <registry>/eemls-inference-service:jetson-<version>
```

One image only:

```bash
sh deploy/images.sh control-plane
```

After a change under [`application/`](../application/), a release and pin of [`src/eemls/`](../src/eemls/) (Part 6), or a change to a file's form under [`config/`](../config/), publish again before a
Kubernetes run. Otherwise the services refuse the changed configuration.

Example: with `registry: docker.io/example` and `version: latest`, the images pushed include
`docker.io/example/eemls-control-plane:latest` and `docker.io/example/eemls-inference-service:cpu-arm64-latest`.

### 9.2 Create the cluster

Create a one-node cluster with CPU metrics.

```bash
minikube start -p eemls --driver=docker --cpus=4 --memory=6g
minikube addons enable metrics-server -p eemls
```

| Parameter | Meaning |
|---|---|
| `-p eemls` | the cluster name; the node gets the same name |
| `--driver=docker` | the node runs as a Docker container on this machine |
| `--cpus`, `--memory` | the CPUs and memory the node may use |

Example: check that the node is ready.

```bash
kubectl get nodes   # eemls   Ready   control-plane
```

### 9.3 Label the node

**Purpose.** A service runs only on a node labelled with its device type. Here one node takes
every role.

```bash
kubectl label node eemls eemls.role.core=true eemls.device.beelink_bt3=true eemls.device.raspberry_pi4=true
```

| Label | Meaning |
|---|---|
| `eemls.role.core=true` | the node runs the database, broker, data hub, Observation Service, control plane and consumers |
| `eemls.device.<device type>=true` | the node runs the inference services of that device type |

Example: list the labels of the node.

```bash
kubectl get node eemls --show-labels | tr ',' '\n' | grep eemls
```

### 9.4 Copy the models and the image store

**Purpose.** The models and images are mounted from the node, at `/srv/eemls/models` and
`/srv/eemls/images`.

```bash
tar c -C models . \
  | docker exec -i eemls sh -c 'mkdir -p /srv/eemls/models /srv/eemls/images && tar x -C /srv/eemls/models'
docker exec -i eemls sh -c 'cat > /srv/eemls/models/models.yaml' < config/local/models.yaml
tar c -C data/hub . | docker exec -i eemls sh -c 'tar x -C /srv/eemls/images && chmod -R a+rX /srv/eemls'
```

Example: check that the models are in the node.

```bash
docker exec eemls ls /srv/eemls/models
```

### 9.5 Run

Start with the local pool, run, stop. Same scripts as the cluster (Part 11).

```bash
OVERRIDES=deploy/one_host/overrides.yaml sh deploy/kube/up.sh
SCENARIO=smoke DURATION_S=360 sh deploy/kube/run.sh
DELETE_VOLUME=1 sh deploy/kube/down.sh
```

| Parameter | Meaning |
|---|---|
| `OVERRIDES` | an overrides file of keys to replace; here the local pool's |
| `SCENARIO`, `DURATION_S` | the scenario and its length in seconds |
| `DELETE_VOLUME=1` | also delete this run's database volume |

Example output:

```text
run id: kube-20261019-041818
up: run kube-20261019-041818
<n> queries accepted of c1_navigation in 360.0 s
<n> queries accepted of c2_safety in 360.0 s
<n> queries accepted of c3_surveillance in 360.0 s
run kube-20261019-041818: rows under runs/kube/kube-20261019-041818/rows
down: run kube-20261019-041818
```

## 10. Kubernetes cluster

### 10.1 Install the server

**Purpose.** One device runs the k3s server.

Install it and write its kubeconfig here:

```bash
sh deploy/kube/k3s/server.sh <username>@<server> <server address>
export KUBECONFIG=~/.kube/eemls.yaml
```

| Parameter | Meaning |
|---|---|
| `<username>@<server>` | the SSH user and device that becomes the server |
| `<server address>` | the address the agents and this machine reach the server at |
| third argument | where the kubeconfig is written; default `~/.kube/eemls.yaml` |
| `K3S_VERSION` | the k3s release; default `v1.33.5+k3s1` |

If `kubectl get nodes` gives `connection refused`, reach the server over SSH (10.3).

Example: install the server, then list the nodes.

```bash
sh deploy/kube/k3s/server.sh <username>@<server> <server address>
export KUBECONFIG=~/.kube/eemls.yaml
kubectl get nodes
```

### 10.2 Join the agents

Join each other device. A GPU device needs the NVIDIA Container Toolkit first, for the `nvidia`
runtime.

```bash
sh deploy/kube/k3s/agent.sh <username>@<agent> <username>@<server> <server address>
```

| Parameter | Meaning |
|---|---|
| `<username>@<agent>` | the device to join |
| `<username>@<server>` | the server, read over SSH for the join token |
| `<server address>` | the address the agent reaches the server at |

Example: join one agent, then list the nodes.

```bash
sh deploy/kube/k3s/agent.sh <username>@<agent> <username>@<server> <server address>
kubectl get nodes
```

### 10.3 Reach the server over SSH only

If only SSH reaches the server: forward its port and use a kubeconfig copy. Keep the tunnel
open.

```bash
ssh -fN -L 127.0.0.1:16443:127.0.0.1:6443 <username>@<server>
sed 's#https://<server address>:6443#https://127.0.0.1:16443#' ~/.kube/eemls.yaml > ~/.kube/eemls-tunnel.yaml
export KUBECONFIG=~/.kube/eemls-tunnel.yaml
```

| Parameter | Meaning |
|---|---|
| `-fN` | keep the tunnel open in the background, with no remote command |
| `-L 127.0.0.1:16443:127.0.0.1:6443` | port 16443 on this machine reaches the server's port 6443 |

Example: check the tunnel.

```bash
kubectl get nodes   # lists the nodes through 127.0.0.1:16443
```

### 10.4 Label the nodes

**Purpose.** A service runs only on a node with its device type. One core node also runs the
database, broker, data hub, Observation Service, control plane and consumers.

```bash
sh deploy/kube/label.sh <node> <device type> [core]
```

| Parameter | Meaning |
|---|---|
| `<node>` | the node name, as `kubectl get nodes` prints it |
| `<device type>` | a device type of [`config/cost/prices.yaml`](../config/cost/prices.yaml) |
| `core` | also mark the node as the core node |

| Purpose | Command |
|---|---|
| the core node | `sh deploy/kube/label.sh <node> beelink_bt3 core` |
| a Jetson Orin Nano | `sh deploy/kube/label.sh <node> jetson_orin_nano` |
| a node that offers a second device type | `kubectl label node <node> eemls.device.<device type>=true` |
| a node taken out of the pool | `kubectl label node <node> eemls.device.<device type>-` |

### 10.5 Copy the models and the image store

**Purpose.** Nodes mount models from `/srv/eemls/models`; the core node mounts images from
`/srv/eemls/images`.

Create `/srv/eemls` on each device, owned by the SSH user:

```bash
ssh <username>@<host> 'sudo mkdir -p /srv/eemls && sudo chown <username>:<username> /srv/eemls'
```

Copy each device its services' models, and the core node the image store:

```bash
OVERRIDES=deploy/kube/overrides.yaml sh deploy/edge/models.sh <username>@<host> <device type>
rsync -a data/hub/ <username>@<core node>:/srv/eemls/images/
```

| Parameter of [`models.sh`](../deploy/edge/models.sh) | Meaning |
|---|---|
| `<username>@<host>` | the device to copy to |
| `<device type>` | its device type; the models of its services are copied |
| third argument | the folder on the device; default `/srv/eemls/models` |
| `OVERRIDES` | the overrides of the run (7.3); the models of its pool are copied; default the pool of [`config/`](../config/) |

Give [`models.sh`](../deploy/edge/models.sh) the same `OVERRIDES` as the run (11.1). Copy again after a pool change. A node with two device types needs [`models.sh`](../deploy/edge/models.sh) once per
type.

Example: a Raspberry Pi 4 of the pool in [`deploy/kube/overrides.yaml`](../deploy/kube/overrides.yaml).

```bash
OVERRIDES=deploy/kube/overrides.yaml sh deploy/edge/models.sh <username>@<host> raspberry_pi4
# prints: <username>@<host>: b1_general b3_motorcycle in /srv/eemls/models
```

## 11. Runs

### 11.1 Start the system

**Purpose.** Start all services, load the profiles and warm the inference services. Needs the
published images (9.1).

```bash
sh deploy/kube/up.sh
```

| Parameter | Meaning |
|---|---|
| `OVERRIDES` | an overrides file of keys to replace; default [`deploy/kube/overrides.yaml`](../deploy/kube/overrides.yaml) |
| `CONFIG_DIR` | a whole configuration tree to use instead of `OVERRIDES` |
| `RUN_ID` | the name the rows are stored under: lowercase letters, digits and hyphens, no hyphen at either end; default `kube-<time>`; a name used before becomes `<n>-<name>`, e.g. `1-profiles`; refused while another run is up |
| `SERVICE_PROFILES` | the profiles the copies are sized from and the policies start from; default [`models/profiles.jsonl`](../models/profiles.jsonl); [`run.sh`](../deploy/kube/run.sh) uses the same |
| `IMAGES` | the image store on the core node; default `/srv/eemls/images` |
| `WARM_IMAGES` | images sent to each service before the run; default 50 |
| `SEED` | the seed the control plane, the warm-up and the consumers draw from; default 0; saved in [`runs/kube/<run id>/seed`](../runs/kube/); [`run.sh`](../deploy/kube/run.sh) uses it |
| `READY_SECONDS` | how long to wait for every service; default 600 |

Copies must carry all contracts' request rates together, or the pool is refused. Rates: each
[`consumers/<consumer>/contract.yaml`](../config/consumers/).

| Purpose | Command |
|---|---|
| the cluster pool of [`deploy/kube/overrides.yaml`](../deploy/kube/overrides.yaml) | `sh deploy/kube/up.sh` |
| an own pool or rates (7.3) | `OVERRIDES=my_overrides.yaml sh deploy/kube/up.sh` |
| one approach of an experiment plan | `CONFIG_DIR=runs/recurring-cap/config sh deploy/kube/up.sh` |
| another seed | `SEED=1 sh deploy/kube/up.sh` |

If a service is not ready, read its log:

```bash
kubectl -n eemls get pods
kubectl -n eemls logs deploy/<service> --tail=20
```

### 11.2 Measure the profiles on the devices

Stop the running system first (12.1). Then start, forward each service to a local port, measure, stop:

```bash
OVERRIDES=examples/overrides/cluster_gpu.yaml RUN_ID=profile-cluster sh deploy/kube/up.sh
port=18001; at=""
for svc in $(kubectl -n eemls get svc -o name | grep infer- | cut -d/ -f2); do
  kubectl -n eemls port-forward "svc/$svc" "$port:8082" >/dev/null 2>&1 &
  at="$at --at $svc=http://127.0.0.1:$port"; port=$((port + 1))
done
poetry run python3 tools/profile_pool.py --config "runs/kube/$(cat runs/kube/current)/config" $at \
  --into models/profiles_cluster.jsonl
pkill -f "[p]ort-forward svc/infer-"
DELETE_VOLUME=1 sh deploy/kube/down.sh
```

| Parameter of [`profile_pool.py`](../tools/profile_pool.py) | Meaning |
|---|---|
| `--config` | the run's configuration tree, which names the pool |
| `--at <service>=<url>` | a running service and its address; measured there instead of on this machine |
| `--into` | where the profiles are written |
| `--images <n>` | clear images measured, spread over the classes; default 10000 |

- Start [`up.sh`](../deploy/kube/up.sh) with the same `OVERRIDES` as the later runs.
- Later runs: `SERVICE_PROFILES=models/profiles_cluster.jsonl` for [`up.sh`](../deploy/kube/up.sh) (11.1); file: [`models/profiles_cluster.jsonl`](../models/profiles_cluster.jsonl).
- If the pool then cannot carry the request rates, lower them or add copies (7.7).

Example output:

```text
infer-b1-general-pi4                      73.9 ms  0.600 accurate  7.1696 per 1k
infer-b5-general-orin-nano                27.5 ms  0.721 accurate  11.0709 per 1k
6 profiles written to models/profiles_cluster.jsonl
```

### 11.3 Run a scenario

**Purpose.** Run the consumers, then collect the rows. Start the system first (11.1).

- After the consumers end, [`run.sh`](../deploy/kube/run.sh) stops the control plane, then the inference services, so their last rows reach the Observation Service.
- Then it stops the Observation Service.
- Then it collects the rows with `eemls-collect`.

```bash
sh deploy/kube/run.sh
```

| Parameter | Meaning |
|---|---|
| `SCENARIO` | a file name in [`experiments/scenarios/`](../experiments/scenarios/), without `.yaml`; default `recurring_ten_days` |
| `SCENARIOS` | the folder of scenario files; the consumers mount the scenario from it; default [`experiments/scenarios/`](../experiments/scenarios/) |
| `DURATION_S` | the run length in seconds; default 18000, 10 simulated days |

The consumers use the seed of [`up.sh`](../deploy/kube/up.sh) (11.1).

Example: the recurring scenario, its 4 simulated days.

```bash
SCENARIO=recurring DURATION_S=7200 sh deploy/kube/run.sh
```

### 11.4 Run a load test

**Purpose.** Find the rate the cluster carries: one consumer at rising rates. Start the system
first (11.1).

```bash
RATES=10,20,50,100 sh deploy/kube/load_test.sh
```

After a scenario run (11.3), stop and start again first: `DELETE_VOLUME=1 sh deploy/kube/down.sh`, then `sh deploy/kube/up.sh`.

The report is written to [`runs/kube/<run id>/load.txt`](../runs/kube/).

| Parameter | Meaning |
|---|---|
| `RATES` | the request rates per second, comma separated; default `10,20,50,100` |
| `STEP_S` | the seconds spent at each rate; default 30 |
| `CONSUMER` | the consumer driven; default `c3_surveillance` |

Burst test: a burst on one service; its CPU and copies print every 10 s:

```bash
sh deploy/kube/burst.sh <service> <image id> 60 8 150
```

| Argument | Meaning |
|---|---|
| `<service>` | the inference service driven |
| `<image id>` | an image of the image store, sent on every request |
| `60` | the seconds of the burst |
| `8` | the requests sent at once |
| `150` | the seconds watched after the burst |

Example: three 20-second steps.

```bash
RATES=5,10,20 STEP_S=20 sh deploy/kube/load_test.sh
```

### 11.5 Run multiple approaches

```bash
PLAN=experiments/runs/recurring_ten_days.yaml APPROACHES=cap,cap_star SEEDS=0 \
  sh deploy/kube/experiment.sh
```

| Parameter | Meaning |
|---|---|
| `PLAN` | the experiment plan file in [`experiments/runs/`](../experiments/runs/) or [`experiments/archived/runs/`](../experiments/archived/runs/) |
| `SCENARIOS` | the folder of scenario files; default the `scenarios/` folder beside the plan's `runs/` folder |
| `APPROACHES`, `SEEDS` | the approaches and seeds to run, comma separated; all when unset |
| `DURATION_S` | the length of each run in seconds; the plan's length when unset |
| `OUT` | the results folder; default [`runs/kube_<time>_<plan>`](../runs/) |
| `OVERRIDES` | the overrides every run starts from; default [`deploy/kube/overrides.yaml`](../deploy/kube/overrides.yaml) |
| `SERVICE_PROFILES` | as for [`up.sh`](../deploy/kube/up.sh) |

| Purpose | Command |
|---|---|
| every approach, short, on measured profiles | `SERVICE_PROFILES=models/profiles_cluster.jsonl OVERRIDES=examples/overrides/cluster_gpu.yaml PLAN=experiments/runs/recurring_ten_days.yaml SEEDS=0 DURATION_S=48 sh deploy/kube/experiment.sh` |
| CAP and its two ablations | `APPROACHES=cap,cap_star,cap_minus` with the same settings |
| all of `recurring_ten_days`: every approach and seed | `PLAN=experiments/runs/recurring_ten_days.yaml sh deploy/kube/experiment.sh` |

A rerun reuses the run ids; each run's working folder is cleared first.

## 12. Shutdown

### 12.1 Stop the system

Stop on Kubernetes. The consumers stop first, then the control plane, the inference services, the Observation Service, the data hub, the broker, then the database.
[`down.sh`](../deploy/kube/down.sh) keeps the `eemls-database` secret; the next [`up.sh`](../deploy/kube/up.sh) replaces it.
The rows stay in the database volume; `DELETE_VOLUME=1` deletes this run's database volume. Rows collected
by [`run.sh`](../deploy/kube/run.sh) stay in [`runs/`](../runs/).

Keep the rows:

```bash
sh deploy/kube/down.sh
```

Or delete them too:

```bash
DELETE_VOLUME=1 sh deploy/kube/down.sh
```

A run killed by hand (kill -9, a closed terminal) can leave its database forward open and block the next start. End it:

```bash
pkill -f "[p]ort-forward svc/postgres"
```

Stop the system on Docker:

```bash
make down
```

Example: clean up after Ctrl-C.

```bash
pkill -f "[p]ort-forward svc/postgres"
DELETE_VOLUME=1 sh deploy/kube/down.sh
```

### 12.2 Delete the cluster

Delete the minikube cluster, or uninstall k3s from each device.

```bash
minikube delete -p eemls
ssh <username>@<server> sudo /usr/local/bin/k3s-uninstall.sh
ssh <username>@<agent> sudo /usr/local/bin/k3s-agent-uninstall.sh
```

Example: remove one agent, keep the cluster.

```bash
kubectl delete node <node>
ssh <username>@<agent> sudo /usr/local/bin/k3s-agent-uninstall.sh
```

## 13. Results

The results of each run are stored in its folder:

| Run | Folder |
|---|---|
| an experiment plan on Kubernetes | [`runs/kube_<time>_<plan>/<scenario>/<approach>/seed-<n>/`](../runs/) |
| an experiment plan on Docker (`make experiment`) | [`runs/local_<time>_<plan>/<scenario>/<approach>/seed-<n>/`](../runs/) |
| one run on Kubernetes | [`runs/kube/<run id>/`](../runs/kube/) |
| one run on Docker (`make run`) | the database `eemls`, in the volume `eemls_metrics-database-<run id>` |

Tables of the database; every row carries its `run_id`:

| Table | Holds |
|---|---|
| `run_owners` | the control plane that owns each run |
| `instances` | each copy that served: its service, node and image |
| `queries` | each query: its consumer, ensemble, answer and response time; `time_of_day`: the point of the simulated day, from 0 to 1 |
| `member_predictions` | each member's answer to each query; `in_ensemble` false: an idle service's probe answer |
| `feedback` | the true labels the consumers sent back |
| `decisions` | each decision: the ensemble chosen and its CDM values; `scored`: how many candidate ensembles the policy scored; empty when it does not count them |
| `inference_service_profiles` | the profile of each inference service |

## 14. Known errors

Grouped by the part where each error shows. Each item: what you see, then the fix.

### 14.1 Setup (Part 2)

- `make install` over SSH stops when the link drops
  - Run it under `nohup`.
- `no config/local/models.yaml: run tools/new_local.py and fill it in`
  - Create the local files (2.2).

### 14.2 Dataset (Part 3)

- `prepare_dataset.py --check` prints `missing: <condition> <class>`
  - That class has no image yet. Download it (3.1) or build the multispectral crops (3.2).

### 14.3 Local test with Docker (Parts 5 and 8)

- `refused: the data hub holds no clear day image of <class>`
  - The image store lacks a class. Check with `prepare_dataset.py --check` (3.3).

- `no image <image> on this host; build it with make up`
  - The local images are missing. Build them: `make images-local` (6.1); `make up` builds them too.
- `run id <run id> has a database volume already; pick another RUN_ID or remove the volume`
  - An old database volume has the same run id; `up.sh` refuses it. Pick another `RUN_ID`, or
    remove it: `docker volume rm eemls_metrics-database-<run id>`.

### 14.4 Container images (Part 6)

- The ONNX Runtime build stops with `context canceled`
  - The SSH link dropped. Run it detached:
    `ssh <username>@<board> 'cd eemls && (nohup setsid docker buildx bake onnxruntime-jetson > ../ort-build.log 2>&1 < /dev/null &)'`.
- An Orin Nano stops answering SSH during the ONNX Runtime build
  - It is out of memory. Build on a larger Jetson Orin; see
    [`application/inference-service/README.md`](../application/inference-service/README.md#jetson-tag).
- `docker push` fails with `502 Bad Gateway` on a large layer
  - Push again; uploaded layers are kept.

### 14.5 Run configuration (Part 7)

- `refused: <service> carries ... below ... requests/s`
  - Lower `request_rate_per_s`, add `replicas`, or measure the profiles on the devices (11.2).

### 14.6 minikube (Part 9)

- mosquitto fails to start inside minikube; broker log: `Error: Unable to open config file /mosquitto-no-auth.conf.`
  - Seen on a laptop with a recent kernel. Use another machine, or the Docker test (Part 8).

### 14.7 Kubernetes cluster (Part 10)

- A node fails to pull with `lookup ... Try again`
  - Pull again: `sudo k3s crictl pull <image>`.
- `k3s crictl rmi --prune` removed images still needed
  - It removes every image no container uses. Remove one tag: `sudo k3s crictl rmi <image>:<tag>`.

### 14.8 Runs (Part 11)

- Every service in `CrashLoopBackOff`; log: `refused: ... Input should be ...`
  - The images are older than the code that reads the config. Publish them again (9.1).
- `kubectl wait` times out and lists every deployment
  - One timeout lists them all. Read the first failing pod's log:
    `kubectl -n eemls logs deploy/<service> --tail=20`.
- Jetson pods `OOMKilled`
  - Raise `runtime_memory_mb.jetson` in [`services/deployment.yaml`](../config/services/deployment.yaml).
- Jetson log: `Failed to load libcurand.so.10`, `GPU device discovery failed`
  - Harmless.
- The control plane restarts a few times at start (2–3 seen)
  - Harmless: it waits for the database and for the profiles to be loaded.
- `the database forward stopped: ... address already in use`
  - A forward from an old run is left. `pkill -f "[p]ort-forward svc/postgres"`, then start again.
- Responses of about 1 s; Raspberry Pi members are slow
  - The profiles were measured on this machine. Measure them on the devices (11.2) and lower
    the request rates.
- `kubectl run -i --rm` loses the first lines of a short pod
  - Wait for the pod to end, then read `kubectl logs <pod>`.
- `grep b5` on pod names also matches other pods
  - Match the full service name.

### 14.9 Shutdown and the shell (Part 12)

- `pkill -f <pattern>` ends its own shell
  - Use a bracket pattern: `pkill -f "[p]ort-forward"`.
- A forward stays after a run stopped by hand
  - `pkill -f "[p]ort-forward svc/postgres"`.
