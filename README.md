# LFR_project

## UV venv - Initial Setup

Here we use [UV](https://docs.astral.sh/uv/pip/environments/) for handeling the venv, but you can also use your alternative setup of choise.

1. Install a uv-managed Python if 3.12.9 is not already available: ```uv python install 3.12.9```

2. Create the venv: ```uv venv --python 3.12.9 lfr-clip.venv```

3. Activate the venv: ```source lfr-clip.venv/bin/activate```

4. Install requirements: ```uv pip install -r requirements.txt```

After initial setup is done, you only need to make sure that you have your venv activated when running the project.


## Dataset

We assume that we have the following resources obtrained from dataset.zip organized as follows:

```text
LFR_project/
├── former_project/
├── RSICD_images/
├── txtclasses_rsicd/
└── dataset_rsicd.json
```
