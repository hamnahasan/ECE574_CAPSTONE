# Literature Baselines

These wrappers retrain published architectures on the **same Sen1Floods11
splits** that our TriModal model uses, so the numbers are directly
comparable in the paper.

## What's here

| Script | Architecture(s) | Purpose | Status |
|---|---|---|---|
| `train_smp_baselines.py` | U-Net, U-Net++, DeepLabV3+, FPN, MAnet (via segmentation_models_pytorch) | Generic strong baselines; vanilla U-Net is the architectural control | working |
| `train_prithvi_lora.py` | Prithvi-EO-1.0-100M with LoRA adapters | Foundation-model fair-comparison: their pretraining + our 446 chips | skeleton; HF model loading may need adaptation |

## What's *not* here yet

**BASNet (Bai 2021).** No canonical pip-installable BASNet implementation
exists. To add: clone the BASNet reference repo into `third_party/`, then
write `train_basnet.py` mirroring the structure of `train_fusion.py`.

**GLNet / SSFNet (2025).** Same situation — no pip distribution. Would
need to vendor the official code if/when published.

## Dependencies

```bash
pip install segmentation-models-pytorch    # for SMP baselines
pip install transformers peft              # for Prithvi-EO LoRA
```

Both should go into the `floodseg` conda env on Isaac. Do not install
into the system Python.

## Submission examples

### SMP baselines × 3 seeds (uses the multi-seed harness)

```bash
RUN_NAME=smp_unet_resnet34_s1_s2_dem \
TRAIN_SCRIPT=scripts/literature_baselines/train_smp_baselines.py \
EXTRA_ARGS="--arch unet --modalities s1_s2_dem --encoder resnet34" \
  sbatch slurm/multiseed_array.sbatch

RUN_NAME=smp_unetplusplus_resnet34_s1_s2_dem \
TRAIN_SCRIPT=scripts/literature_baselines/train_smp_baselines.py \
EXTRA_ARGS="--arch unetplusplus --modalities s1_s2_dem --encoder resnet34" \
  sbatch slurm/multiseed_array.sbatch

RUN_NAME=smp_deeplabv3plus_resnet34_s1_s2_dem \
TRAIN_SCRIPT=scripts/literature_baselines/train_smp_baselines.py \
EXTRA_ARGS="--arch deeplabv3plus --modalities s1_s2_dem --encoder resnet34" \
  sbatch slurm/multiseed_array.sbatch
```

### Prithvi-EO LoRA fine-tune

```bash
sbatch slurm/train_prithvi_lora.sbatch       # see ../slurm/
```
