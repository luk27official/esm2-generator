# ESM2 generator
You can use this repository to generate ESM2 embeddings using `gpulab` university cluster.

## Choose model
You can choose the desired model by uncommenting the appropriate line in the `compute-esm.py` file:
```
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
# model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
```

## Run sbatch
You can either run the `compute-esm.py` directly, or you can use the university cluster by running
```
sbatch --gpus=1 run-sbatch.sh
```
