# Private dataset layout

The assignment dataset is not versioned. Obtain the authorized course copy and place its contents here without an extra wrapper directory:

```text
dataset/
|-- train/
|   |-- styles_train.csv
|   `-- images_train/
|       `-- <product-id>.jpg
`-- test/
    |-- styles_prediction.csv
    `-- images_test/
        `-- <product-id>.jpg
```

Each image filename must match the `id` in its CSV. Do not publish or commit the supplied images or metadata. The shared audit, frozen split, and normalization values are versioned separately under `scripts/data/` so every authorized team member trains against the same contract. Because those manifests contain course item IDs and image hashes, keep the collaboration repository private and accessible only to the authorized assignment team.

Verify the authorized CSV release before using the frozen manifests:

| File | SHA-256 |
|---|---|
| `train/styles_train.csv` | `54D503743F79F22BF03F9E2C216B53E75D5F2129EDA5934CFA25C642C70AF44D` |
| `test/styles_prediction.csv` | `7F83F219D0FB7F2EC86B8B5C0BA28C2CFADD8F8E6E85A36C9919C9AB5C41315E` |
