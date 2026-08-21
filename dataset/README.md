# Dataset placement

Copy the **contents** of the supplied `FashionDataset` directory here. Do not add another `FashionDataset` wrapper directory.

The required layout is:

```text
dataset/
|-- train/
|   |-- styles_train.csv
|   `-- images_train/
|       |-- <product-id>.jpg
|       `-- ...
`-- test/
    |-- styles_prediction.csv
    `-- images_test/
        |-- <product-id>.jpg
        `-- ...
```

For example, a row with `id` equal to `15970` in `styles_train.csv` must have its image at `train/images_train/15970.jpg`.

The dataset is private and ignored by Git. This README is the only tracked file in this directory.
