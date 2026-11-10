CosmoTokens: Capacity-Controlled CNN and Transformer Comparison for CAMELS
CosmoTokens is an exploratory machine-learning study of cosmological and astrophysical parameter regression from CAMELS IllustrisTNG stellar-mass maps. The code compares a compact residual CNN, a wider CNN used as a capacity control, and a CNN encoder with a GOTHAM-inspired Transformer decoder. All three models use the same simulation-level data split, training protocol, input representation, and structured-masking evaluation.
Important: The Transformer is inspired by the encoder-decoder idea used in GOTHAM; this project is not a reproduction of GOTHAM. GOTHAM uses three-dimensional particle-mesh fields and autoregressive halo-catalogue generation. CosmoTokens uses two-dimensional CAMELS stellar-mass maps and predicts six continuous parameters simultaneously.

Research question
Does a CNN-Transformer decoder improve six-parameter regression from CAMELS maps when compared with both a compact CNN and a CNN with approximately the same number of trainable parameters as the Transformer? How does performance change when spatial regions of the input maps are masked?

The study deliberately separates improvements due to model capacity from improvements that might be associated with attention. It also reports parameter-wise performance and prediction shrinkage rather than relying only on one aggregate score.
Models
Model	Description	Trainable parameters
Compact CNN	Residual CNN encoder with global average pooling and an MLP regression head	610,630
Wider CNN	Wider version of the CNN baseline; capacity-control model	953,046
CNN + Transformer decoder	Residual CNN encoder, 64 spatial tokens, six learned parameter queries, and a four-layer Transformer decoder	974,854


The wider CNN and Transformer differ in parameter count by approximately 2.2%. This is an approximate capacity match, not an exactly parameter-identical comparison.

The Transformer has six learned queries, one for each output parameter. Its decoder uses four layers, four attention heads per layer, and a feed-forward dimension of 256. The CNN encoder converts each two-channel 64 × 64 input into 64 spatial tokens with 64 features per token.
Data and inputs
The notebook expects preprocessed data that already exist in Google Drive. It does not download the raw CAMELS release or recreate the preprocessing pipeline from the raw simulation products.

The experiment uses:
- 15,000 IllustrisTNG Latin-hypercube CAMELS stellar-mass maps;
- 64 × 64 preprocessed maps, derived from 256 × 256 maps by non-overlapping 4 × 4 sum-pooling and log scaling;
- 15 maps for each of 1,000 parent simulations;
- six target parameters: Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, and A_AGN2.
Required Google Drive layout
The notebook sets its project root to MyDrive/CosmoTokens. It expects the following files and folders:
MyDrive/
└── CosmoTokens/
    ├── artifacts/
    │   └── step1/
    │       └── step1_manifest.json
    ├── data/
    │   ├── processed/
    │   │   └── maps_Mstar_sum64_log1p.npy
    │   └── splits/
    │       ├── train_simulation_ids.npy
    │       ├── validation_simulation_ids.npy
    │       ├── test_simulation_ids.npy
    │       ├── train_map_indices.npy
    │       ├── validation_map_indices.npy
    │       └── test_map_indices.npy
    └── experiments/
        └── capacity_matched_v1/   # created by the notebook
The JSON manifest must contain valid paths under the keys labels_per_map_path and parent_simulation_id_path. Those paths must resolve to the label array and parent-simulation-ID array used by the notebook. The expected array dimensions are:
- Maps: (15000, 64, 64)
- Labels: (15000, 6)
- Parent simulation IDs: 15,000 entries
- Train / validation / test simulations: 800 / 100 / 100
- Train / validation / test maps: 12,000 / 1,500 / 1,500
The notebook checks these dimensions and verifies that the simulation-ID splits do not overlap. It also checks that each parent simulation has 15 maps and that the six labels are constant across the maps belonging to the same simulation.

Target means and standard deviations are calculated using one label row per training simulation only. The same training-derived scaling is applied to validation and test targets.
Structured missingness experiment
The model receives two channels:
1. the stellar-mass map, with masked pixels set to zero;
2. a binary validity channel, where 1 means observed and 0 means missing.
The notebook evaluates three synthetic mask geometries:
- Square occlusion
- Vertical stripe
- Block dropout, using 8 × 8 blocks on the 64 × 64 map
The nominal missing fractions are 0%, 20%, 40%, and 60%. The realised fraction can differ slightly from the nominal value for geometric masks.

Training samples four randomly selected map views per training simulation per epoch. A missing fraction is selected from the four configured levels; for non-zero fractions, a mask geometry is also sampled.

Validation uses five conditions: clean input, square masks at 40% and 60%, vertical stripe at 40%, and block dropout at 40%.

The test-condition list contains a clean condition plus each geometry at all four configured fractions. The three geometry-specific 0% conditions are equivalent to clean input, so the list contains 13 condition labels but 10 distinct input conditions. Do not count the duplicated 0% conditions as separate independent masking scenarios when calculating a single aggregate across unique conditions.
Training configuration
The final capacity-matched run uses the following settings:
Setting	Value
Random seeds	101, 202, 303
Batch size	64
Maximum epochs	60
Early-stopping patience	10 epochs
Minimum validation-score improvement	1 × 10⁻⁴
Optimizer	AdamW
Initial learning rate	3 × 10⁻⁴
Weight decay	1 × 10⁻⁴
Learning-rate scheduler	ReduceLROnPlateau, factor 0.5, patience 3
Gradient clipping	Maximum norm 1.0
Training loss	Mean squared error (MSE) on standardized targets
Checkpoint selection	Lowest robust-validation normalized MAE (nMAE)
Mixed precision	Enabled when a CUDA GPU is available


The validation score is averaged over the five predefined validation conditions. The three architectures are trained separately for each of the three seeds, giving nine model runs in the final capacity-matched experiment.

Loss/metric distinction: Training optimizes MSE, while checkpoint selection uses nMAE. The notebook also calculates normalized RMSE (nRMSE) as a complementary metric. These objectives are not interchangeable; the current experiment does not include a controlled MSE-versus-MAE retraining ablation.
Recorded validation results
The notebook's saved execution output contains the following best robust-validation nMAE values across the three seeds (lower is better):
Model	Mean validation nMAE	SD across seeds
Compact CNN	0.627705	0.002649
Wider CNN	0.613458	0.000841
CNN + Transformer decoder	0.612283	0.006026


These are validation results from the stored notebook run, not a new execution performed for this README. They are not test-set performance estimates. Test metrics are saved by condition and parameter in the output CSV files.
Metrics and uncertainty
The evaluation is performed at the parent-simulation level. Predictions from the 15 maps belonging to each simulation are aggregated before calculating the simulation-level metrics. For the ensemble, predictions from the three seed-specific models are averaged per simulation before metrics are calculated.

The notebook reports:
- normalized MAE and normalized RMSE;
- physical-unit MAE and RMSE after reversing target standardization;
- physical prediction bias;
- standardized R²;
- prediction-to-truth standard-deviation ratio;
- paired simulation-level bootstrap intervals for model comparisons.
The final expanded code cell additionally saves per-simulation ensemble predictions and paired bootstrap summaries for the six-parameter aggregate score. Negative delta_B_minus_A values mean model B has lower error than model A under the corresponding comparison definition.
Running the notebook
The notebook is designed for Google Colab and Google Drive. A CUDA GPU is recommended; the code automatically falls back to CPU if CUDA is unavailable, but training will generally be slower.
1. Open the notebook in Google Colab.
2. Make sure the required preprocessed arrays, manifest, and simulation split files are available under MyDrive/CosmoTokens in the layout above.
3. Run the final expanded capacity-matched experiment block (the last major training cell in the notebook). Authorize Google Drive access when prompted.
4. Wait for the three models to train across the three seeds and for evaluation and bootstrap analysis to finish.
5. Retrieve the result files from MyDrive/CosmoTokens/experiments/capacity_matched_v1 or use the automatic Colab downloads at the end of the run.
Note about the notebook's earlier cells
The notebook retains earlier experiment versions for development history. It includes an earlier two-model CNN-versus-Transformer run and an earlier capacity-matched run, followed by the expanded final capacity-matched run. Do not execute every full training block sequentially unless you intentionally want to rerun those earlier experiments. The last major training cell is the expanded run that also saves per-simulation ensemble predictions and the aggregate paired-bootstrap file.
Software
The notebook imports:
- Python standard-library modules (os, gc, json, time, random, math, warnings, pathlib, contextlib, zipfile);
- NumPy;
- pandas;
- Matplotlib;
- PyTorch;
- Google Colab Drive and file-download utilities;
- openpyxl for writing the Excel workbook.
A separate pinned requirements.txt is not included in the notebook. The notebook was run in Google Colab with a Tesla T4 GPU. The recorded run reported approximately 0.35 hours of elapsed time; actual runtime depends on the allocated accelerator, runtime state, and data I/O. For a local environment, the Google Colab Drive mounting and download cells need to be adapted.
Outputs
The final capacity-matched experiment writes results to:
MyDrive/CosmoTokens/experiments/capacity_matched_v1/
The main outputs are:
File	Contents
checkpoints/	Best checkpoint for each architecture and seed
*_seed*_history.csv	Per-epoch training loss, validation score, and learning rate
training_summary.csv	Best epoch and robust-validation nMAE for each run
test_metrics_per_seed.csv	Per-seed, per-condition, per-parameter test metrics
test_metrics_seed_ensemble.csv	Metrics after averaging predictions across the three seeds
test_simulation_predictions.csv	Per-simulation true and ensemble-predicted standardized targets
paired_bootstrap_CI.csv	Paired bootstrap intervals for per-parameter model differences by condition
overall_paired_bootstrap_CI.csv	Paired bootstrap intervals for six-parameter aggregate scores by condition
overall_condition_summary.csv	Mean normalized MAE across the six targets for each model and condition
cnn_vs_gotham_inspired_transformer.png	Comparison plot of normalized MAE by test condition
run_manifest.json	Run configuration, split IDs, target-scaling values, device, and test-set caveat
CosmoTokens_Capacity_Matched_Results.xlsx	Excel workbook containing the principal result tables (not the full per-simulation prediction table)
CosmoTokens_Capacity_Matched_CSVs.zip	ZIP archive of the result CSV files, including per-simulation predictions


Scientific limitations and interpretation
- The nominal test split has been examined during earlier CosmoTokens development iterations. Test-set metrics and confidence intervals are therefore exploratory, not a pristine confirmatory benchmark.
- Three random seeds provide only a limited estimate of training variability.
- The Transformer and wider CNN are approximately, not exactly, parameter matched.
- The experiment uses one CAMELS simulation family, one field (stellar mass), one snapshot, and a fixed 64 × 64 representation.
- Synthetic masks do not reproduce all real survey effects, such as correlated noise, instrument response, survey footprints, or image reconstruction.
- Some target parameters, particularly the two AGN feedback parameters, are poorly recovered in this setup. This should not be interpreted as evidence that they are fundamentally unobservable.
- The nested-CV ExtraTrees diagnostic is complementary and is not a matched-protocol comparison with the neural-network test evaluation.
- This is a point-regression and robustness study, not a posterior-producing simulation-based inference method.
The results should be interpreted within the specified CAMELS dataset, map representation, architectures, and evaluation protocol. They should not be presented as constraints from observational data.
References
- Villaescusa-Navarro, F. et al. (2021), The CAMELS project: Cosmology and Astrophysics with Machine-learning Simulations, The Astrophysical Journal, 915, 71. https://doi.org/10.3847/1538-4357/abf7ba
- Villaescusa-Navarro, F. et al. (2022), The CAMELS Multifield Dataset: Learning the Universe's Fundamental Parameters with Artificial Intelligence, The Astrophysical Journal Supplement Series, 259, 61. https://doi.org/10.3847/1538-4365/ac5ab0
- Pandey, S. et al. (2024), Teaching Dark Matter Simulations to Speak the Halo Language (GOTHAM). https://arxiv.org/abs/2409.11401 · GOTHAM code
Data and citation
The raw CAMELS simulation products are not redistributed by this notebook. Obtain the data from the CAMELS collaboration and follow the access, usage, and citation terms associated with the dataset release. Cite the CAMELS project and Multifield Dataset papers when using the data.
License
No software license is specified by the notebook itself. Add an explicit LICENSE file to this repository before granting or implying permissions for reuse or redistribution. The CAMELS data remain subject to their own terms, independently of the code license.
