# Replication Materials for *Language Disparities in Moderation Workforce Allocation by Social Media Platforms*

This repository contains the data and code necessary to replicate the analyses and figures from the paper  
**"Language Disparities in Moderation Workforce Allocation by Social Media Platforms."**

---

## Repository Structure

### `data/`
Contains the datasets used in the study.

- **`dsa_data/`**  
  - Moderator counts per language: `${PLATFORM}_moderator.csv`  
  - User counts per country (limited to LinkedIn, Twitter/X, and Snapchat, i.e., platforms with at least one language blind spot): `${PLATFORM}_user_count.csv`  

- **`calibration/`**  
  - **Annotated samples**: `${PLATFORM}/samples_annotated/`  
    - IDs (no text) of tweets or YouTube videos annotated as being in the language of interest or not, in line with Twitter/X and YouTube data-sharing guidelines.  
  - **Calibration models**: `${PLATFORM}/calibration_models/`  
    - Models used to calibrate raw *fastText* language inference scores.  

- **`plotted_data/`**  
  - Preprocessed data files used to generate the figures in the paper.  

---

### `code/`
Contains scripts and notebooks for modeling, calibration, and visualization.

- **Plotting**
  - `plotting.Rmd`: Generates plots for the paper.

- **Calibration Modeling**
  - `calibration_modeling_${PLATFORM}.ipynb`: Trains calibration models using annotated data.

- **Twitter/X Volume Estimation**  
  (Data is too large to process locally; these scripts enable large-scale processing.)  
  - `deploy_fasttext_twitterday.py`: Deploy *fastText* for language inference on Twitter data.  
  - `calibrate_scores_twitterday.py`: Apply calibration models to raw *fastText* scores.  
  - `draw_bootstrap_twitterday.py`: Draw bootstrapped samples for estimation.  

- **Moderator Count Normalization**
  - `normalized_moderator_count_analysis_${PLATFORM}.ipynb`: Produces normalized moderator counts for each platform.  

---

## Notes
- All Twitter/X and YouTube data shared here follow platform policies: only IDs are included, not raw text. We also do not share the representative samples of Twitter/X and YouTube and refer the reader to [this website](https://search.gesis.org/research_data/SDN-10.7802-2516) to get access to tweet IDs from the TwitterDay dataset.
- Calibration models are provided to allow replication of language inference adjustments.  

---

## Citation
If you use this repository, please cite the paper:

Tonneau, M., Liu, D., McGrady, R., Zheng, K., Schroeder, R., Zuckerman, E., & Hale, S. (2025, August 28). Language Disparities in Moderation Workforce Allocation by Social Media Platforms. https://doi.org/10.31235/osf.io/amfws_v1

