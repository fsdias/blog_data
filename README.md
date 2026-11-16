# blog_data

Data, code annexes and figures for posts on my Substack, [fsdias.substack.com](https://fsdias.substack.com).

Each folder holds the material for one post. The posts read their data straight from this repository through the `raw.githubusercontent.com` URLs, so you can run the code without downloading anything first.

| Folder | Post | Published |
|---|---|---|
| [`birding/`](birding) | [Portugal's birding awakening: temporal and spatial trends of a growing hobby](https://fsdias.substack.com/p/portugals-birding-awakening-temporal) | October 2024 |
| [`survival/`](survival) | [Survival of the wormiest: a Bayesian approach to deworming sheep](https://fsdias.substack.com/p/stan-sheep-and-bayesian-survival) | September 2023 |

---

## `birding/`: Portugal's birding awakening

This post uses eBird checklists to look at how birding in Portugal grew between 2002 and 2023, and which features of a municipality go with more birding activity.

### Files

| File | Description |
|---|---|
| `annex1-substack-birding-post.pdf` | Annex with the full R/Stan/Arrow code and its output (27 pages). It covers converting the eBird Basic Dataset to Parquet, the temporal trends, and the Bayesian hierarchical model of checklists per municipality. |
| `ebird_paper_data.gpkg` | GeoPackage with two layers (see below). |

### `ebird_paper_data.gpkg`

**Layer `ebird_paper_data`**: 278 mainland Portuguese municipalities (multipolygons, ETRS89 / Portugal TM06). The model in the annex uses these columns:

| Column | Description |
|---|---|
| `municip` | Municipality name |
| `district` | District the municipality belongs to (grouping level in the model) |
| `n_lists` | Number of eBird checklists submitted in the municipality, 2002–2023 (response variable) |
| `s` | Total number of species recorded in the municipality, 2002–2023 |
| `pop` | Resident population |
| `os` | Overnight stays in tourist accommodation in 2019, a measure of tourist attractiveness. Stored as text; convert it with `as.numeric()`, which turns the missing values into `NA`. |
| `iba` | Area of Important Bird Areas (IBA) in the municipality |

The layer also has `area_ha` (municipality area, ha), `pa`, `income_avg` and `lud`, which the post does not use.

**Layer `IBA`**: the 54 Important Bird Areas in mainland Portugal (multipolygons, WGS 84). It includes code, name, criteria and area.

### Quick start

```r
library(sf)

munic <- st_read(
  "https://raw.githubusercontent.com/fsdias/blog_data/main/birding/ebird_paper_data.gpkg",
  layer = "ebird_paper_data"
)

iba <- st_read(
  "https://raw.githubusercontent.com/fsdias/blog_data/main/birding/ebird_paper_data.gpkg",
  layer = "IBA"
)
```

### Raw eBird data

The temporal analyses (section 3 of the annex) need the **eBird Basic Dataset for Portugal** (release Dec-2023, about 4 GB). That file is not in this repository because eBird's terms of use do not allow it to be redistributed. You can request it free of charge from [ebird.org/data/download](https://ebird.org/data/download).

### Software

R with `arrow`, `tidyverse`, `lubridate`, `sf`, `GGally`, `ggpubr`, `cmdstanr` (with CmdStan installed), `loo`, `bayesplot`, `posterior` and `tidybayes`.

---

## `survival/`: Survival of the wormiest

This post is a short tutorial on Bayesian survival analysis in Stan. It fits a Weibull survival model with censoring to an experiment on deworming wild sheep. The dataset comes from Michael J. Crawley, *Statistics: An Introduction Using R* (Wiley).

### Files

| File | Description |
|---|---|
| `sheep.deaths.csv` | The dataset: 150 male wild sheep, 50 per treatment group |
| `fig1_surv.png` … `fig6_counter.png` | Figures shown in the post |

### `sheep.deaths.csv`

| Column | Description |
|---|---|
| `death` | Months the animal survived. The study lasted 50 months, so a value of 50 means the animal outlived the experiment. |
| `status` | `1` = died during follow-up, `0` = censored (alive at the end). 130 deaths and 20 censored animals. |
| `weight` | Body weight at the start of the study (kg) |
| `group` | Treatment: `A` = high dose of dewormer, `B` = low dose, `C` = placebo |

### Figures

| File | Content |
|---|---|
| `fig1_surv.png` | Exploratory survival curves |
| `fig2_prior.png` | Prior predictive check |
| `fig3_val.png` | Model check: the Kaplan–Meier curve compared with model-predicted survival and its uncertainty band |
| `fig4_params.png` | Posterior intervals for the group intercepts (`a_group`) and weight slopes (`b_group`) |
| `fig5_ribbon.png` | Predicted survival time against body weight for each treatment |
| `fig6_counter.png` | Contrasts between treatments (A vs B, A vs C, B vs C) across body weight |

### Quick start

```r
d <- read.csv("https://raw.githubusercontent.com/fsdias/blog_data/main/survival/sheep.deaths.csv")
str(d)
```

The full R and Stan code is in the post itself. It uses `tidyverse`, `survival`/`survminer`, `cmdstanr`, `posterior`, `bayesplot` and `tidybayes`.

---

## Reuse

The code and my derived data are free to reuse, but please cite the corresponding post. Third-party data keep their original terms: eBird data are © Cornell Lab of Ornithology, and the sheep dataset comes from Crawley's textbook. If you spot an error, please open an issue or leave a comment on the post.
