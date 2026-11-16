# Portugal's birding awakening

Data and code annex for the Substack post [Portugal's birding awakening: temporal and spatial trends of a growing hobby](https://fsdias.substack.com/p/portugals-birding-awakening-temporal) (October 2024).

This post uses eBird checklists to look at how birding in Portugal grew between 2002 and 2023, and which features of a municipality go with more birding activity.

## Files

| File | Description |
|---|---|
| `annex1-substack-birding-post.pdf` | Annex with the full R/Stan/Arrow code and its output (27 pages). It covers converting the eBird Basic Dataset to Parquet, the temporal trends, and the Bayesian hierarchical model of checklists per municipality. |
| `ebird_paper_data.gpkg` | GeoPackage with two layers (see below). |

## `ebird_paper_data.gpkg`

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

## Quick start

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

## Raw eBird data

The temporal analyses (section 3 of the annex) need the **eBird Basic Dataset for Portugal** (release Dec-2023, about 4 GB). That file is not in this repository because eBird's terms of use do not allow it to be redistributed. You can request it free of charge from [ebird.org/data/download](https://ebird.org/data/download).

## Software

R with `arrow`, `tidyverse`, `lubridate`, `sf`, `GGally`, `ggpubr`, `cmdstanr` (with CmdStan installed), `loo`, `bayesplot`, `posterior` and `tidybayes`.
