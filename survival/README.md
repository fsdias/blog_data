# Survival of the wormiest

Data and figures for the Substack post [Survival of the wormiest: a Bayesian approach to deworming sheep](https://fsdias.substack.com/p/stan-sheep-and-bayesian-survival) (September 2023).

This post is a short tutorial on Bayesian survival analysis in Stan. It fits a Weibull survival model with censoring to an experiment on deworming wild sheep. The dataset comes from Michael J. Crawley, *Statistics: An Introduction Using R* (Wiley).

## Files

| File | Description |
|---|---|
| `sheep.deaths.csv` | The dataset: 150 male wild sheep, 50 per treatment group |
| `fig1_surv.png` … `fig6_counter.png` | Figures shown in the post |

## `sheep.deaths.csv`

| Column | Description |
|---|---|
| `death` | Months the animal survived. The study lasted 50 months, so a value of 50 means the animal outlived the experiment. |
| `status` | `1` = died during follow-up, `0` = censored (alive at the end). 130 deaths and 20 censored animals. |
| `weight` | Body weight at the start of the study (kg) |
| `group` | Treatment: `A` = high dose of dewormer, `B` = low dose, `C` = placebo |

## Figures

| File | Content |
|---|---|
| `fig1_surv.png` | Exploratory survival curves |
| `fig2_prior.png` | Prior predictive check |
| `fig3_val.png` | Model check: the Kaplan–Meier curve compared with model-predicted survival and its uncertainty band |
| `fig4_params.png` | Posterior intervals for the group intercepts (`a_group`) and weight slopes (`b_group`) |
| `fig5_ribbon.png` | Predicted survival time against body weight for each treatment |
| `fig6_counter.png` | Contrasts between treatments (A vs B, A vs C, B vs C) across body weight |

## Quick start

```r
d <- read.csv("https://raw.githubusercontent.com/fsdias/blog_data/main/survival/sheep.deaths.csv")
str(d)
```

The full R and Stan code is in the post itself. It uses `tidyverse`, `survival`/`survminer`, `cmdstanr`, `posterior`, `bayesplot` and `tidybayes`.
