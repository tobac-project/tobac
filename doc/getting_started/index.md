# Getting Started

```{toctree}
:hidden: true
:maxdepth: 3

installguide
data_input
data_output
feature_detection_overview
tracking_basics
segmentation
merge_split
plotting
get_help
```

Welcome to *tobac*! We hope that you will find this tracking library to be useful in your research or operational work. To help you get started, we have put together this quickstart guide with some of the basic information you need to know to get started tracking your data with *tobac*.

The easiest way to get started is by perusing our [Example Gallery](/examples/index), which contains a number of worked examples using different kinds of input data from start to finish. 

## What is *tobac*?

*tobac*, or Tracking and Object-Based Analysis of Clouds, is an open-source package designed to enable objective, automatic identification and tracking of phenomena in the Earth System. Such tracks enable **lifecycle-centric** or **object-based** analysis, allowing for a multi-Lagrangian view of clouds, storms, cyclones, or other phenomena. *tobac* functions on any regular (or semi-regular) grid, and on any variable. Where possible, parameters are physical, and are described in the documentation. 

*tobac* is designed from the ground up to be transparent and modular. Each component of *tobac* can be replaced if you want. For example, if you have a pre-built set of features, you can still use *tobac* to track the features through time.  

## What has *tobac* been used to track?

*tobac* has been used in [several dozen peer-reviewed manuscripts](../userguide/publications), identifying and tracking fields as diverse as dust in cloud model output, clouds from geostationary IR, and vorticity in large-scale model output. A subset of these differnet use cases can be seen in the [example gallery](../examples/index) The key requirement for identification of a phenomena is that the variable that you are tracking on is continuously increasing (e.g., vertical velocity) or decreasing (e.g., brightness temperature), and that the variable be able to be represented on a regular or semi-regular grid. [^comingsoonugrid]


## *tobac* Core Concepts

*tobac* can be generally thought of as five steps:

1. [Data Preparation](./data_input)
2. [Feature Detection](./feature_detection_overview), which **identifies** the features (e.g., clouds, fronts, vorticity maxima, etc.) of interest
3. [Tracking](./tracking_basics), which **links** the features over time
4. [Segmentation](./segmentation), which **identifies the regions** represented by every feature
   1. This is an optional feature, and is not required for all kinds of science questions.
5. Analysis and Postprocessing, which **enables** you to answer the science questions you are interested in

Each of these steps are *modular*, meaning that if you want to replace any one of these steps with your own code or procedure, you can, so long as you follow the same data model. 

```{mermaid}
flowchart LR
  A[Feature Detection] -- or --> B
  B[Tracking] --> C[Segmentation]
  A -- or --> C
    click A "./feature_detection_overview.html" "Feature Detection"
    click B "./tracking_basics.html" "Tracking"
    click C "./segmentation.html" "Segmentation"
```


## *tobac* installation

The easiest way to [install *tobac*](./installguide) and all dependencies is installing via `conda-forge`:
::::{tab-set}
:::{tab-item} `mamba`
:sync: mamba
```shell
mamba install -c conda-forge tobac
```
:::
:::{tab-item} `conda`
:sync: conda
```shell
conda install -c conda-forge tobac
```
:::
::::

[^comingsoonugrid]: Support for more diverse grids is planned in a future version of *tobac*. To see all currently planned large projects, look at the publicly-available *tobac* roadmap: https://github.com/tobac-project/tobac-roadmap/blob/master/tobac-roadmap-main.md