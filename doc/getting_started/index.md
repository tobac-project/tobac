# Getting Started

```{toctree}
:hidden: true
:maxdepth: 3

installguide
data_input
feature_detection_overview
tracking_basics
segmentation
merge_split
plotting
```

Welcome to *tobac*! We hope that you will find this tracking library to be useful in your research or operational work. To help you get started, we have put together this quickstart guide with some of the basic information you need to know to get started tracking your data with *tobac*.

The easiest way to get started is by perusing our [Example Gallery](/examples/), which contains a number of worked examples using different kinds of input data from start to finish. 

![An example of tobac running on GOES-R series ABI visible data](/images/tobac_vis_example.gif)


## *tobac* Core Concepts

*tobac* can be generally thought of as five steps:

1. [Data Preparation](./data_input)
2. [Feature Detection](./feature_detection_overview), which **identifies** the features (e.g., clouds, fronts, vorticity maxima, etc.) of interest
3. [Tracking](./tracking_basics), which **links** the features over time
4. [Segmentation](./segmentation), which **identifies the regions** represented by every feature
   1. This is an optional feature, and is not required for all kinds of science questions.
5. Analysis and Postprocessing, which **enables** you to answer the science questions you are interested in

Each of these steps are *modular*, meaning that if you want to replace any one of these steps with your own code or procedure, you can, so long as you follow the same data model. 

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
