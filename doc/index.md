---
html_theme.sidebar_secondary.remove:
sd_hide_title: true
---
<!-- sphinx-autobuild ./doc/ /Users/seanfreeman/Documents/Research/tobac_dev/tobac_docs_refresh/dev_docs/ -->
<!-- CSS overrides on the homepage only -->
<style>
.bd-main .bd-content .bd-article-container {
max-width: 70rem; /* Make homepage a little wider instead of 60em */
}
/* Extra top/bottom padding to the sections */
article.bd-article section {
padding: 3rem 0 7rem;
}
/* Override all h1 headers except for the hidden ones */
h1:not(.sd-d-none) {
font-weight: bold;
font-size: 48px;
text-align: center;
margin-bottom: 4rem;
}
/* Override all h3 headers that are not in hero */
h3:not(#hero h3) {
  font-weight: bold;
  text-align: center;
}
</style>

(homepage)=
# tobac - Tracking and Object-Based Analysis of Clouds

<div id="hero">

<div id="hero-left">  <!-- Left side of the hero section -->
  <h2 style="font-size: 60px; font-weight: bold; margin: 2rem auto 0;"><em>tobac</em></h2>
  <h3 style="font-weight: bold; margin-top: 0;">Tracking atmospheric phenomena with <em>your data</em></h3>
  <p><em>tobac</em> (Tracking and Object-Based Analysis of Clouds) is a package that identifies and tracks 
atmospheric phenomena, enabling object-based analysis in model, observational, and remote sensing datasets.</p>

<div class="homepage-button-container">
  <div class="homepage-button-container-row">
  <a href="./getting_started/index.html" class="homepage-button primary-button">Get Started</a>
  <a href="./examples/index.html" class="homepage-button secondary-button">Examples</a>
  </div>
  <div class="homepage-button-container-row">
  <a href="./tobac.html" class="homepage-button-link">See API Reference →</a>
  </div>
</div>
</div>  <!-- End Hero Left -->

<div id="hero-right">  <!-- Start Hero Right -->

::::::{grid} 2
:gutter: 3

:::::{grid-item-card}
:link: examples/Example_vorticity_tracking_model/Example_vorticity_tracking_model.html
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Tracking a cyclones with vorticity in a model
:::

:::{image} ./_static/thumbnails/Example_vorticity_tracking_model_Thumbnail.png
:::
:::::

:::::{grid-item-card}
:link: examples/Example_OLR_Tracking_satellite/Example_OLR_Tracking_satellite.html
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Tracking OLR from Satellite
:::

:::{image} _static/thumbnails/Example_OLR_Tracking_satellite_Thumbnail.png
:::
:::::

:::::{grid-item-card}
:link: examples/Example_Track_on_Radar_Segment_on_Satellite/Example_Track_on_Radar_Segment_on_Satellite.html
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Track on Radar and Combine with Satellite
:::

:::{image} _static/thumbnails/Example_Track_on_Radar_Segment_on_Satellite_Thumbnail.png
:::
:::::

:::::{grid-item-card}
:link: examples/Example_Updraft_Tracking/Example_Updraft_Tracking.html
:shadow: none
:class-card: example-gallery

:::{div} example-img-plot-overlay
Updraft Tracking
:::

:::{image} _static/thumbnails/Example_Updraft_Tracking_Thumbnail.png
:::
:::::

::::::

<!-- grid ended above, do not put anything on the right of markdown closings -->

</div>  <!-- End Hero Right -->
</div>  <!-- End Hero -->



# *tobac* Use and Development

::::{grid} 1 1 2 2

:::{grid-item}

<h3>Contributions</h3>

Contributions and issue reports are very welcome at
[the GitHub repository](https://github.com/tobac-project/tobac).
We have a {ref}`developer guide <Developer-Guide>` to help you through the process. We also offer mentoring for new contributors; see the {ref}`developer guide <Developer-Guide>` for details. 

<h3>Scientific Use</h3>

_tobac_ has been used in over 40 peer-reviewed publications, a subset of which are listed on our {ref}`publications page <Refereed-Publications>`. 

:::
:::{grid-item}

<h3>Citation</h3>

If you use *tobac* for scholarly work, please cite it using the software citation: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2577046.svg)](https://doi.org/10.5281/zenodo.2577046), the original paper: 

> Heikenfeld, M., Marinescu, P. J., Christensen, M., Watson-Parris, D., Senf, F., van den Heever, S. C., and Stier, P.: tobac 1.2: towards a flexible framework for tracking and analysis of clouds in diverse datasets, Geosci. Model Dev., 12, 4551–4570, https://doi.org/10.5194/gmd-12-4551-2019, 2019

or, if you are using features introduced from v1.5.0 or above, please also cite the v1.5.0 paper:
> Sokolowsky, G. A., Freeman, S. W., Jones, W. K., Kukulies, J., Senf, F., Marinescu, P. J., Heikenfeld, M., Brunner, K. N., Bruning, E. C., Collis, S. M., Jackson, R. C., Leung, G. R., Pfeifer, N., Raut, B. A., Saleeby, S. M., Stier, P., and van den Heever, S. C.: tobac v1.5: introducing fast 3D tracking, splits and mergers, and other enhancements for identifying and analysing meteorological phenomena, Geosci. Model Dev., 17, 5309–5330, https://doi.org/10.5194/gmd-17-5309-2024, 2024.

:::


::::




:::{toctree}
:maxdepth: 1
:hidden:

Getting Started<getting_started/index>
Example Gallery<examples/index>
User Guide<userguide/index>
Developer Guide<developer_guide/index>
API <api/index>
:::