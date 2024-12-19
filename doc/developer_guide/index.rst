.. _Developer-Guide:

##############################
  Contributing
##############################

.. toctree::
   :maxdepth: 3
   :hidden:

   code_reviews
   code_structure
   mentoring
   testing_sphinx-based_rendering



Thank you for your eagerness to contribute to *tobac*. We have a step-by-step overview of most important points: https://github.com/tobac-project/tobac/blob/main/CONTRIBUTING.md on contributing, and this documentation goes into more detail.

=========================
Code of conduct
=========================

We are a multi-institutional and international community that aims to maintain and increase our diversity. We acknowledge that we all come with different experiences and capacities. Therefore, we strive to foster an inclusive and respectful environment where we help and support each other. We welcome any types of contributions and believe that we together can create accessible, reusable, and maintanable code that empowers researchers and enables groundbreaking science.

We would like to refer to the `Python code of conduct <https://www.python.org/psf/conduct/>`_ as we follow the same principles for communication and working with each other!

=========================
git basics
=========================

* **Create a Github account**: The first thing, you need to do is to `create a GitHub account <https://docs.github.com/en/get-started/signing-up-for-github/signing-up-for-a-new-github-account>`_ if you do not already have one.

* **Get familiar with the basics of GitHub and git**:
   * Getting started with the `basics <https://docs.github.com/en/get-started/quickstart/hello-world>`_
   * Learn about `branches <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches>`_
   * Learn about `forks <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_
   * Learn about `pull requests <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_
   * Learn about `how to commit and push changes from your local repository <https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github>`_

* **Create an issue**: If you have an idea for a new feature or a suggestion for any kind of code changes, please create an issue for this. We sort `our issues <https://github.com/tobac-project/tobac/issues>`_ into `milestones <https://github.com/tobac-project/tobac/milestones>`_ to priorize work and manage our workflow, i.e. the different versions of **tobac** to come.

  The issues act, therefore, not only as a place for reporting bugs, but also as a collection of *to do* points.

* **Work on an issue**: You can also work on any issue that was created by somebody else and is already out there. A tip is to look for the **good first issue** label, if you are a new developer. These issues are usually fairly easy to address and can be good to practice our GitHub workflow.


* **Create a pull request from your fork:** We use our personal forks of the tobac repository to create pull requests. This means that you have to first commit and push your local changes to your personal fork and then create a pull request from that fork: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork

===================================
Writing proper documentation
===================================

Please provide **Numpy Docstrings** for all new functions.

**Example**:

.. code::

   '''
   Calculates centre of gravity and mass for each individually tracked cell in the simulation.


    Parameters
    ----------
    tracks : pandas.DataFram
        DataFrame containing trajectories of cell centres

    param mass : iris.cube.Cube
        cube of quantity (need coordinates 'time', 'geopotential_height','projection_x_coordinate' and
        'projection_y_coordinate')

    param mask : iris.cube.Cube
        cube containing mask (int > where belonging to cloud volume, 0 everywhere else )


    Returns
    -------
    track_out : pandas.DataFrame
        Dataframe containing t,x,y,z positions of centre of gravity and total cloud mass each tracked cells
        at each timestep

    '''



===================================
Tips on working on your local code
===================================

* Install tobac package with :code:`pip install -e`
    * This allows you to directly test your local code changes as you run tobac. Instead of using the **tobac** version of the latest release, your local version of tobac will be used when you import **tobac** in a python script or IDE.
    * *Note that* this way of installing a local package will use the code of the checked in branch, so this allows you also to test code while switching between branches.

* You can locally **build the documentation page**:
    * see :doc:`testing_sphinx-based_rendering`

* Writing `meaningful commit messages <https://www.conventionalcommits.org/en/v1.0.0/>`_ can be very helpful for you and people who review your code to better understand the code changes.


=========================
Our branching strategy
=========================

While you can use any type of branching strategy and naming as you work in your personal fork, we have three branches in the tobac repository:

* :code:`RC_*`
* :code:`dev_*`
* :code:`hotfix`

:code:`RC_*` is the release candidate of the next tobac version. The asterisk stands here for the specific tobac version: RC_vx.x.x (e.g. RC_v1.5.0). Pull requests to this branch need two reviews to be accepted before it can be merged into main.

:code:`dev_*` is the development branch where we experiment with new features. This branch is perfectly suited to collaboratively work on a feature together with other **tobac** developers (see :doc:`mentoring`). In general, this branch is used for long-term, comprehensive code changes that might not be covered by a single pull request and where it might not be conceivable in which future **tobac** version to include the changes. There are no branch protection rules for this branch, which means that collaborators of our GitHub organization can directly push changes to this branch. Note that **dev_** can never directly merged into main, it has be merged into the release candidate branch :code:`RC_*` first! There can be more than one `dev_*` branch, therefore it we recommend to describe the feature to work on in the respective branch (e.g. :code:`dev_xarray_transition`).

:code:`hotfix` is the branch we use for hotfixes, i.e. bug fixes that need to be released as fast as possible because it influences people's code. This branch needs only one review before it can directly merged into :code:`main`.

In brief: **Unless you are collaboratively working on a comprehensive feature or on a hotfix, the branch to submit your pull request to is the next release candidate RC_v.x.x.x**


=========================
GitHub workflow
=========================

We use several `GitHub actions <https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions>`_ to
assure continuous integration and to enable an efficient code development and release process. Our workflow
configuration can be found in
`.github/workflows <https://github.com/tobac-project/tobac/tree/main/.github/workflows>`_ and encompass

* check that code is formatted using the latest stable version of black
* linting of the latest code changes that checks the code quality and results in a score compared to the most recent released version
* check of the zenodo JSON file that ensures that the citation is correct
* check that all unit tests pass (including testing on multiple operating systems) and report test coverage
* check that the example jupyter notebooks run without problems

=========================
Writing unit tests
=========================

We use unit tests that ensure that the functions of each module and submodule work properly. If you add a new
functionality, you should also add a unit test. All tests are located in the `test
folder <https://github.com/tobac-project/tobac/tree/main/tobac/tests>`_ The module :py:mod:`tobac.testing` may help to
create simple, idealized cases where objects can be tracked to test if the new features result in the expected outcome.

If you are unsure on how to construct tests and run tests locally, you can find additional documentation on
`pytest <https://docs.pytest.org/en/7.1.x/getting-started.html>`_ and `pytest
fixtures <https://docs.pytest.org/en/6.2.x/fixture.html>`_.

You will also notice that we report the test coverage, i.e. how much of our current code is triggered and thus tested by
the unit tests. When you submit a pull request, you will see if your code changes have increased or decreased the test
coverage. Ideally, test coverage should not decrease, so please make sure to add appropriate unit tests that cover
all newly added functions.

=========================
Add examples
=========================

In addition to the unit tests, we aim to provide examples on how to use all functionalities and how to choose different
tracking parameters. These `examples <https://github.com/tobac-project/tobac/tree/main/examples>`_ are in form of jupyter
notebooks and can be based on simple, idealized test cases or real data. We strongly encourage the use of real data that
is publicly accessible, but another option for new examples with real data is to either upload the data to our `zenodo
repository <https://zenodo.org/records/10863405>`_ or create your own data upload on zenodo. Please include the name "tobac" in the data title for the latter.

=========================
Releasing a new version
=========================

This is the checklist of steps for a release of a new **tobac** version:

* Bump version in :code:`__init__.py` in :code:`RC_vXXX`
* Add changelog in :code:`RC_vXXX`
* Regenerate example notebooks with the new version
* Merge :code:`RC_vXXX` into :code:`main`
* Merge updated :code:`main` branch back into release and dev branches
* Delete :code:`RC_vXXX` branch
* Create release
* Push release to conda-forge: https://github.com/tobac-project/tobac-notes/blob/main/uploading_to_conda-forge.md
* Create new tag
* E-mail tobac mailing list
