..
    How to contribute to the tobac project

How to contribute 
-------------------------

=========================
Code of conduct 
=========================

We are a multi-institutional and international community that aims to maintain and increase our diversity. We acknowledge that we all come with different experiences and capacities. Therefore, we strive to foster an inclusive and respectful environmentwhere we help and support each other. We welcome any types of contributions and believe that we together can create accessible, reusable, and maintanable code that empowers researchers and enables groundbreaking science. 

We would like to refer to the `Python code of conduct <https://www.python.org/psf/conduct/>`_ as we follow the same principlesfor communication and working with each other!

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

* **Create an issue**: If you have an idea for a new feature or a suggestion for any kind of code changes, please create an issue for this. We use `our issues <https://github.com/tobac-project/tobac/issues>`_ to milestones `Link text <https://github.com/tobac-project/tobac/milestones>`_, i.e. the different versions of **tobac** to come.
The issues act, therefore, not only as a place for reporting bugs, but also as a collection of *to do* points. 

You can also work on any issue that was created by somebody else and is already out there. A tip is to look for the **good first issue** label, if you are a new developer. These issues are usually fairly easy to address and can be good to practice our GitHub workflow. 

* https://github.com/tobac-project/tobac/blob/main/CONTRIBUTING.md
  
* **Create a pull request from your fork:** We use our personal forks of the tobac repository to create pull requests. This means that you have to first commit and push your local changes to your personal fork and then create a pull request from that fork: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork



===================================
Tips on working on your local code
===================================

* Install tobac package with :code:`pip install -e` . This allows you to directly test your local code changes as you run tobac. Instead of using the **tobac** version of the latest release, your local version of tobac will be used when you import **tobac** in a python script or IDE. Note that this way of installing a local package will use the code of the checked in branch, so this allows you also to test code while switching between branches.

* You can locally build the documentation page:

* Writing `meaningful commit messages <https://www.conventionalcommits.org/en/v1.0.0/>`_ can be very helpful for you and people who review your code to better understand the code changes.

=========================
Our branching strategy
=========================

While you can use any type of branching strategy and naming as you work in your personal fork, we have three branches in the tobac repository: 

* :code:``RC_*`
* :code:``dev_*`
* :code:``hotfix`

:code:``RC_*` is the release candidate of the next tobac version. The asterisk stands here for the specific tobac version: RC_vx.x.x (e.g. RC_v1.5.0). Pull requests to this branch need two reviews to be accepted before it can be merged into main. 

:code:``dev_*` is the development branch where we experiment with new features. This branch is perfectly suited to collaboratively work on a feature together with other **tobac** developers (see :doc:`mentoring`). In general, this branch is used for long-term, comprehensive code changes that might not be covered by a single pull request and where it might not be conceivable in which future **tobac** version to include the changes. There are no branch protection rules for this branch, which means that collaborators of our GitHub organization can directly push changes to this branch. Note that **dev_** can never directly merged into main, it has be merged into the release candidate branch RC_* first! There can be more than one `dev_*` branch, therefore it we recommand to describe the feature to work on in the respective branch (e.g. :code:`dev_xarray_transition`). 

:code:``hotfix` is the branch we use for hotfixes, i.e. bug fixes that need to be released as fast as possible because it influences people's code. This branch needs only one review before it can directly merged into :code:``main`.

In brief: **Unless you are collaboratively working on a comprehensive feature or on a hotfix, the branch to submit your pull request to is the next release candidate RC_v.x.x.x**


=========================
GitHub workflow
=========================

* Briefly describe how CI works
* Other github actions such as code format check 

=========================
Writing unit tests
=========================

* Make use of :py:mod:`tobac.testing`
* Pytest and pytest fixtures
* Test coverage

=========================
Add examples 
=========================

* Jupyter notebooks
* Upload example data to zenodo

=========================
Releasing a new version 
=========================

This is the checklist of steps for a release of a new **tobac** version:

* Bump version in `__init__.py `in :code:`hotfix`
* Add changelog in :code:`hotfix` 
* Regenerate example notebooks with the new version
* Get the two additional bug fixes into :code:`hotfix`
* Merge :code:`hotfix` into :code:`main` 
* Merge :code:`main` into release and dev branches 
* Delete :code:`hotfix` branch
* Create release
* Push release to conda-forge: https://github.com/tobac-project/tobac-notes/blob/main/uploading_to_conda-forge.md
* E-mail tobac mailing list
* Create new tag

