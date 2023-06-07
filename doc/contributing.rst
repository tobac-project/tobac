How to contribute 
-------------------------

=========================
Code of conduct 
=========================

* We, the **tobac** community, value diversity and acknowledge that we all come with different experiences and capacities. We strive to foster an inclusive and respectful space for collaboration. To achieve this, we would like to refer to the `Python code of conduct <https://www.python.org/psf/conduct/>`_ as we follow the same principles for communicating and interacting with each other!

=========================
git basics
=========================

* Create a Github account

The first thing, you need to do is to `create a GitHub account <https://docs.github.com/en/get-started/signing-up-for-github/signing-up-for-a-new-github-account>`_ if you do not already have one. 

* Get familiar with the basics of GitHub and git:

- `Getting started < https://docs.github.com/en/get-started/quickstart/hello-world>`_
- `Learn about branches <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches >`_ 
- `Learn about forks <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_
- `Learn about pull requests < https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_
- `Learn how to commit and push changes from your local repository< https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github>`_ 

* Create an issue

If you have an idea for a new feature or a suggestion for any kind of code changes, please create an issue for this. We use `our issues <https://github.com/tobac-project/tobac/issues>`_ to milestones `Link text <https://github.com/tobac-project/tobac/milestones>`_, i.e. the different versions of **tobac** to come.
The issues act, therefore, not only as a place for reporting bugs, but also as a collection of *to do* points. 

You can also work on any issue that was created by somebody else and is already out there. A tip is to look for the **good first issue** label, if you are a new developer. These issues are usually fairly easy to address and can be good to practice our GitHub workflow. 

* Link to CONTRIBUTING.md (?)

* Tips when working with your local code

Install tobac package with pip install -e . This allows you to directly test your local code changes as you run tobac. Instead of using the **tobac** version of the latest release, your local version of tobac will be used when you import **tobac** in a python script or IDE. Note that this way of installing a local package will use the code of the checked in branch, so this allows you also to test code while switching between branches.

- How to locally build the documentation page  


* Create a pull request from your fork 

We use our personal forks of the tobac repository to create pull requests. This means that you have to first commit and push your local changes to your personal fork and then create a pull request from that fork:

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork


=========================
Branching strategy
=========================

While you can use any type of branching strategy and naming as you work in your personal fork, we have three branches in the tobac repository: 

* `RC_*`
* `dev_*`
* `hotfix`

`RC_*` is the release candidate of the next tobac version. The asterisk stands here for the specific tobac version: RC_vx.x.x (e.g. RC_v1.5.0). Pull requests to this branch need two reviews to be accepted before it can be merged into main. 

`dev_*` is the development branch where we experiment with new features. This branch is perfectly suited to collaboratively work on a feature together with other **tobac** developers (see :doc:`mentoring`). In general, this branch is used for long-term, comprehensive code changes that might not be covered by a single pull request and where it might not be conceivable in which future **tobac** version to include the changes. There are no branch protection rules for this branch, which means that collaborators of our GitHub organization can directly push changes to this branch. Note that **dev_** can never directly merged into main, it has be merged into the release candidate branch RC_* first! There can be more than one `dev_*` branch, therefore it we recommand to describe the feature to work on in the respective branch (e.g. `dev_xarray_transition`). 

`hotfix` is the branch we use for hotfixes, i.e. bug fixes that need to be released as fast as possible because it influences people's code. This branch needs only one review before it can directly merged into `main`.

After a release, we need to make sure that the latest changes of main are merged back into existing `RC_*` and `dev_*` branches and that the branch names are updated accordingly (e.g. replace `RC_v1.5.0` with the next version to come). The hotfix is directly deleted when merged into main. It can be re-created whenever a new hotfix comes up. 

The most important thing to remember from this when you are a new contributor: **The common branch to submit the pull request to is the next release candidate RC_v.x.x.x**


=========================
Writing unit tests
=========================

* CI and other GitHub actions (code formatting etc. )
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



