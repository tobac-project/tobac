# Branching and Workflow

## Our branching strategy

While you can use any type of branching strategy and naming as you work in your personal fork, we have three branches in the tobac repository:

- {code}`RC_*`
- {code}`dev_*`
- {code}`hotfix`

{code}`RC_*` is the release candidate of the next tobac version. The asterisk stands here for the specific tobac version: RC_vx.x.x (e.g. RC_v1.5.0). Pull requests to this branch need one review to be accepted before it can be merged into main. The PR author is ultimately responsible for how many reviews they require, and they or the 1st reviewer may also request a second review by tagging reviewers in the comment on the PR on Github. If an assigned reviewer does not respond on GitHub, contact the assigned person via email. 

{code}`dev_*` is the development branch where we experiment with new features. This branch is perfectly suited to collaboratively work on a feature together with other **tobac** developers (see {doc}`mentoring`). In general, this branch is used for long-term, comprehensive code changes that might not be covered by a single pull request and where it might not be conceivable in which future **tobac** version to include the changes. There are no branch protection rules for this branch, which means that collaborators of our GitHub organization can directly push changes to this branch. Note that **dev\_** can never directly merged into main, it has be merged into the release candidate branch {code}`RC_*` first! There can be more than one `dev_*` branch, therefore it we recommend to describe the feature to work on in the respective branch (e.g. {code}`dev_xarray_transition`).

{code}`hotfix` is the branch we use for hotfixes, i.e. bug fixes that need to be released as fast as possible because it influences people's code. This branch needs only one review before it can directly merged into {code}`main`.

In brief: **Unless you are collaboratively working on a comprehensive feature or on a hotfix, the branch to submit your pull request to is the next release candidate RC_v.x.x.x**

## GitHub workflow

We use several [GitHub actions](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions) to
assure continuous integration and to enable an efficient code development and release process. Our workflow
configuration can be found in
[.github/workflows](https://github.com/tobac-project/tobac/tree/main/.github/workflows) and encompass

- check that code is formatted using the latest stable version of black
- linting of the latest code changes that checks the code quality and results in a score compared to the most recent released version
- check of the zenodo JSON file that ensures that the citation is correct
- check that all unit tests pass (including testing on multiple operating systems) and report test coverage
- check that the example jupyter notebooks run without problems

## Releasing a new version

This is the checklist of steps for a release of a new **tobac** version:

- Bump version in {code}`__init__.py` in {code}`RC_vXXX`
- Add changelog in {code}`RC_vXXX`
- Regenerate example notebooks with the new version
- Merge {code}`RC_vXXX` into {code}`main`
- Merge updated {code}`main` branch back into release and dev branches
- Delete {code}`RC_vXXX` branch
- Create release
- Push release to conda-forge: <https://github.com/tobac-project/tobac-notes/blob/main/uploading_to_conda-forge.md>
- Create new tag
- E-mail tobac mailing list
