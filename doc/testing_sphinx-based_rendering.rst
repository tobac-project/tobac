How to check the Sphinx-based rendering
---------------------------------------


The workflow has been tested in a linux system. We aim to build a static
website out of the documentation material present in ``tobac``.

==================================
1. Preparing the Local Environment
==================================

-  **choose a separate place for your testing**

   I will use the temporary directory ``/tmp/website-testing`` which I
   need to create. You can use a dedicated place of your choice …

   .. code:: bash

      > mkdir /tmp/website-testing
      > cd /tmp/website-testing

   I will indicate my position now with the ``/tmp/website-testing>``
   prompt.

-  **get the official repository**

   .. code:: bash

      /tmp/website-testing> git clone https://github.com/tobac-project/tobac

   You might like to test a certain remote branch ``<BRANCH>`` then do:

   .. code:: bash

      /tmp/website-testing/tobac> git fetch --all
      /tmp/website-testing/tobac> git checkout -t origin/<BRANCH> 

-  **Python environment**

   -  create a python virtual env

      .. code:: bash
      
         /tmp/website-testing> python -m venv .python3-venv


   -  and install requirements
      
      .. code:: bash
         
         # deactivation conda is only necessary if your loaded conda before … 
         /tmp/website-testing> conda deactivate

         # activate the new env and upgrade ``pip`` 
         /tmp/website-testing> source .python3-venv/bin/activate 
         /tmp/website-testing> pip install –upgrade pip

         # now everything is installed into the local python env!
         /tmp/website-testing> pip install -r tobac/doc/requirements.txt

         # and also install RTD scheme 
         /tmp/website-testing> pip install sphinx_rtd_theme

      `pip`-based installation takes a bit of time, but is much faster than `conda`.
   

If the installation runs without problems, you are ready to build the website.


==================================
1. Building the Website
==================================

Actually, only few steps are needed to build the website, i.e.

-  **running sphinx for rendering**

   .. code:: bash

      /tmp/website-testing> cd tobac

      /tmp/website-testing/tobac> sphinx-build -b html doc doc/_build/html

   If no severe error appeared

-  **view the HTML content**

   .. code:: bash

      /tmp/website-testing/tobac> firefox doc/_build/html/index.html

==================================
3. Parsing Your Local Changes
==================================

Now, we connect to your locally hosted ``tobac`` repository and your
development branch.

-  **connect to your local repo**: Assume your repo is located at
   ``/tmp/tobac-testing/tobac``, then add a new remote alias and fetch
   all content with

   .. code:: bash

      /tmp/website-testing/tobac> git remote add local-repo /tmp/tobac-testing/tobac
      /tmp/website-testing/tobac> git fetch --all

-  **check your development branch out**: Now, assume the your
   development branch is called ``my-devel``, then do

   .. code:: bash

      # to get a first overview on available branches
      /tmp/website-testing/tobac> git branch --all

      # and then actually get your development branch
      /tmp/website-testing/tobac> git checkout -b my-devel local-repo/my-devel

   You should see your developments, now …

-  **build and view website again**

   .. code:: bash

      /tmp/website-testing/tobac> sphinx-build -M clean doc doc/_build
      /tmp/website-testing/tobac> sphinx-build -b html doc doc/_build/html
      /tmp/website-testing/tobac> firefox _build/html/index.html


==========================================
Option: Check Rendering of a Pull requests
==========================================

-  **check the pull request out**: Now, assume the PR has the ID ``<ID>`` and you define the branch name ``BRANCH_NAME`` as you like

   .. code:: bash

      # to get PR shown as dedicated branch
      /tmp/website-testing/tobac> git fetch upstream pull/ID/head:BRANCH_NAME

      # and then actually get this PR as branch
      /tmp/website-testing/tobac> git checkout BRANCH_NAME

   You should see the PR now ...

-  **build and view website again**

   .. code:: bash

      /tmp/website-testing/tobac> sphinx-build -M clean doc doc/_build
      /tmp/website-testing/tobac> sphinx-build -b html doc doc/_build/html
      /tmp/website-testing/tobac> firefox _build/html/index.html


