# Testing

## Writing unit tests

We use unit tests that ensure that the functions of each module and submodule work properly. If you add a new
functionality, you should also add a unit test. All tests are located in the [test
folder](https://github.com/tobac-project/tobac/tree/main/tobac/tests) The module {py:mod}`tobac.testing` may help to
create simple, idealized cases where objects can be tracked to test if the new features result in the expected outcome.

If you are unsure on how to construct tests and run tests locally, you can find additional documentation on
[pytest](https://docs.pytest.org/en/7.1.x/getting-started.html) and [pytest
fixtures](https://docs.pytest.org/en/6.2.x/fixture.html).

You will also notice that we report the test coverage, i.e. how much of our current code is triggered and thus tested by
the unit tests. When you submit a pull request, you will see if your code changes have increased or decreased the test
coverage. Ideally, test coverage should not decrease, so please make sure to add appropriate unit tests that cover
all newly added functions.


