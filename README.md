# pyvlasiator

<p align="center">
  <a href="https://github.com/henry2004y/pyvlasiator/actions">
    <img src="https://github.com/henry2004y/pyvlasiator/actions/workflows/CI.yml/badge.svg">
  </a>
  <a href="https://henry2004y.github.io/pyvlasiator/">
    <img src="https://img.shields.io/badge/docs-dev-blue">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue">
  </a>
  <a href="https://app.codecov.io/gh/henry2004y/pyvlasiator/">
    <img src="https://img.shields.io/codecov/c/github/henry2004y/pyvlasiator">
  </a>
</p>

Python package for processing Vlasiator data.

## Installation

`pyvlasiator` has not been registered yet since it's under development and being considered
as an experiment to refactor [`analysator`](https://github.com/fmihpc/analysator).
Once reaching a stable stage we plan to release the package under `pip`, and then it can be
installed via

```bash
$ pip install pyvlasiator
```

## Usage

`pyvlasiator` can be used to process VLSV files generated from Vlasiator.

```python
from pyvlasiator.vlsv import Vlsv

file = "test.vlsv"
meta = Vlsv(file)
```

Plotting is supported via Matplotlib. For more detailed usage, please refer to the [documentation](https://henry2004y.github.io/pyvlasiator/).

## License

`pyvlasiator` was created by Hongyang Zhou. It is licensed under the terms of the MIT license.
