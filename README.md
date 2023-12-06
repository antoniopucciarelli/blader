# blader

Blade generator/converter using Kulfan parametrization.

## Installation

After having cloned the repository, in the ``blader`` main directory, install the program (and all its dependencies) with:

```python

pip install .

```

## First run

For a first analysis, just run the program with:

```python

python3 src/guiLIB/gui.py

```

Then load the blade coordinates, it is possible to load these coordinates using an inner directory ```./blader/test/data/*.dat``` files.
These files are plain coordinate based blades; the program will convert this representation into Kulfan based representation.

For converting the blade into Kulfan parametrization: an optimization algorithm has to be used (in this program the **Nelder-Mead** method has been chosen).

It is possible to optimize the blade with different degree of freedom for the suction side or the pressure side of the blade.

The DOFs can be selected with the ``SUCTION`` and ``PRESSURE`` index in the window. The DOFs are between the [4, 20] interval.

Once everything is set, the optmized can be launched using the ``OPTIMZE`` button.

If the result is considered acceptable, it is possible to save the blade into a ``*.dat`` file. This file comprehends the Kulfan parameters and the blade coordinates.
