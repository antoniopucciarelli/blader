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

The following is an example:

![Example](./misc/runGIF.gif)

### Blade Data

The blade target data has to be stored into a ```*.dat``` file. This file must follow a defined order.

It is possible to comment the blade name at the first line using a ```#``` sign.

Following the first line comment, it is necessary to write down the coordinates of the blade in [x, y] faschion.
The coordinates have to be delimited by a ***space*** and the first coordinate is the one of the upper side tralining edge.
The last coordinate is the one of the lower side traliling edge. The coordinates have to follow a monotonic behavior in the upper side and lower side of the blade.

### DELTA ANGLE

The delta angle value allows the optimizer to use different optimization steps if an optimization error occours. The program automatically optimizes for different angle of attack because the Kulfan parametrization suffers of numerical errors in the camberline parametrization: rotating the blade coordinates allows the optimizer to find a working camberline for the global optimzation.

### SUCTION/PRESSURE

These two values define the suction side and pressure side DOF for the parametrization. The higher the more parameters used for the blade representation and the slower the optimization. It is suggested to start with a low DOF both for suction side and pressure side.
