# Master Thesis - Taming Complexity on Digital Platforms Using Generative AI - An Agent Based Model Approach

## Summary

This program is the simulation base of master thesis. An agent based simulation method based on mesa library.

## Installation

To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
    $ pip install -r requirements.txt
```

## How to Run

To run the model single time simulation, run ``run.py`` in this directory. It launches a model visualization server

Then open your browser to [http://127.0.0.1:8521/](http://127.0.0.1:8521/) and press Reset, then Run.

To run the batched simulation and generate csv file for analysis, run ``model.py``.

To analysis combined simulations, run ``combined_simulations.ipynb`` under code file.

To view sensitivity analysis, run ``sensitivity analysis.ipynb`` under code file.

## Files

Code File:

* ``agent.py``: Contains the agent class, and the overall agent class.
* ``model.py``: Contains the model class, and the overall model class. This is the most important file to define the parameter of the model, under ''params_ranges'' function.
* ``server.py``: Defines classes for visualizing the model (network layout) in the browser via Mesa's modular server, and instantiates a visualization server.

Plots File:

Contains all the plots for the thesis.

Simulation Data File:

Contains all simulation data generated from ``model.py``.

## Further Reading

The mesa library:
https://github.com/projectmesa/mesa
