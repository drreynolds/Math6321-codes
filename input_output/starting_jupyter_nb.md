# Basic instructions on how to launch a Jupyter notebook, depending on the age of your Python/Jupyter installation

D.R. Reynolds  
Math 6321 @ SMU  
Fall 2020

## Current versions of the Jupyter notebook

To launch a from-scratch Jupyter notebook server on your machine, with interface popping up as a new tab in your default web browser, at the Linux/OSX command line run:

```bash
jupyter notebook
```

or

```bash
jupyter-notebook
```

Alternately, if you already have a notebook file that you wish to open/modify (typically this has the extension .ipynb):

```bash
jupyter notebook <notebookname>
```

or

```bash
jupyter-notebook <notebookname>
```

## Older versions (i.e., when it was still called "iPython notebook")

To launch an iPython notebook, with pylab already enabled, and where
graphics will be displayed inline with the code:

```bash
ipython notebook --pylab inline
```
