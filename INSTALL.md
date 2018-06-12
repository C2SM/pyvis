# Install conda and the environment

## Install conda

 * Follow the [conda installation guide](https://conda.io/docs/user-guide/install/) for your operating system.
   * Choose Anaconda

## Install the environment

 * Download the `packages.yml` file 
   * Go to [`packages.yml`](https://github.com/C2SM/pyvis/blob/master/packages.yml)
   * right click on the `Raw` button
   * Save File As...
 *  and open jupyter notebook

### Install the environment - from the command line

~~~~bash
conda env create -f packages.yml
~~~~

### Install the environment - from Anaconda Navigator (Windows)

 * Open Anaconda Navigator
 * Go to `Environments`
 * Select Import
   * Name: pyvis
   * Specification File: Select the `packages.yml` file
   * This will take about 20 minutes.
 * Go to `Home`
 * Switch to pyvis environment; Select `Applications on 'pyvis'`


