![c2sm logo](./figures/c2sm.png)

# pyvis - Python Visualisation Course - c2sm


* Mathias Hauser ([ETHZ](http://www.iac.ethz.ch/people-iac/person-detail.html?persid=146568), [github](https://github.com/mathause)) <mathias.hauser@env.ethz.ch>
* Tarun Chadha  ([ETHZ](https://www.ethz.ch/services/en/organisation/departments/it-services/people/person-detail.html?persid=166149), [github](https://github.com/chadhat)) <tarun.chadha@id.ethz.ch>


## Format

The lectures consist of IPython Notebooks.

## Audience

Python novices who should have programming experience in other
languages. A number of examples and problems are drawn from the field
of Climate Reseach.

## Duration

It's envisioned as a two day course.


## How to use it

### Obtain the course material

Clone the git repository:

~~~~bash
git clone https://github.com/C2SM/pyvis.git
~~~~

Alternatively, you can download the [zip-archive](https://github.com/C2SM/pyvis/archive/master.zip)
and unpack.


### Start jupyter notebook

#### Computer room at ETH (HG D 12)

 * log in to Fedora (you may have to reboot the computer)
 * execute the following commands:

~~~~bash
export CONDA_ENVS_PATH=/opt/kunden/hauser/conda/envs
source activate pyvis
# go to the directory of the material
jupyter notebook
~~~~


#### At IAC ETH (on linux computers)

 * execute the following commands:

~~~~bash
module load conda
source activate pyvis
# go to the directory of the material
jupyter notebook
~~~~

See also the [IAC wiki on conda](https://wiki.iac.ethz.ch/bin/viewauth/IT/CondaPython) (restricted access).

#### On your personal computer

 * [Download](https://conda.io/docs/user-guide/install/download.html) and [install conda](https://conda.io/docs/user-guide/install/linux.html)
 * Create the `pyvis` environment, using the [`pyvis.yml`](https://github.com/C2SM/pyvis/blob/master/pyvis.yml) file (this will take a while)

~~~~bash
conda env create -f pyvis.yml
~~~~

 * open jupyter notebook 

~~~~bash

source activate pyvis
# go to the directory of the material
jupyter notebook
~~~~

## What's New

 * See [version history](./WHATS_NEW.md)


## Feedback

I would be very happy to hear from you (mail to <mathias.hauser@env.ethz.ch>).

Particularly welcome are problem reports, errors, criticism.

# License

Copyright (C) 2018 C2SM / Mathias Hauser / Tarun Chadha

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see [http://www.gnu.org/licenses/].
