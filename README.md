# Ionizing photon escape calculation from dwarf galaxies

## TODOs

- Use a precise model for ionising photon cross section. References: Rosdahl+2013, appendix E4

## How to use this module?

### First time

Follow these steps to install this module on your computer.

1. Install pymses.

On Zaratan:

Follow these steps to install pymses for python3. After each 'pip install', you may see some dependency errors, but as long as you see `Successfully installed ... PACKAGE-x.x.x`, the package is successfully installed and you can ignore the warnings/errors.

```sh
PYMSESDIR=~/softwares/pymses

module load hdf5
module load python/3.8.12
# unset PYTHONPATH so that the system-wide python packages won't be seen
export PYTHONPATH=
mkdir -p ${PYMSESDIR}/python
cd ${PYMSESDIR}/python
python -m venv pymses
source pymses/bin/activate
# ensure this returns the correct python path: ./pymses/bin/python
which python
python -m pip install --upgrade pip
# Install yt
python -m pip install yt
# Check yt installation: this shouldn't return error if yt is properly installed
python -c 'import yt'
# Install other required packages
python -m pip install numpy scipy matplotlib ipython h5py tables astropy healpy nose
# Or, if there is dependency conflict, install the following versions of astropy and healpy
# pip install astropy==5.2.2 healpy==1.16.5

# Clone and install pymses
cd $PYMSESDIR
git clone https://github.com/chongchonghe/pymses_python3.git
cd pymses_python3
make cython
# You should see zero error messages from previous command, otherwise it's a failure. If make is successful, the last line of the printing should be something like 'make[1]: Leaving directory '/home/che1234/softwares/pymses/pymses_python3'
make
export PYTHONPATH=$PYMSESDIR/pymses_python3
# You should add the previous command into your .bashrc or your project setup script
```

On macOS:

```sh
# Set the directory where pymses will be installed
PYMSESDIR=~/softwares/pymses

# Check if brew or conda are not installed
brew --version
conda --version

# Install hdf5
brew install hdf5

# Create a new conda environment
conda create -n pymses python==3.8.11
conda activate pymses
conda install numpy scipy matplotlib ipython anaconda::hdf5 -y
conda install pytables -y
conda install anaconda::cython -y
python -m pip install astropy healpy 
# Or, if there is dependency conflict, install the following versions
# python -m pip install astropy==5.2.2 healpy==1.16.5
python -m pip install nose

# Clone and install pymses
mkdir -p $PYMSESDIR
cd $PYMSESDIR
git clone https://github.com/chongchonghe/pymses_python3.git
cd pymses_python3
make cython
make
export PYTHONPATH=$PYTHONPATH:$PYMSESDIR/pymses_python3

# install yt
conda install --channel conda-forge yt

echo "Add the following to your .bashrc/.zshrc or to your project setup script"
echo ""
echo "export PYTHONPATH=\$PYTHONPATH:$PYMSESDIR/pymses_python3"
```

2. Clone this repo into your project directory: `git clone https://github.com/chongchonghe/PhotonEscape.git`
3. Do `export PYTHONPATH=$PYTHONPATH:PATH-TO-THIS-REPO/src`.
4. Run the test in this folder: `python test.py` . You should see `Test passed.` in the end.
5. Use the script `loop_fred_sims.py` to post-process Fred's simulation:

```bash
#python loop_fred_sims.py -h
python loop_fred_sims.py process path-to-sim-data --subsample 0.01 --dist 1_kpc --refine 2
python loop_fred_sims.py fesc path-to-sim-data --outdir outs/
```

See examples in run.sh

### Second time and onward

Run the following configuration on your bash shell:

```bash
# activate python environment
source ~/softwares/pymses/python/pymses/bin/activate
# set PYTHONPATH
export PYTHONPATH=path-to-pymses_python3:path-to-this-repo/src
```

Then, you can continue to post-process Fred's simulations.

## Physics

- About the fields
  - In the code, I assumed xHeI + xHeII + xHeIII = 1.0 and xH2 + xHI + xHII = 1.0, and the mass fraction of helium is 0.24.
  - A star stops radiating when it is older than 30 Myr.
