# Ionizing photon escape calculation

## TODOs

- Use a precise model for ionising photon cross section. References: Rosdahl+2013, appendix E4

## How to use this module?

1. Install pymses. The following script works only on macOS because it uses brew to install hdf5 but brew is only availble on macOS. For linux clusters, you need to `module load` hdf5. 

On Zaratan: 

After each 'pip install', you may see some dependency errors, but as long as you see `Successfully installed ... PACKAGE-x.x.x`, it means the package is successfully installed and you can ignore the warnings/errors. 

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
5. You can use this module in your own python script. See examples in test.py. Or, use the script I wrote, `loop_fred_sims.py`, to fast-process Fred's simulation. First a test run, you can run the following commands in this folder. For production run, copy `loop_fred_sims.py` and `pymses_field_descrs.py` to your working directory (preferably on scratch because the code will generate lots of data). Then, run `python loop_fred_sims.py process path-to-sim-data` followed by `python loop_fred_sims.py fesc path-to-sim-data`. For available command line options, do `python loop_fred_sims.py -h`. 

## About the fields

In my code, I assumed rho(xHI) + rho(xHII) = rho

## Things to check

- Check the ionization cross sections in the beginning of fesc.py. I might be wrong. 
