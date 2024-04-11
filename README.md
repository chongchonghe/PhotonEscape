# Ionizing photon escape calculation

## How to use this module?

1. Install pymses. The following script works only on macOS because it uses brew to install hdf5 but brew is only availble on macOS. For linux clusters, you need to `module load` hdf5. 

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

echo "Add the following to your .bashrc/.zshrc or to your project setup script"
echo ""
echo "export PYTHONPATH=\$PYTHONPATH:$PYMSESDIR/pymses_python3"
```

2. Clone this repo

3. Copy (or symbolic link) the snapshot to `data/cluster/output_00273`

4. Run the test in this folder: `python test.py`

5. To use this module in your script. Then, add the `./src` folder to `PYTHONPATH` by `export PYTHONPATH=$PYTHONPATH:PATH-TO-THIS-REPO/src`. Then, you can write script like in test.py. Or, use the script I wrote, `loop_fred_sims.py`, to fast-process Fred's simulation. Run `python loop_fred_sims.py -h` for help message. The general routine is to do `python loop_fred_sims.py process path-to-sim-data` followed by `python loop_fred_sims.py fesc path-to-sim-data`. You can copy loop_fred_sims.py to your folder and do your own tweaking. 

6. Note: you need to copy pymses_field_descrs.py to your working folder in order to let Pymses read the right fields. 

## About the fields

In my code, I assumed rho(xHI) + rho(xHII) = rho

## Things to check

- Check the ionization cross sections in the beginning of fesc.py. I might be wrong. 
