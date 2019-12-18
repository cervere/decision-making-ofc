# decision-making-ofc
Computational model of value-based decision making in the Orbitofrontal Cortex


### Some standard dependencies
```
sudo apt-get install python-numpy python-matplotlib ipython ipython-notebook
pip install Cython
```

### In the project folder
```
python model_setup.py build_ext --inplace ; mv model*cpython*.so model.so
```

### Sample run - no learning, random trials, plots showing sample 3 trial activities
