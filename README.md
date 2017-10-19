# CNN Model for System Identification of Neural Types
This is a code repository to the paper (cite as):

Klindt, D., Ecker, A., Euler, T. & Bethge, M. (2017). Neural system identification for large populations separating “what” and “where”. In Advances in Neural Information Processing Systems.

[Arxiv]

## Requirements:
- Tensorflow (with GPU support, see https://www.tensorflow.org/install/install_linux)

# Instructions
To reproduce the figures from the paper (see above) open the corresponding notebooks:

### For Figure 3 and 4 go to

`fig{3,4}.ipynb`

and execute the cells with further instructions provided in the comments.

### For Figure 5b-d execute

`fig5/fig5.ipynb`

as well as

`fig5/CNN_{Batty,McInt}.ipynb`

where 'Batty' is the CNN model with fixed location mask and 'McInt' the CNN model with fully connected readout.

### For Figure 5e execute

`fig5/more_types/{fig5,Batty,Mcint}_more_types.ipynb`

### For Table 1 see

Folder `v1data`

The results of the grid search are stored in a database using the data management toolkit
[DataJoint](http://datajoint.io). If you intend to actually run the code yourself there
will be additional work needed setting up a MySQL server and installing DataJoint. We're
happy to help with that.

If your goal is to just use the code to fit a model to your own data, consult
`standalone.py` for a working example.

If you want to check the code we used: `convnet.py` defines the neural networks and does
the heavy lifting; `database.py` contains the database classes and exact parameter settings
that we used (`Fit._make_tuples()` is a good starting point).

