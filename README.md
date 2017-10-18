# CNN Model for System Identification of Neural Types
This is a code repository to the paper (cite as):

Klindt, D., Ecker, A., Euler, T. & Bethge, M. (2017). Neural system identification for large populations separating “what” and “where”. In Advances in Neural Information Processing Systems.

[Arxiv]

## Requirements:
- Tensorflow (with GPU support, see https://www.tensorflow.org/install/install_linux)

# Instructions
To reproduce the figures from the paper (see above) open the corresponding notebooks:

### for figure 3 and 4 go to

fig{3,4}.ipynb

and execute the cells with further instructions provided in the comments.

### for figure 5b-d execute

./fig5/fig5.ipynb

as well as

./fig5/CNN_{Batty,McInt}.ipynb

where 'Batty' is the CNN model with fixed location mask and 'McInt' the CNN model with fully connected readout.

### for figure 5e execute

./fig5/more_types/{fig5,Batty,Mcint}_more_types.ipynb

### for table1/figure6

[Alex?]
