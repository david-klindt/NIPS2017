#!/usr/bin/python3

import numpy as np
import datajoint as dj
from database import Fit, FitFC, FitFixedMask
import tensorflow as tf

Fit().populate(reserve_jobs=True)
FitFC().populate(reserve_jobs=True)
FitFixedMask().populate(reserve_jobs=True)

