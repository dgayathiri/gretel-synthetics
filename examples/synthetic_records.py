#!/usr/bin/env python
# coding: utf-8

# # Generating synthetic data
# 
# This notebook walks through training a probabilistic, generative RNN model<br>
# on a rental scooter location dataset, and then generating a synthetic<br>
# dataset with greater privacy guarantees. 
# 
# For both training and generating data, we can use the ``config.py`` module and<br>
# create a ``LocalConfig`` instance that contains all the attributes that we need<br>
# for both activities.

# In[ ]:


# Google Colab support
# Note: Click "Runtime->Change Runtime Type" set Hardware Accelerator to "GPU"
# Note: Use pip install gretel-synthetics[tf] to install tensorflow if necessary
# 
#!pip install gretel-synthetics --upgrade


# In[ ]:


from pathlib import Path

from gretel_synthetics.config import LocalConfig

import tensorflow as tf
orig_jacobian = tf.GradientTape.jacobian
def patched_jacobian(self, *args, **kwargs):
    kwargs['experimental_use_pfor'] = False
    return orig_jacobian(self, *args, **kwargs)

tf.GradientTape.jacobian = patched_jacobian

# Create a config that we can use for both training and generating data
# The default values for ``max_lines`` and ``epochs`` are optimized for training on a GPU.

config = LocalConfig(
    max_lines=0,         # maximum lines of training data. Set to ``0`` to train on entire file
    max_line_len=2048,   # the max line length for input training data
    epochs=10,           # 15-50 epochs with GPU for best performance
    vocab_size=20000,    # tokenizer model vocabulary size
    gen_lines=1000,      # the number of generated text lines
    dp=True,             # train with differential privacy enabled (privacy assurances, but reduced accuracy)
    field_delimiter=",", # specify if the training text is structured, else ``None``
    dp_noise_multiplier=0.005,
    dp_l2_norm_clip=0.5,
    dp_microbatches=1,
    # embedding_dim=512,
    overwrite=True,      # overwrite previously trained model checkpoints
    checkpoint_dir=(Path.cwd() / 'checkpoints').as_posix(),
    input_data_path="https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/uber_scooter_rides_1day.csv" # filepath or S3
)


# In[ ]:


# Train a model
# The training function only requires our config as a single arg
from gretel_synthetics.train import train_rnn

train_rnn(config)


# In[ ]:


# Let's generate some text!
#
# The ``generate_text`` funtion is a generator that will return
# a line of predicted text based on the ``gen_lines`` setting in your
# config.
#
# There is no limit on the line length as with proper training, your model
# should learn where newlines generally occur. However, if you want to
# specify a maximum char len for each line, you may set the ``gen_chars``
# attribute in your config object
from gretel_synthetics.generate import generate_text

# Optionally, when generating text, you can provide a callable that takes the 
# generated line as a single arg. If this function raises any errors, the 
# line will fail validation and will not be returned.  The exception message
# will be provided as a ``explain`` field in the resulting dict that gets
# created by ``generate_text``
def validate_record(line):
    rec = line.split(", ")
    if len(rec) == 6:
        float(rec[5])
        float(rec[4])
        float(rec[3])
        float(rec[2])
        int(rec[0])
    else:
        print(f"REJECTED: {line}")
        raise Exception('record not 6 parts')
        
for line in generate_text(config, line_validator=validate_record):
    print(line)

