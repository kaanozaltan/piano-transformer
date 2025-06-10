#!/bin/bash

groups
ls -l /hpcwork/lect0148/data/maestro/maestro-v3.0.0.csv
python models/mistral-162M_remi_maestro/train.py
