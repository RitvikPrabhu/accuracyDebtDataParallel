#!/bin/bash

nEvents=10000000
logDir="/Users/daniellersch/Desktop/SciDAC/sample_data/dalitz_v0 "
maskLoc="dalitz_ortho"

CMD=" create_dalitz_data.py"
CMD=$CMD" n_events=$nEvents logdir=$logDir mask_loc=$maskLoc device=mps"


python $CMD
rm -r outputs