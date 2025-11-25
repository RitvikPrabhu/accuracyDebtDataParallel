#!/bin/bash

# This script lets you run the workflow on your local machine. 

# Basic settings: 
mainDir="/Users/daniellersch/Desktop/SciDAC/multi_datasets/quantom-ips/src/quantom_ips"
driver=$mainDir"/drivers/distributed_multi_data_training_workflow.py"
dataLoc="/Users/daniellersch/Desktop/SciDAC/sample_data/proxy2d_app_4datasets"

CMD="environment.parser.paths=[[$dataLoc/events_2dproxy_data0.npy],[$dataLoc/events_2dproxy_data1.npy],[$dataLoc/events_2dproxy_data2.npy],[$dataLoc/events_2dproxy_data3.npy]]"
CMD=$CMD" environment.parser.n_randomly_chosen_sets=1 environment.parser.chosen_data_set_idx=[] environment.parser.use_deterministic_data_idx=False"
CMD=$CMD" environment/theory=proxy2d_v2"
CMD=$CMD" environment.theory.ratios=[0.78,0.65,0.87,0.38] environment.theory.a_min=[0.0,0.0,0.0] environment.theory.a_max=[0.1,0.15,0.01]"

CMD=$CMD" environment.n_samples=1000 optimizer.layers.Upsample.config.size=[50,50]"
CMD=$CMD" optimizer.layers.ConvTranspose2dStack.config.out_channels=3"
CMD=$CMD" trainer.batch_size=300 environment.objective.normalize_inputs=False"
CMD=$CMD" trainer.n_epochs=500 analysis.frequency=10 analysis.gradient_snapshot_frequency=50 analysis.model_snapshot_frequency=50 analysis.printout_frequency=100"
CMD=$CMD" hydra.run.dir=multi_dataset_local_test_v2 environment.use_auto_scaling=False"
CMD=$CMD" trainer.distribute_disc=False trainer.outer_update_epochs=20"
CMD=$CMD" environment.objective.use_log=False"
CMD=$CMD" trainer.gradient_transport.pcgrad_threshold=0.0 trainer.gradient_transport.loss_ema_beta=0.9"

mpirun -n 6 python $driver $CMD