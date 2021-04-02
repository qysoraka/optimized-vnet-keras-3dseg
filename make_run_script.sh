
bs=1; is=128; lr=0.01; gs=8; fr=16; op=adam
datadir=orig_data
logdir=eval_log

script=run_vnet3d_with_ag.py
ct=AG_Aug1_B${bs}_I${is}_L${lr}_G${gs}_fs${fr}_${op}

cmd="python3 $script"
cmd="$cmd -ct $ct"
cmd="$cmd -I $datadir"
cmd="$cmd -bs $bs"
cmd="$cmd -is $is"
cmd="$cmd -lr $lr"
cmd="$cmd -gs $gs"
cmd="$cmd -fr $fr"