startTime=`date "+%F %T"`
CONFIG="LeNet-Sigmoid"
OUTOUT=output/${CONFIG}
CONFIG=${OUTOUT}/${CONFIG}.yaml
MODEL=${OUTOUT}/latest.pdparams

# train
python test.py \
    --config_file $CONFIG \
    --model_file $MODEL

endTime=`date "+%F %T"`
startTimestamp=`date -d "$startTime" +%s`
endTimestamp=`date -d "$endTime" +%s`
deltaTimestamp=$[ $endTimestamp - $startTimestamp ]
deltaTime="$(($deltaTimestamp / 3600))h$((($deltaTimestamp % 3600) / 60))m"
echo "$startTime ---> $endTime" "Total: $deltaTime"
