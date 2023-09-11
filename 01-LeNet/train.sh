startTime=`date "+%F %T"`
CONFIG="LeNet-Sigmoid"
OUTOUT=output/${CONFIG}
if [ ! -d $OUTOUT ]; then
    mkdir -p $OUTOUT
fi
cp configs/${CONFIG}.yaml $OUTOUT
CONFIG=${OUTOUT}/${CONFIG}.yaml
LOG_FILE=${OUTOUT}/train.txt

# train
python train.py \
    --config_file $CONFIG \
    --save_path $OUTOUT \
    --evaluate

endTime=`date "+%F %T"`
startTimestamp=`date -d "$startTime" +%s`
endTimestamp=`date -d "$endTime" +%s`
deltaTimestamp=$[ $endTimestamp - $startTimestamp ]
deltaTime="$(($deltaTimestamp / 3600))h$((($deltaTimestamp % 3600) / 60))m"
echo "$startTime ---> $endTime" "Total: $deltaTime"
