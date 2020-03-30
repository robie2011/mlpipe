WAIT=10

SCRIPT=$(realpath "$0")
workdir=$(dirname "$SCRIPT")
workdir=$(dirname "$workdir")
export PYTHONPATH=$workdir
PATH_DSL_INSTANCE_INTEGRATION=${1:-mlp_simple.integrate.yml}
PATH_LIVE_CSV=$(python -m mlpipe.admin.yaml_config_reader .live_monitoring csv_file_out)
PATH_PREDICTION=$(python -m mlpipe.admin.yaml_config_reader $PATH_DSL_INSTANCE_INTEGRATION integrate.output.outputPath)

while [ True ]; do
    result=$(python -m mlpipe.admin.live_compare_csv $PATH_LIVE_CSV $PATH_PREDICTION)
    clear
    echo "Comparision Visualizer API Data vs. Model Prediction"
    echo "  refresh interval: ${WAIT} (seconds)"
    echo "  last refresh: `date`"
    echo "  csv file live: $PATH_LIVE_CSV"
    echo "  csv file prediction: $PATH_PREDICTION"
    echo "----------------------------------------------------"
    echo "$result"
    echo
    sleep $WAIT
done
