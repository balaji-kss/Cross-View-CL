LOGFILE=loggers/${1}.log

python3 trainDIR_CV_withCL.py > "$LOGFILE" 2>&1 &