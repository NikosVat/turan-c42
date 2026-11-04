#!/bin/bash
cd /home/user/turan-c42
JOBS=2 checkers/run_queue.sh logs/S5/q4a.txt
JOBS=1 checkers/run_queue.sh logs/S5/q4b.txt
JOBS=2 checkers/run_queue.sh logs/S5/q4c.txt
echo CHAIN4_DONE
