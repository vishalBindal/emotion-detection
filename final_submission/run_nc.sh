#!/bin/bash

if [[ $1 == "1" ]]; then
    python3 q1_auto.py $2 $3 $4
elif [[ $1 == "2" ]]; then
    python3 q2_auto.py $2 $3 $4
elif [[ $1 == "3" ]]; then
    python3 q3_auto.py $2 $3 $4
fi

