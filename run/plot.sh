# !/bin/bash
# Set to true to only echo command instead of running it
build_only=false

# Set path to root directory containing exps/ directory
result_root_dir=$(pwd)

# Set task to true if you want plot config file to be ran
catcher=false
flappy=false
room=false

# Add or uncomment plots configs you want to generate
configs=(
# main
# qreg-ablation
# qreg-lambda
# buffer
# rel-update-freq
)

# Command template
command='generate_plots.py 
+experiment=${task}/${config} 
++default_root_dir=${result_root_dir}'

# Run plot configs 
for config in ${configs[@]}
do
    if ${catcher} ; then
        task=ple/catcher
        
        if ${build_only}; then
            eval echo ${command}
        else
            eval python ${command}
        fi
    fi

    if ${flappy} ; then
        task=ple/flappy

        if ${build_only}; then
            eval echo ${command}
        else
            eval python ${command}
        fi
    fi

    if ${room} ; then
        task=minihack/room

        if ${build_only}; then
            eval echo ${command}
        else
            eval python ${command}
        fi
    fi
done