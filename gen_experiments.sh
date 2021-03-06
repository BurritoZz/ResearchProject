#!/bin/bash
# This script will generate shell scripts to schedule multiple job steps in multiple jobs.
# This script does the following.
# * Detect some settings in the SLURM config
# * Determine abslute paths to binaries and model specifications
# * Iterate over directories and generate job step scripts to run binaries
#   with multiple models and options. Each experiment will be run inside memtime.
# * Randomize the order in which experiments are run.
# * Generate job shell scripts in parallel
# the directory layout this script uses is as follows.

# LAYOUT:
# +-- ~/
#      |-- bin/                                     // locally installed software
#      |    |-- memtime/bin/memtime
#      |    |-- ltsminPerf/
#      |    |    +-- bin/
#      |    |         |-- pnml2lts-mc
#      |    |         +-- pnml2lts-sym
#      |    +-- ltsminStat/
#      |         +-- bin/
#      |              |-- pnml2lts-mc
#      |              +-- pnml2lts-sym
#      +-- experiments/
#           |-- in/                                 // the models to experiment with
#           |    |-- ptAll/
#           |    |    |-- <a petri net definition>.pnml
#           |    |    +-- ...
#           |    +-- ptSelect/
#           |         |-- <a petri net definition>.pnml
#           |         +-- ...
#           |-- out/                                // root directory of experiment output
#           |    |-- 0/
#           |    |    |-- <experiment_output>.dve2C // results of a dve experiment
#           |    |    +-- ...
#           |    +-- 1/failed                       // file which contains failed experiments
#           +-- generated/
#           |    |-- jobs/                          // contains job scripts (sbatch command with srun commands)
#           |    |    |-- 0
#           |    |    +-- ...
#           |    |-- steps/                         // contains job step scripts; shell scripts to do an experiment
#           |    |    |-- step_<uuid>.sh
#           |    |    +-- ...
#           |    +-- shuffle.txt                    // file to randomize the order of job steps
#           |-- gen_experiments.sh                  // this file
#           +-- submit_jobs.sh                      // script to submit all jobs

# memtime options
# max time = 60*30 seconds
MT_MAXCPU=1800
# max virtual mem = 8000000 kB ~ 8000 MB ~ 8GB (approx 3.5 GB will be claimed by node and cache table)
MT_MAXMEM=8000000

# Parameters of ltsmin applied to all testcases
# Resource bounds imposed by SLURM
PARAM_SLURM='-N1 -n1 -c2 --time35:00'
# General parameters of ltsmin
PARAM_GENERAL='--when'
# BDD pack used
PARAM_BDDPACK='--vset=lddmc --lace-workers=1 --sylvan-sizes=26,26,26,26'
# Fixed variables during testing
PARAM_TEST='--save-sat-levels --next-union -rtg,bs,hf --sat-granularity=5'

# Additional options for ltsminStat; defines the testcases
DO_STAT_CASES=true
# Default testcase
POPTS_VERBOSE=$'--order=chain --peak-nodes --graph-metrics\n'
POPTS_VERBOSE+=$'--order=chain-prev --peak-nodes --graph-metrics\n'
POPTS_VERBOSE+=$'--order=bfs --peak-nodes --graph-metrics\n'
POPTS_VERBOSE+=$'--order=bfs-prev --peak-nodes --graph-metrics'

# Additional options for ltsminPerf; defines the testcases
POPTS=$'--order=chain --saturation=none --graph-metrics\n'
POPTS+=$'--order=chain --saturation=sat-like --graph-metrics\n'
POPTS+=$'--order=chain --saturation=sat-loop --graph-metrics\n'
POPTS+=$'--order=chain --saturation=sat-fix --graph-metrics\n'
POPTS+=$'--order=chain --saturation=sat --graph-metrics\n'
POPTS+=$'--order=chain-prev --saturation=none --graph-metrics\n'
POPTS+=$'--order=chain-prev --saturation=sat-like --graph-metrics\n'
POPTS+=$'--order=chain-prev --saturation=sat-loop --graph-metrics\n'
POPTS+=$'--order=chain-prev --saturation=sat-fix --graph-metrics\n'
POPTS+=$'--order=chain-prev --saturation=sat --graph-metrics\n'
POPTS+=$'--order=bfs --saturation=none --graph-metrics\n'
POPTS+=$'--order=bfs --saturation=sat-like --graph-metrics\n'
POPTS+=$'--order=bfs --saturation=sat-loop --graph-metrics\n'
POPTS+=$'--order=bfs --saturation=sat-fix --graph-metrics\n'
POPTS+=$'--order=bfs --saturation=sat --graph-metrics\n'
POPTS+=$'--order=bfs-prev --saturation=none --graph-metrics\n'
POPTS+=$'--order=bfs-prev --saturation=sat-like --graph-metrics\n'
POPTS+=$'--order=bfs-prev --saturation=sat-loop --graph-metrics\n'
POPTS+=$'--order=bfs-prev --saturation=sat-fix --graph-metrics\n'
POPTS+=$'--order=bfs-prev --saturation=sat --graph-metrics'

# Absolute path to scontrol
sc=`which scontrol`

STEPS_PER_JOB=342
MAX_JOBS=1000

echo "Maximum number of steps per job is '$STEPS_PER_JOB'"
echo "Maximum number of jobs is '$MAX_JOBS'"

# A commandline option to generate job steps
OPT_GEN_STEPS="--gen-steps"

# Get the absolute path to the home directory
HOME=$(eval echo ~)
USER="<DEFINE USER>" # Name of the user posting jobs to cluster

# General path to executables
BIN="$HOME/bin"

# Path to performance and stats versions of LTSMin
LTSMIN_PERF_DIR="$BIN/ltsminPerf/bin"
LTSMIN_STAT_DIR="$BIN/ltsminStat/bin"

# Paths for LD_LIBRARY_PATH
# mCRL2 201409.1
MCRL2_2014091="$HOME/.local/bin/mrcl2" # TODO Is this needed?

# Memtime
MEMTIME="$BIN/memtime"

# Working directory
WDIR="$HOME/experiments"

# Paths to experiments
EXP="$WDIR/in"

# Echo to std err
echoerr() { echo "$@" 1>&2; }

# Path to experiment results
# Successful results
RES="$WDIR/out/0"
# Failed results
FAILED="$WDIR/out/1"

# Directory where the generated files are stored
G_DIR="$WDIR/generated"

# Directory with shell scripts for jobs
BATCH_DIR=$G_DIR/jobs
# Directory with shell scripts for job steps
STEP_DIR=$G_DIR/steps
# Executable to submit jobs
RUN_JOBS=$WDIR/submit_jobs.sh
# A file which contains all job steps to execute
# so that we can randomise the order of execution
SHUFFLE=$G_DIR/shuffle.txt
# A file to which std out and std err will be written to
OUT=$G_DIR/slurm_out.log

## Generate job step
## 1: command to execute
## 2: file to write results to
## 3: unique id of this job step
gen_job_step() {

	# Location to the shell script of this job
	script="$STEP_DIR/step_$3.sh"

	# Writes some commands to the shell script.
	# The shell script will first look in a file named 'failed'
	# to see if the experiment has been executed before in a previous iteration
	# in another job step
	echo "#!/bin/bash" > $script
	# echo "match=\$(cat \"$FAILED/failed\" | grep -o \"$1\")">>$script
	# echo "if [[ -z \$match ]]; then" >> $script
	cat "$STEP_DIR/env_$3" >> $script
	rm "$STEP_DIR/env_$3"
	echo " $1 > $RES/$2 2>&1" >> $script
	echo " if [ \$? -ne 0 ]; then" >> $script
	echo "  echo \"$1\" >> $FAILED/failed" >> $script
	echo " fi" >> $script
	# echo "fi" >> $script
	chmod u+x $script

	# Add to the job step the 'shuffle' file.
	echo "srun $PARAM_SLURM --partition=$PARTITION $script $" >> $SHUFFLE
}

# Generate job steps of specifications in a directory
# 1: array of models
# 2: options
# 3: command
# 4: LD_LIBRARY_PATH
# 5: iteration #
gen_job_steps() {

	# Loop over the directory
	for m in $(find -L "$FEXP" -type f); do

		while read -r p; do

			# Since we generate job steps in parallel
			# we want to have a unique id with "uuidgen"
			uuid=`uuidgen`

			# We have installed everything locally in our home directory,
			# so we want to configure environment variables.
			envir="$STEP_DIR/env_$uuid"
			echo " export LD_LIBRARY_PATH=$4" > $envir

			# ???
			o="$p"

			# The command the job step will execute
			c="$MEMTIME -m$MT_MAXMEM -c$MT_MAXCPU $3 $PARAM_GENERAL $PARAM_BDDPACK $PARAM_TEST $o $m"

			# The base-name of the command to execute
			n=$(basename "$3")

			# The base-name of the model
			f=$(basename "$m")

			# Make sure the options do not have characters invalid for a filename
			o=$(echo $p | tr " " "-")

			# The file to write results to
			r="$5""_""$v""_""$n""_""$o""_""$f"
			gen_job_step "$c" "$r" "$uuid"
		done <<< "$2"
	done
}

# Usage information
if [[ -z "$1" || -z "$2" || -z "$3" || -z "$4" || -z "$5" ]]; then
	echoerr "Usage $0 repeat dir part bin nodes [$OPT_GEN_STEPS]"
	exit 1
fi

# Full path to experiments
FEXP="$EXP/$2"

PARTITION="$3"

BINARY="$LTSMIN_PERF_DIR/$4"
BINARY_STATS="$LTSMIN_STAT_DIR/$4"
NODES="$5"

# Delete old job scripts
rm -r "$BATCH_DIR"
mkdir "$BATCH_DIR"

# If --gen-steps is supplied, we will generate job steps, else we will only generate the jobs
if [[ -n "$6" && $6=="$OPT_GEN_STEPS" ]]; then

	if [ ! -d "$FEXP" ]; then
		>&2 echo "$FEXP is not a valid directory"
		exit 1
	fi

	sinfo -p $PARTITION | grep "up" > /dev/null
	if [[ $? == 1 ]]; then
		>&2 echo "Partition $PARTITION does not exist or is down"
		exit 1
	fi

	if [ ! -f "$BINARY" ]; then
		>&2 echo "Executable $BINARY is not available"
		exit 1
	fi

	if [ ! -f "$BINARY_STATS" ]; then
		>&2 echo "Executable $BINARY_STATS is not available"
		exit 1
	fi

	echo "Number of instances per cestcase is $1"
	echo "Input model directory is $FEXP"
	echo "Partition for the experiments is $PARTITION"
	echo "Nodes used for each job is $NODES"
	echo "Binaries are $BINARY and $BINARY_STATS"
	echo "Generating steps, please wait..."

	# Delete old job steps
	rm -r "$STEP_DIR"
	mkdir "$STEP_DIR"
	rm "$SHUFFLE"

	# This for loop will repeat the same experiment $1 times
	for i in `seq 1 $1` do

		# Generating job steps can take a long time, so we will run each call to gen_job_steps in parallel
		gen_job_steps "$PNML" "$POPTS" "$BINARY" '""' "$i" &

	done

	if $DO_STAT_CASES; then
		gen_job_steps "$PNML" "$POPTS_VERBOSE" "$BINARY_STATS" '""' "0" &
	fi
	wait

	# Shuffle our job step scripts so that they will be executed in random order
	count=`cat $SHUFFLE | wc -l`
	echo "shuffling steps"
	cat $SHUFFLE | shuf > $SHUFFLE.tmp
	mv $SHUFFLE.tmp $SHUFFLE
fi

if [ ! $SHUFFLE ]; then # Make sure the shuffle exists.
	echoerr "$SHUFFLE does not exist, generate this file first with the '$OPT_GEN_STEPS' option"
	exit 1
fi

# Count the number of job steps generated
let step_count=`cat $SHUFFLE | wc -l`
if [[ $step_count -ge $(($STEPS_PER_JOB * $MAX_JOBS)) ]]; then
	echoerr "Too many job steps ($step_no; $(($STEPS_PER_JOB * $MAX_JOBS)))); can only schedule '$MAX_JOBS' jobs and '$STEPS_PER_JOB' steps per job."
	exit 1
fi

# Ceil the amount of jobs necessary
job_count=$(($step_count/$STEPS_PER_JOB))
job_count=$(( `echo $job_count|cut -f1 -d"."` + 1 ))

# Generated the script which can schedule all jobs.
# The script will detect two things;
# * Whether there is enough room in the queue to schedule all jobs.
# * Whether we won't overwrite the old experiments.
echo "#!/bin/bash" > $RUN_JOBS
echo -e "echoerr() { echo \"\$@\" 1>&2; }\n" >> $RUN_JOBS
echo "# make sure we can submit '$JOB_COUNT' batch(es)" >> $RUN_JOBS
echo "queue_len=\$((\`squeue | wc -l\` -1))" >> $RUN_JOBS
echo "if [ \$((\$queue_len+$job_count)) -gt $MAX_JOBS ]; then" >> $RUN_JOBS
echo "	echoerr \"Can not submit jobs. Currently you can only submit '\$(($MAX_JOBS - \$queue_len))' jobs.\"" >> $RUN_JOBS
echo "	exit 1" >> $RUN_JOBS
echo -e "fi \n" >> $RUN_JOBS

echo "# Make sure we do not overwrite older experiments" >> $RUN_JOBS
echo "if [ \"\$(ls -A $RES)\" ]; then" >> $RUN_JOBS
echo "	echoerr \"Directory $RES is not empty\"" >> $RUN_JOBS
echo "	exit 1" >> $RUN_JOBS
echo "fi" >> $RUN_JOBS
echo "/bin/rm $FAILED/failed" >> $RUN_JOBS
echo "touch $FAILED/failed" >> $RUN_JOBS
echo "/bin/rm $OUT" >> $RUN_JOBS

echo "'$step_count' steps to submit"
echo "'$job_count' job(s) needed"

# Generate each job shell script
for i in `seq 0 $(($job_count-1))`; do

	current_job=$BATCH_DIR/$i

	#echo "read -n1 -r -p \"Press any key to sumbit next batch ($i)...\" key" >> $RUN_JOBS
	echo "${current_jobs}_jobs" >> $RUN_JOBS
	if [ $i -lt $(($job_count-1)) ]; then
		echo 'myjobs=1000' >> $RUN_JOBS
		echo 'otherjobs=1000' >> $RUN_JOBS
		echo 'while [ $myjobs -gt 608 -o $otherjobs -gt 5000 ]; do' >> $RUN_JOBS
		echo '	myjobs=$(squeue -h -p'"$PARTITION"' -u'"$USER"' | wc -l)' >> $RUN_JOBS
		echo '	otherjobs=$(squeue -h | wc -l)' >> $RUN_JOBS
		echo '	sleep 10' >> $RUN_JOBS
		echo 'done' >> $RUN_JOBS
	fi

	# We sleep one minute after scheduling each job
	# to relax the SLURM control daemon
	if [ $i -lt $(($job_count-1)) ]; then
		echo "sleep 10" >> $RUN_JOBS
	fi

	# Add options for the 'sbatch' command to the job
	# The --cpus-per-task option makes sure the job step will be executed
	# exclusively on one node.
	echo "#!/bin/bash" > ${current_job}_batch
	echo "#SBATCH --partition=$PARTITION -N$NODES --output=$OUT --open-mode=append" >> ${current_job}_batch
	echo "${current_job}_jobs" >> ${current_job}_batch

	# Determine the lines to take from the randomised 'shuffle' file
	min=$(($i*$STEPS_PER_JOB+1))
	max=$(($(($i+1))*$STEPS_PER_JOB))
	if [ $max -gt $step_count ]; then
		let max=$step_count
	fi
	echo "#!/bin/bash" > ${current_job}_jobs
	echo "#steps $min - $(($max))" >> ${current_job}_jobs
	sedargs="${min},${max}p;${max}q"
	sed -n "$sedargs" $SHUFFLE >> ${current_job}_jobs

	# echo "wait" >> ${current_job}_jobs
	chmod +x ${current_job}_batch
	chmod +x ${current_job}_jobs
done
chmod +x $RUN_JOBS
echo "Done generating"
echo "You can now run '$RUN_JOBS'"
