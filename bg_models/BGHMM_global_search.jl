#JOB FILEPATHS
#sample record and hmm serialisation output path
sample_output = "/bench/PhD/NGS_binaries/BGHMM/BGHMM_samples"
survey_input = "/bench/PhD/NGS_binaries/BGHMM/survey_chains"
hmm_output = "/bench/PhD/NGS_binaries/BGHMM/global_chains"

#GENERAL SETUP
@info "Loading libraries..."
using Distributed, ProgressMeter, Serialization

#JOB CONSTANTS
#CONSTANTS FOR BGHMM LEARNING
const A = 4 #base alphabet size is 4 (DNA)
const replicates = 8 #repeat optimisation from this many seperately initialised samples from the prior
const no_models = replicates * 3
const input_hmms= RemoteChannel(()->Channel{Tuple}(no_models)) #channel to hold HMM learning jobs
const learnt_hmms= RemoteChannel(()->Channel{Tuple}(Inf)) #channel to take EM iterates off of
const delta_thresh=1e-5 #stopping/convergence criterion (log probability difference btw subsequent EM iterates)
const max_iterates=25000

#DISTRIBUTED CLUSTER CONSTANTS
#remote_machine = "10.0.0.3"
no_local_processes = 4
no_remote_processes = 0
#SETUP DISTRIBUTED BAUM WELCH LEARNERS
@info "Spawning workers..."
addprocs(no_local_processes, topology=:master_worker)
#addprocs([(remote_machine,no_remote_processes)], tunnel=true, topology=:master_worker)
pool_size = no_remote_processes + no_local_processes
worker_pool = [i for i in 2:pool_size+1]

@info "Loading worker libraries everywhere..."
@everywhere using BGHMM, DataFrames, Distributions, Random, CLHMM
@everywhere Random.seed!(2786)

#LOAD SAMPLES
@info "Loading samples from $sample_output..."
sample_dfs = deserialize(sample_output)

#BUILD TRAINING AND TEST SETS FROM SAMPLES
training_sets, test_sets = BGHMM.split_obs_sets(sample_dfs)

#INTIIALIZE HMMS
@info "Setting up HMMs..."
if isfile(hmm_output) #load chains from incomplete
    @info "Loading incomplete chains..."
    hmm_results_dict = deserialize(hmm_output)
else #otherwise, load survey results
    @info "Loading survey chains..."
    hmm_results_dict = deserialize(survey_input)
end

search_params = BGHMM.HMM_global_search_params(hmm_results_dict)
no_input_hmms = BGHMM.HMM_global_search_setup!(hmm_results_dict, search_params, replicates, delta_thresh, input_hmms, training_sets, A)

#SEND HMM FIT JOBS TO WORKERS
if isready(input_hmms) > 0
    @info "Fitting HMMs.."
    #WORKERS FIT HMMS
    for worker in worker_pool
        remote_do(linear_hmm_converger!, worker, input_hmms, learnt_hmms, no_models, delta_thresh=delta_thresh, max_iterations=max_iterates)
    end
else
    @warn "No input HMMs (all already converged?), skipping fitting.."
end

#GET LEARNT HMMS OFF REMOTECHANNEL, SERIALISE AT EVERY ITERATION, UPDATE PROGRESS METERS
job_counter=no_input_hmms
learning_meters=Dict{Tuple, BGHMM.ProgressHMM}()
overall_meter=Progress(no_input_hmms,"Overall batch progress:")

while job_counter > 0
    wait(learnt_hmms)
    workerid, jobid, iterate, hmm, log_p, delta, converged = take!(learnt_hmms)
    #either update an existing ProgressHMM meter or create a new one for the job
    if haskey(learning_meters, jobid) && iterate > 2
            BGHMM.update!(learning_meters[jobid], delta)
    else
        offset = workerid - 1
        if iterate <=2
            learning_meters[jobid] = BGHMM.ProgressHMM(delta_thresh, "Jobid $jobid on worker $workerid converging:", offset, 2)
        else
            learning_meters[jobid] = BGHMM.ProgressHMM(delta_thresh, "Jobid $jobid on worker $workerid converging:", offset, iterate)
        end
    end
    #push the hmm and related params to the results_dict
    push!(hmm_results_dict[jobid], [iterate, hmm, log_p, delta, converged])
    #decrement the job counter, update overall progress meter, and save the current results dict on convergence or max iterate
    if converged || iterate == max_iterates
        global job_counter -= 1
        ProgressMeter.update!(overall_meter, (no_input_hmms-job_counter))
        serialize(hmm_output, hmm_results_dict)
        if !isready(input_hmms) #if there are no more jobs to be learnt, retire the worker
            rmprocs(workerid)
        end
    end
end

#count converged & unconverged jobs, report results
converged_counter = 0
unconverged_counter = 0
for (k,v) in hmm_results_dict
    v[end][5] == true ? (global converged_counter += 1) : (global unconverged_counter += 1)
end

@info "BGHMM batch learning task complete, $converged_counter converged jobs, $unconverged_counter jobs failed to converge in $max_iterates iterates since job start."