#JOB FILEPATHS
hmm_output = "/bench/PhD/NGS_binaries/BBM/survey_chains"
refined_path = "/bench/PhD/NGS_binaries/BBM/refined_chains"
sample_output = "/bench/PhD/NGS_binaries/BBM/refinement_samples"

survey_folders = "/bench/PhD/NGS_binaries/BBM/survey_folders"

#GENERAL SETUP
@info "Loading libraries..."
using BioBackgroundModels, DataFrames, Distributed, Serialization

const delta_thresh=1e-4 #stopping/convergence criterion (log probability difference btw subsequent EM iterates)
const max_iterates=15000

#LOAD SAMPLES
@info "Loading samples from $sample_output..."
sample_dfs = deserialize(sample_output)

#BUILD TRAINING AND TEST SETS FROM SAMPLES
training_sets, test_sets = split_obs_sets(sample_dfs)

#GENERATE Chain_ID Vector AND CHAINS SUBSET
@info "Loading reports from $survey_folders"
report_folders=deserialize(survey_folders)
job_ids=Vector{Chain_ID}()
for (partition,folder) in report_folders
    report=folder.partition_report
    for id in report.best_repset
        push!(job_ids,id)
    end
end

if isfile(refined_path) #if some results have already been collected, load them
    @info "Loading incomplete results..."
    refined_chains=deserialize(refined_path)
else
    @info "Initialising new HMM results file at $refined_path"
    chains = deserialize(hmm_output)
    refined_chains=Dict{Chain_ID,Vector{EM_step}}()
    for job_id in job_ids
        refined_chains[job_id]=[EM_step(1, chains[job_id][end].hmm, 0, 0, false)]
    end
end

#DISTRIBUTED CLUSTERs CONSTANTS
remote_machine = "10.0.0.3"
no_local_processes = 1
no_remote_processes = 1
load_dict=Dict{Int64,LoadConfig}()
local_config=LoadConfig(1:6,0:2)
remote_config=LoadConfig(1:6,0:2)

#SETUP DISTRIBUTED BAUM WELCH LEARNERS
@info "Spawning local cluster workers..."
worker_pool=addprocs(no_local_processes, topology=:master_worker)
for worker in worker_pool
    load_dict[worker]=local_config
end

remote_pool=addprocs([(remote_machine,no_remote_processes)], tunnel=true, topology=:master_worker)
for worker in remote_pool
    load_dict[worker]=remote_config
end

worker_pool=vcat(worker_pool, remote_pool)

@info "Loading worker libraries everywhere..."
@everywhere using BioBackgroundModels

em_jobset = setup_EM_jobs!(job_ids, training_sets; delta_thresh=delta_thresh, chains=refined_chains)
execute_EM_jobs!(worker_pool, em_jobset..., refined_path; delta_thresh=delta_thresh, max_iterates=max_iterates, load_dict=load_dict)
