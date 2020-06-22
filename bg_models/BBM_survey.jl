#JOB FILEPATHS
#sample record and hmm serialisation output path
sample_output = "/bench/PhD/NGS_binaries/BBM/BBM_samples"
hmm_output = "/bench/PhD/NGS_binaries/BBM/hmm_chains"

#GENERAL SETUP
@info "Loading libraries..."
using BioBackgroundModels, DataFrames, Distributed, Serialization
include("/srv/git/rys_nucleosomes/aws/aws_wrangler.jl")

#JOB CONSTANTS
const replicates = 3 #repeat optimisation from this many seperately initialised samples from the prior
const Ks = [1,2,4,6] #mosaic class #s to test
const order_nos = [0,1,2] #DNA kmer order #s to test
const delta_thresh=1e-3 #stopping/convergence criterion (log probability difference btw subsequent EM iterates)
const max_iterates=15000

#LOAD SAMPLES
@info "Loading samples from $sample_output..."
sample_dfs = deserialize(sample_output)

#BUILD TRAINING AND TEST SETS FROM SAMPLES
training_sets, test_sets = split_obs_sets(sample_dfs)

#PROGRAMATICALLY GENERATE Chain_ID Vector
job_ids=Vector{Chain_ID}()
for (obs_id, obs) in training_sets, K in Ks, order in order_nos, rep in 1:replicates
    push!(job_ids, Chain_ID(obs_id, K, order, rep))
end

#DISTRIBUTED CLUSTER CONSTANTS
load_dict=Dict{Int64,LoadConfig}()
local_config=LoadConfig(6:6,2:2,[""])
remote_config=LoadConfig(4:6,1:2,[""])
aws_instance_config=LoadConfig(1:6,0:2,[""])
remote_machine = "10.0.0.3"
no_local_processes = 1
no_remote_processes = 3
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

#AWS PARAMS
@info "Setting up AWS wrangling..."

security_group_name="calc1"
security_group_desc="calculation group"
ami="ami-0ca1e8131cf324364"
skeys="AWS"
instance_type="c5.4xlarge"
zone,spot_price=get_cheapest_zone(instance_type)
no_instances=2
instance_workers=4
bid=spot_price+.01

@assert bid >= spot_price

@info "Wrangling AWS instances..."
aws_ips = spot_wrangle(no_instances, bid, security_group_name, security_group_desc, skeys, zone, ami, instance_type)
@info "Giving instances 90s to boot..."
sleep(90)

# aws_ips = ["18.224.21.217","3.19.241.206"]

@info "Spawning AWS cluster workers..."
for ip in aws_ips
    instance_pool=addprocs([(ip, instance_workers)], tunnel=true, topology=:master_worker, sshflags="-o StrictHostKeyChecking=no")
    for worker in instance_pool
        load_dict[worker]=aws_instance_config
    end
    global worker_pool=vcat(worker_pool, instance_pool)
end

@info "Loading worker libraries everywhere..."
@everywhere using BioBackgroundModels
#INTIIALIZE HMMS
@info "Setting up HMMs..."
if isfile(hmm_output) #if some results have already been collected, load them
    @info "Loading incomplete results..."
    hmm_results_dict = deserialize(hmm_output)
else #otherwise, pass a new results dict
    @info "Initialising new HMM results file at $hmm_output"
    hmm_results_dict = Dict{Chain_ID,Vector{EM_step}}()
end

em_jobset = setup_EM_jobs!(job_ids, training_sets; chains=hmm_results_dict)
execute_EM_jobs!(worker_pool, em_jobset..., hmm_output; load_dict=load_dict, delta_thresh=delta_thresh, max_iterates=max_iterates)
