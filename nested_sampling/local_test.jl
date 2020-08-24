using Distributed, Distributions, Serialization

test_e_pth = "/bench/PhD/NGS_binaries/BMI/test_e"

@info "Assembling worker pool..."

#DISTRIBUTED CLUSTERs CONSTANTS
remote_machine = "10.0.0.3"
no_local_processes = 1

@info "Spawning local cluster workers..."
worker_pool=addprocs(no_local_processes, topology=:master_worker)

@info "Loading worker libraries everywhere..."
@everywhere using BioMotifInference,Random
@everywhere Random.seed!(myid()*100000)

#JOB CONSTANTS
models_to_permute=5000
func_limit=30
clamp=.02
funcvec=full_perm_funcvec
push!(funcvec, BioMotifInference.permute_source)
push!(funcvec, BioMotifInference.permute_source)
args=[Vector{Tuple{Symbol,Any}}() for i in 1:length(funcvec)]
args[end-1]=[(:weight_shift_freq,0.),(:length_change_freq,1.),(:length_perm_range,1:1)]
args[end]=[(:weight_shift_freq,.1),(:length_change_freq,0.),(:weight_shift_dist,Uniform(.00001,.01))]

instruct = Permute_Instruct(funcvec, ones(length(funcvec))./length(funcvec), models_to_permute, func_limit, clamp; args=args)

display_rotation=[true,10,1,[[:tuning_disp,:lh_disp,:src_disp],[:conv_plot,:liwi_disp,:ens_disp]]]

@info "Beginning nested sampling for test ensemble..."
test_e=deserialize(test_e_pth*"/ens")
test_logZ = converge_ensemble!(test_e, instruct, worker_pool, .001, backup=(true,1),tuning_disp=true,lh_disp=true,src_disp=true, disp_rotate_inst=display_rotation)

rmprocs(worker_pool)

@info "Job done!"
