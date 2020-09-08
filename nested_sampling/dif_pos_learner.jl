using Distributed, Distributions, Serialization
include("/srv/git/rys_nucleosomes/aws/aws_wrangler.jl")

sib_e_pth = "/bench/PhD/NGS_binaries/BMI/sib_e"
rys_e_pth = "/bench/PhD/NGS_binaries/BMI/rys_e"
combined_e_pth = "/bench/PhD/NGS_binaries/BMI/combined_e"

@info "Assembling worker pool..."

#DISTRIBUTED CLUSTERs CONSTANTS
remote_machine = "10.0.0.3"
no_local_processes = 2
no_remote_processes = 6

@info "Spawning local cluster workers..."
worker_pool=addprocs(no_local_processes, topology=:master_worker)
remote_pool=addprocs([(remote_machine,no_remote_processes)], tunnel=true, topology=:master_worker)

worker_pool=vcat(worker_pool, remote_pool)

#AWS PARAMS
@info "Setting up AWS wrangling..."

security_group_name="calc1"
security_group_desc="calculation group"
ami="ami-0534a64ad75fb173a"
skeys="AWS"
instance_type="c5a.24xlarge"
zone,spot_price=get_cheapest_zone(instance_type)
no_instances=4
instance_workers=48
bid=spot_price+.1

@assert bid >= spot_price

@info "Wrangling AWS instances..."
aws_ips = spot_wrangle(no_instances, bid, security_group_name, security_group_desc, skeys, zone, ami, instance_type)
@info "Giving instances 120s to boot..."
sleep(120)

# aws_ips = ["18.223.21.162","18.225.5.234","18.222.238.67","3.134.91.251"]

@info "Spawning AWS cluster workers..."
for ip in aws_ips
    instance_pool=addprocs([(ip, instance_workers)], tunnel=true, topology=:master_worker, sshflags="-o StrictHostKeyChecking=no")
    global worker_pool=vcat(worker_pool, instance_pool)
end

@info "Loading worker libraries everywhere..."
@everywhere using BioMotifInference,Random
@everywhere Random.seed!(myid()*100000)

#JOB CONSTANTS
funcvec=full_perm_funcvec
models_to_permute=5000
func_limit=20
push!(funcvec, BioMotifInference.perm_src_fit_mix)
push!(funcvec, BioMotifInference.permute_source)
push!(funcvec, BioMotifInference.permute_mix)

min_clamps=fill(.015,length(funcvec))
max_clamps=fill(1.,length(funcvec))
max_clamps[6:9].=.5 #merges should be clamped to no more than half of function calls to prevent network hammering

initial_weights= ones(length(funcvec))./length(funcvec)
override_weights=fill(.015,length(funcvec))
override_weights[6]=.05;override_weights[7:9].=.045;override_weights[13:14].=.3475
override_time=10.

args=[Vector{Tuple{Symbol,Any}}() for i in 1:length(funcvec)]
args[end-2]=[(:weight_shift_freq,0.),(:length_change_freq,1.),(:length_perm_range,1:1)]
args[end-1]=[(:weight_shift_freq,.1),(:length_change_freq,0.),(:weight_shift_dist,Uniform(.000001,.01))]
args[end]=[(:iterates,50),(:mix_move_range,1:250)]


instruct = Permute_Instruct(funcvec, initial_weights, models_to_permute, func_limit;min_clmps=min_clamps, max_clmps=max_clamps, override_time=override_time, override_weights=override_weights, args=args)

display_rotation=[true,10,1,[[:tuning_disp,:lh_disp,:src_disp],[:conv_plot,:liwi_disp,:ens_disp]]]

@info "Beginning nested sampling for sib ensemble..."
sib_e=deserialize(sib_e_pth*"/ens")
sib_logZ = converge_ensemble!(sib_e, instruct, worker_pool, .001, backup=(true,25), tuning_disp=true,lh_disp=true,src_disp=true, disp_rotate_inst=display_rotation)
sib_e=[]; Base.GC.gc();

@info "Beginning nested sampling for rys ensemble..."
rys_e=deserialize(rys_e_pth*"/ens")
rys_logZ = converge_ensemble!(rys_e, instruct, worker_pool, .001, backup=(true,25), tuning_disp=true,lh_disp=true,src_disp=true, disp_rotate_inst=display_rotation)
rys_e=[]; Base.GC.gc();

@info "Beginning nested sampling for combined ensemble..."
combined_e=deserialize(combined_e_pth*"/ens")
combined_logZ = converge_ensemble!(combined_e, instruct, worker_pool, .001, backup=(true,25), tuning_disp=true,lh_disp=true,src_disp=true,disp_rotate_inst=display_rotation)
combined_e=[]; Base.GC.gc();

@info "Evidence ratio of joint sib+rys / combined:"
ratio=exp((logZ1+logZ2)-logZ3)
println(ratio)

rmprocs(worker_pool)

@info "Job done!"
