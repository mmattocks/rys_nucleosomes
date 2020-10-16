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
ami="ami-0bb0f5609b995d39e"
skeys="AWS"
instance_type="c5a.24xlarge"
zone,spot_price=get_cheapest_zone(instance_type)
no_instances=5
instance_workers=48
bid=spot_price+.1

@assert bid >= spot_price

@info "Wrangling AWS instances..."
aws_ips = spot_wrangle(no_instances, bid, security_group_name, security_group_desc, skeys, zone, ami, instance_type)
@info "Giving instances 150s to boot..."
sleep(150)

# aws_ips=["18.223.214.57","18.224.4.81", "3.17.13.150","3.21.113.229","18.222.211.44"]


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
models_to_permute=10000
func_limit=25
push!(funcvec, BioMotifInference.perm_src_fit_mix)
push!(funcvec, BioMotifInference.random_decorrelate)

min_clamps=fill(.01,length(funcvec))
min_clamps[2:3].=.1 #perm_src_fit_mix & permute_mix
min_clamps[8]=.1 #difference_merge
max_clamps=fill(.5,length(funcvec))
max_clamps[6:7].=.15 #ss and am
max_clamps[9]=.15 #sim merge

initial_weights= ones(length(funcvec))./length(funcvec)
# override_weights=fill(.034375,length(funcvec))
# override_weights[6:9].=.1;override_weights[13:14].=.1625
# override_time=20.

args=[Vector{Tuple{Symbol,Any}}() for i in 1:length(funcvec)]
args[end-1]=[(:weight_shift_freq,0.),(:length_change_freq,1.),(:length_perm_range,1:1)]
args[end]=[(:iterates,50),(:source_permute_freq,.3),(:mix_move_range,1:10)]

instruct = Permute_Instruct(funcvec, initial_weights, models_to_permute, func_limit;min_clmps=min_clamps, max_clmps=max_clamps, args=args)

display_rotation=[true,10,1,[[:tuning_disp,:lh_disp,:src_disp],[:conv_plot,:liwi_disp,:ens_disp]]]

@info "Beginning nested sampling for sib ensemble..."
sib_e=deserialize(sib_e_pth*"/ens")
sib_logZ = converge_ensemble!(sib_e, instruct, worker_pool, converge_criterion="compression", converge_factor=150., backup=(true,25), tuning_disp=true,lh_disp=true,src_disp=true, disp_rotate_inst=display_rotation)
sib_logZ_err=sqrt(sib_e.Hi[end]/length(sib_e.log_Li))
sib_e=[]; Base.GC.gc();

@info "Beginning nested sampling for rys ensemble..."
rys_e=deserialize(rys_e_pth*"/ens")
rys_logZ = converge_ensemble!(rys_e, instruct, worker_pool, converge_criterion="compression", converge_factor=150., backup=(true,25), tuning_disp=true,lh_disp=true,src_disp=true, disp_rotate_inst=display_rotation)
rys_logZ_err=sqrt(rys_e.Hi[end]/length(rys_e.log_Li))
rys_e=[]; Base.GC.gc();

@info "Beginning nested sampling for combined ensemble..."
combined_e=deserialize(combined_e_pth*"/ens")
combined_logZ = converge_ensemble!(combined_e, instruct, worker_pool, converge_criterion="compression", converge_factor=150., backup=(true,25), tuning_disp=true,lh_disp=true,src_disp=true,disp_rotate_inst=display_rotation)
combined_logZ_err=sqrt(combined_e.Hi[end]/length(combined_e.log_Li))
combined_e=[]; Base.GC.gc();

@info "log evidence ratio of joint sib+rys / combined:"
ratio=(sib_logZ+rys_logZ)-combined_logZ
ratio_err=sqrt(sib_logZ_err^2 + rys_logZ_err^2 + combined_logZ_err^2)
println("$ratio ± $ratio_err, $(ratio/ratio_err) standard deviation significance")

serialize("/bench/PhD/NGS_binaries/BMI/dif_pos_learner_report", "$ratio ± $ratio_err, $(ratio/ratio_err) standard deviation significance")

rmprocs(worker_pool)

@info "Job done!"
