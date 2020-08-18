using Distributed, Distributions, Serialization

sib_e_ui_pth = "/bench/PhD/NGS_binaries/BMI/sib_e_ui"
rys_e_ui_pth = "/bench/PhD/NGS_binaries/BMI/rys_e_ui"
combined_e_ui_pth = "/bench/PhD/NGS_binaries/BMI/combined_e_ui"

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
ami="ami-02df7e1881a08f163"
skeys="AWS"
instance_type="c5.4xlarge"
zone,spot_price=get_cheapest_zone(instance_type)
no_instances=3
instance_workers=8
bid=spot_price+.01

@assert bid >= spot_price

@info "Wrangling AWS instances..."
aws_ips = spot_wrangle(no_instances, bid, security_group_name, security_group_desc, skeys, zone, ami, instance_type)
@info "Giving instances 90s to boot..."
sleep(90)

#aws_ips = ["18.223.122.201"]

@info "Spawning AWS cluster workers..."
for ip in aws_ips
    instance_pool=addprocs([(ip, instance_workers)], tunnel=true, topology=:master_worker, sshflags="-o StrictHostKeyChecking=no")
    for worker in instance_pool
        load_dict[worker]=aws_instance_config
    end
    global worker_pool=vcat(worker_pool, instance_pool)
end

@info "Loading worker libraries everywhere..."
@everywhere using BioMotifInference,Random
@everywhere Random.seed!(myid()*100000)

#JOB CONSTANTS
funcvec=full_perm_funcvec
push!(funcvec, BioMotifInference.permute_source)
push!(funcvec, BioMotifInference.permute_source)
args=[Vector{Tuple{Symbol,Any}}() for i in 1:length(funcvec)]
args[end-1]=[(:weight_shift_freq,0.),(:length_change_freq,1.),(:length_perm_range,1:1)]
args[end]=[(:weight_shift_freq,.1),(:length_change_freq,0.),(:weight_shift_dist,Uniform(.00001,.01))]

instruct = Permute_Instruct(funcvec, ones(length(funcvec))./length(funcvec),models_to_permute,100, .02; args=args)

display_rotation=[true,150,1,[[:tuning_disp,:lh_disp,:src_disp],[:conv_plot,:liwi_disp,:ens_disp]]]

@info "Beginning nested sampling for sib ensemble..."
sib_e=deserialize(sib_e_ui_pth)
sib_logZ = converge_ensemble!(sib_e, instruct, worker_pool, .001, backup=(true,250),disp_rotate_inst=display_rotation)
sib_e=[]

@info "Beginning nested sampling for rys ensemble..."
rys_e=deserialize(sib_e_ui_pth)
rys_logZ = converge_ensemble!(rys_e, instruct, worker_pool, .001, backup=(true,250),disp_rotate_inst=display_rotation)
rys_e=[]

@info "Beginning nested sampling for combined ensemble..."
combined_e=deserialize(combined_e_ui_pth)
combined_logZ = converge_ensemble!(combined_e, instruct, worker_pool, .001, backup=(true,250),disp_rotate_inst=display_rotation)
combined_e=[]

@info "Evidence ratio of joint sib+rys / combined:"
ratio=exp((logZ1+logZ2)-logZ3)
println(ratio)

rmprocs(worker_pool)

@info "Job done!"
