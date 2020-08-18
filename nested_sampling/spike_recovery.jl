#Testbed for synthetic spike recovery from example background
using BioBackgroundModels, BioMotifInference, Random, Distributed, Distributions, Serialization
Random.seed!(786)
#CONSTANTS
no_obs=500
obsl=100:200

const folder_path="/bench/PhD/NGS_binaries/BBM/refined_folders"

report_folders=deserialize(folder_path)

bhmm_vec=Vector{BHMM}()
push!(bhmm_vec,report_folders["intergenic"].partition_report.best_model[2])
push!(bhmm_vec,report_folders["periexonic"].partition_report.best_model[2])
push!(bhmm_vec,report_folders["exon"].partition_report.best_model[2])

bhmm_dist=Categorical([.6,.2,.2])

struc_sig_1=[.1 .7 .1 .1
           .1 .1 .1 .7
           .1 .7 .1 .1]
struc_sig_2=[.1 .1 .7 .1
           .1 .1 .7 .1
           .7 .1 .1 .1]
periodicity=8
struc_frac_obs=.75


tata_box=[.05 .05 .05 .85
          .85 .05 .05 .05
          .05 .05 .05 .85
          .85 .05 .05 .05
          .425 .075 .075 .425
          .85 .05 .05 .05
          .425 .075 .075 .425]
caat_box=[.15 .15 .55 .15
          .15 .15 .55 .15
          .05 .85 .05 .05
          .05 .85 .05 .05
          .85 .05 .05 .05
          .85 .05 .05 .05
          .05 .05 .05 .85
          .15 .55 .15 .15
          .15 .15 .15 .55]
motif_frac_obs=.7
motif_recur_range=1:4

@info "Constructing synthetic sample set 1..."
obs1, bg_scores1, hmm_truth1, spike_truth1 = synthetic_sample(no_obs,obsl,bhmm_vec,bhmm_dist,[struc_sig_1,tata_box],[(true,(struc_frac_obs,periodicity)),(false,(motif_frac_obs,motif_recur_range))])

@info "Constructing synthetic sample set 2..."
obs2, bg_scores2, hmm_truth2, spike_truth2 = synthetic_sample(no_obs,obsl,bhmm_vec,bhmm_dist,[struc_sig_2,caat_box],[(true,(struc_frac_obs,periodicity)),(false,(motif_frac_obs,motif_recur_range))])

@info "Assembling combined sample set..."
obs3=hcat(obs1,obs2)
bg_scores3=hcat(bg_scores1, bg_scores2)

@info "Assembling worker pool..."

#DISTRIBUTED CLUSTERs CONSTANTS
remote_machine = "10.0.0.3"
no_local_processes = 2
no_remote_processes = 6

@info "Spawning local cluster workers..."
worker_pool=addprocs(no_local_processes, topology=:master_worker)
remote_pool=addprocs([(remote_machine,no_remote_processes)], tunnel=true, topology=:master_worker)

worker_pool=vcat(worker_pool, remote_pool)

@info "Loading worker libraries everywhere..."
@everywhere using BioMotifInference, Random
@everywhere Random.seed!(myid())

e1 = "/bench/PhD/NGS_binaries/BMI/e1"
e2 = "/bench/PhD/NGS_binaries/BMI/e2"
e3 = "/bench/PhD/NGS_binaries/BMI/e3"

#JOB CONSTANTS
const ensemble_size = 250
const no_sources = 2
const source_min_bases = 3
const source_max_bases = 12
const source_length_range= source_min_bases:source_max_bases
const mixing_prior = .5
const models_to_permute = ensemble_size * 3
funcvec=full_perm_funcvec
push!(funcvec, BioMotifInference.permute_source)
push!(funcvec, BioMotifInference.permute_source)
args=[Vector{Tuple{Symbol,Any}}() for i in 1:length(funcvec)]
args[end-1]=[(:weight_shift_freq,0.),(:length_change_freq,1.),(:length_perm_range,1:1)]
args[end]=[(:weight_shift_freq,.1),(:length_change_freq,0.),(:weight_shift_dist,Uniform(.00001,.01))]


instruct = Permute_Instruct(funcvec, ones(length(funcvec))./length(funcvec),models_to_permute,100, .02; args=args)

@info "Assembling source priors..."
prior_array= Vector{Matrix{Float64}}()
source_priors = BioMotifInference.assemble_source_priors(no_sources, prior_array)

@info "Assembling ensemble 1..."
isfile(e1*"/ens") ? (ens1 = deserialize(e1*"/ens")) :
    (ens1 = IPM_Ensemble(worker_pool, e1, ensemble_size, source_priors, (falses(0,0), mixing_prior), bg_scores1, obs1, source_length_range))
    
@info "Converging ensemble 1..."
logZ1 = BioMotifInference.converge_ensemble!(ens1, instruct, worker_pool, .001, backup=(true,250), wk_disp=false, tuning_disp=true, ens_disp=false, conv_plot=true, src_disp=true, lh_disp=false, liwi_disp=false)

@info "Assembling ensemble 2..."
isfile(e2*"/ens") ? (ens1 = deserialize(e2*"/ens")) :
    (ens2 = IPM_Ensemble(worker_pool, e2, ensemble_size, source_priors, (falses(0,0), mixing_prior), bg_scores2, obs2, source_length_range))
    
@info "Converging ensemble 2..."
logZ2 = BioMotifInference.converge_ensemble!(ens2, instruct, worker_pool, .001, backup=(true,250), wk_disp=false, tuning_disp=true, ens_disp=false, conv_plot=true, src_disp=true, lh_disp=false, liwi_disp=false)

@info "Assembling ensemble 3..."
isfile(e3*"/ens") ? (ens1 = deserialize(e3*"/ens")) :
    (ens3 = IPM_Ensemble(worker_pool, e3, ensemble_size, source_priors, (falses(0,0), mixing_prior), bg_scores3, obs3, source_length_range))
    
@info "Converging ensemble 3..."

logZ3 = BioMotifInference.converge_ensemble!(ens3, instruct, worker_pool, .001, backup=(true,250), wk_disp=false, tuning_disp=true, ens_disp=false, conv_plot=true, src_disp=true, lh_disp=false, liwi_disp=false)

@info "Evidence ratio of joint 1&2 / 3:"
ratio=exp((logZ1+logZ2)-logZ3)
println(ratio)

rmprocs(worker_pool)
