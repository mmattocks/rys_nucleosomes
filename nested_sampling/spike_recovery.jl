#Testbed for synthetic spike recovery from example background
using BioBackgroundModels, BioMotifInference, Serialization
Random.seed!(786)
#CONSTANTS
no_obs=300
obsl=100:200

const folder_path="/bench/PhD/NGS_binaries/BBM/refined_folders"

report_folders=deserialize(folder_path)

bhmm_vec=Vector{BHMM}()
push!(bhmm_vec,report_folders["intergenic"].best_model[2])
push!(bhmm_vec,report_folders["periexonic"].best_model[2])
push!(bhmm_vec,report_folders["exon"].best_model[2])

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

ensemble = "/bench/PhD/NGS_binaries/BioMotifInference/sr_ens"

#JOB CONSTANTS
const position_size = 141
const ensemble_size = 100
const no_sources = 3
const source_min_bases = 3
const source_max_bases = 12
@assert source_min_bases < source_max_bases
const source_length_range= source_min_bases:source_max_bases
const mixing_prior = .3
@assert mixing_prior >= 0 && mixing_prior <= 1
const models_to_permute = ensemble_size * 3

@info "Setting up synthetic observation set..."

@info "Assembling source priors..."
#prior_array= [struc_sig, tata_box]
prior_array= Vector{Matrix{Float64}}()
source_priors = nnlearn.assemble_source_priors(no_sources, prior_array, prior_wt, source_length_range)

@info "Assembling ensemble..."
path=randstring()
isfile(string(path,'/',"ens")) ? (ens = deserialize(string(path,'/',"ens"))) :
    (ens = nnlearn.IPM_Ensemble(path, ensemble_size, source_priors, (falses(0,0), mixing_prior), bg_lhs, obs, source_length_range))


job_set_thresh=[-Inf,ens.naive_lh]
param_set=(job_sets,job_set_thresh,job_limit)
    
@info "Converging ensemble..."

final_logZ = converge_ensemble!(ensemble, instruct, worker_pool, .001, backup=(true,250), wk_disp=false, tuning_disp=true, ens_disp=false, conv_plot=true, src_disp=true, lh_disp=false, liwi_disp=false)

rmprocs(worker_pool)

rm(ensemble,recursive=true)

#811973