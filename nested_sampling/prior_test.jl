@info "Setting up for job..."
#JOB FILEPATHS
sib_wms_path = "/bench/PhD/NGS_binaries/BMI/sib_nuc_position_sequences.fa_wms.tr"
rys_wms_path = "/bench/PhD/NGS_binaries/BMI/rys_nuc_position_sequences.fa_wms.tr"

sib_df_binary = "/bench/PhD/NGS_binaries/BMI/sib_diff_positions"
rys_df_binary = "/bench/PhD/NGS_binaries/BMI/rys_diff_positions"
combined_df_binary = "/bench/PhD/NGS_binaries/BMI/combined_diff_positions"

sib_code_binary = "/bench/PhD/NGS_binaries/BMI/sib_diff_codes"
rys_code_binary = "/bench/PhD/NGS_binaries/BMI/rys_diff_codes"
combined_code_binary = "/bench/PhD/NGS_binaries/BMI/combined_diff_codes"

sib_diff_bg = "/bench/PhD/NGS_binaries/BMI/sib_diff_bg"
rys_diff_bg = "/bench/PhD/NGS_binaries/BMI/rys_diff_bg"
combined_diff_bg = "/bench/PhD/NGS_binaries/BMI/combined_diff_bg"

sib_e_ui_pth = "/bench/PhD/NGS_binaries/BMI/sib_e_ui"
sib_e_inf_pth = "/bench/PhD/NGS_binaries/BMI/sib_e_inf"
rys_e_ui_pth = "/bench/PhD/NGS_binaries/BMI/rys_e_ui"
rys_e_inf_pth = "/bench/PhD/NGS_binaries/BMI/rys_e_inf"
combined_e_ui_pth = "/bench/PhD/NGS_binaries/BMI/combined_e_ui"
combined_e_inf_pth = "/bench/PhD/NGS_binaries/BMI/combined_e_inf"


#JOB CONSTANTS
const ensemble_size = 1000
const no_sources = 8
const source_min_bases = 3
const source_max_bases = 10
@assert source_min_bases < source_max_bases
const source_length_range= source_min_bases:source_max_bases
const mixing_prior = .07
@assert mixing_prior >= 0 && mixing_prior <= 1
const prior_wt=1.2

@info "Loading master libraries..."
using Distributed, Distributions, Serialization

@info "Adding librarians and workers..."
no_local_procs=2
no_remote_procs=6
remote_machine = "10.0.0.119"
remote_pool=addprocs([(remote_machine, no_remote_procs)], tunnel=true, topology=:master_worker)
local_pool=addprocs(no_local_procs, topology=:master_worker)
worker_pool=vcat(remote_pool,local_pool)

@info "Loading libraries everywhere..."
@everywhere using BioMotifInference, Random
Random.seed!(myid()*10000)

@info "Loading informative source priors..."
sib_wms = BioMotifInference.read_fa_wms_tr(sib_wms_path)
rys_wms = BioMotifInference.read_fa_wms_tr(rys_wms_path)
sib_mix_prior = BioMotifInference.cluster_mix_prior!(deserialize(sib_df_binary), sib_wms)
rys_mix_prior = BioMotifInference.cluster_mix_prior!(deserialize(rys_df_binary), rys_wms)

@info "Filtering informative priors..."
sib_prior_wms = BioMotifInference.filter_priors(Int(floor(no_sources/2)), source_max_bases, sib_wms, sib_mix_prior)
rys_prior_wms = BioMotifInference.filter_priors(Int(floor(no_sources/2)), source_max_bases, rys_wms, rys_mix_prior)
combined_prior_wms = BioMotifInference.combine_filter_priors(Int(floor(no_sources/2)), source_max_bases, (sib_wms, rys_wms), (sib_mix_prior, rys_mix_prior))

@info "Assembling informative source priors..."
sib_inf_sp = BioMotifInference.assemble_source_priors(no_sources, sib_prior_wms, prior_wt)
rys_inf_sp = BioMotifInference.assemble_source_priors(no_sources, rys_prior_wms, prior_wt)
combined_inf_sp = BioMotifInference.assemble_source_priors(no_sources, combined_prior_wms, prior_wt)

@info "Assembling uninformative source priors..."
sib_ui_sp = BioMotifInference.assemble_source_priors(no_sources, Vector{Matrix{Float64}}())
rys_ui_sp = BioMotifInference.assemble_source_priors(no_sources, Vector{Matrix{Float64}}())
combined_ui_sp = BioMotifInference.assemble_source_priors(no_sources, Vector{Matrix{Float64}}())

@info "Loading BGHMM likelihood matrix binaries..."
sib_matrix=deserialize(sib_diff_bg)
rys_matrix=deserialize(rys_diff_bg)
combined_matrix=deserialize(combined_diff_bg)

@info "Loading coded observation sets..."
sib_obs = deserialize(sib_code_binary)
rys_obs = deserialize(rys_code_binary)
combined_obs = deserialize(combined_code_binary)

@info "Assembling sib IPM ensemble on uninformative priors..."
isfile(string(sib_e_ui_pth,'/',"ens")) ? (sib_e_ui = deserialize(string(sib_e_ui_pth,'/',"ens"))) :
    (sib_e_ui = BioMotifInference.IPM_Ensemble(worker_pool, sib_e_ui_pth, ensemble_size, sib_ui_sp, (falses(0,0), mixing_prior), sib_matrix, sib_obs, source_length_range); serialize(string(sib_e_ui_pth,'/',"ens"),sib_e_ui))

@info "Assembling sib IPM ensemble on informative priors..."
isfile(string(sib_e_inf_pth,'/',"ens")) ? (sib_e_inf = deserialize(string(sib_e_inf_pth,'/',"ens"))) :
        (sib_e_inf = BioMotifInference.IPM_Ensemble(worker_pool, sib_e_inf_pth, ensemble_size, sib_inf_sp, (falses(0,0), mixing_prior), sib_matrix, sib_obs, source_length_range); serialize(string(sib_e_inf_pth,'/',"ens"),sib_e_inf))
    
@info "Assembling rys IPM ensemble on uninformative priors..."
isfile(string(rys_e_ui_pth,'/',"ens")) ? (rys_e_ui = deserialize(string(rys_e_ui_pth,'/',"ens"))) :
    (rys_e_ui = BioMotifInference.IPM_Ensemble(worker_pool, rys_e_ui_pth, ensemble_size, rys_ui_sp, (falses(0,0), mixing_prior), rys_matrix, rys_obs, source_length_range); serialize(string(rys_e_ui_pth,'/',"ens"),rys_e_ui))

@info "Assembling rys IPM ensemble on informative priors..."
isfile(string(rys_e_inf_pth,'/',"ens")) ? (rys_e_inf = deserialize(string(rys_e_inf_pth,'/',"ens"))) :
        (rys_e_inf = BioMotifInference.IPM_Ensemble(worker_pool, rys_e_inf_pth, ensemble_size, rys_inf_sp, (falses(0,0), mixing_prior), rys_matrix, rys_obs, source_length_range); serialize(string(rys_e_inf_pth,'/',"ens"),rys_e_inf))

@info "Assembling combined IPM ensemble on uninformative priors..."
isfile(string(combined_e_ui_pth,'/',"ens")) ? (combined_e_ui = deserialize(string(combined_e_ui_pth,'/',"ens"))) :
    (combined_e_ui = BioMotifInference.IPM_Ensemble(worker_pool, combined_e_ui_pth, ensemble_size, combined_ui_sp, (falses(0,0), mixing_prior), combined_matrix, combined_obs, source_length_range); serialize(string(combined_e_ui_pth,'/',"ens"),combined_e_ui))

@info "Assembling combined IPM ensemble on informative priors..."
isfile(string(combined_e_inf_pth,'/',"ens")) ? (combined_e_inf = deserialize(string(combined_e_inf_pth,'/',"ens"))) :
        (combined_e_inf = BioMotifInference.IPM_Ensemble(worker_pool, combined_e_inf_pth, ensemble_size, combined_inf_sp, (falses(0,0), mixing_prior), combined_matrix, combined_obs, source_length_range); serialize(string(combined_e_inf_pth,'/',"ens"),combined_e_inf))

rmprocs(worker_pool)

@info "Fitting distributions..."

sib_ui_dist=fit_mle(Normal,[model.log_Li for model in sib_e_ui.models])
sib_inf_dist=fit_mle(Normal,[model.log_Li for model in sib_e_inf.models])
@info "Is informative a better prior for sibs?"
println(sib_inf_dist.μ > sib_ui_dist.μ && quantile(sib_inf_dist,.5) > sib_ui_dist.μ)

rys_ui_dist=fit_mle(Normal,[model.log_Li for model in rys_e_ui.models])
rys_inf_dist=fit_mle(Normal,[model.log_Li for model in rys_e_inf.models])
@info "Is informative a better prior for rys?"
println(rys_inf_dist.μ > rys_ui_dist.μ && quantile(rys_inf_dist,.5) > rys_ui_dist.μ)

combined_ui_dist=fit_mle(Normal,[model.log_Li for model in combined_e_ui.models])
combined_inf_dist=fit_mle(Normal,[model.log_Li for model in combined_e_inf.models])
@info "Is informative a better prior for combined?"
println(combined_inf_dist.μ > combined_ui_dist.μ && quantile(combined_inf_dist,.5) > combined_ui_dist.μ)

@info "Job done!"