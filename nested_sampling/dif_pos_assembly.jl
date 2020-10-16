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

sib_e_pth = "/bench/PhD/NGS_binaries/BMI/sib_e"
rys_e_pth = "/bench/PhD/NGS_binaries/BMI/rys_e"
combined_e_pth = "/bench/PhD/NGS_binaries/BMI/combined_e"

#JOB CONSTANTS
const ensemble_size = 500
const no_sources = 8
const source_min_bases = 3
const source_max_bases = 10
@assert source_min_bases < source_max_bases
const source_length_range= source_min_bases:source_max_bases
const mixing_prior = .07
@assert mixing_prior >= 0 && mixing_prior <= 1

@info "Loading master libraries..."
using Distributed, Distributions, Serialization

@info "Adding workers..."
no_local_procs=2
worker_pool=addprocs(no_local_procs, topology=:master_worker)

@info "Loading libraries everywhere..."
@everywhere using BioMotifInference, Random
Random.seed!(myid()*10000)

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
isfile(string(sib_e_pth,'/',"ens")) ? (sib_e = deserialize(string(sib_e_pth,'/',"ens"))) :
    (sib_e = BioMotifInference.IPM_Ensemble(worker_pool, sib_e_pth, ensemble_size, sib_ui_sp, (falses(0,0), mixing_prior), sib_matrix, sib_obs, source_length_range, posterior_switch=true); serialize(string(sib_e_pth,'/',"ens"),sib_e))
sib_e=[]; Base.GC.gc();

@info "Assembling rys IPM ensemble on uninformative priors..."
isfile(string(rys_e_pth,'/',"ens")) ? (rys_e = deserialize(string(rys_e_pth,'/',"ens"))) :
    (rys_e = BioMotifInference.IPM_Ensemble(worker_pool, rys_e_pth, ensemble_size, rys_ui_sp, (falses(0,0), mixing_prior), rys_matrix, rys_obs, source_length_range, posterior_switch=true); serialize(string(rys_e_pth,'/',"ens"),rys_e))
rys_e=[]; Base.GC.gc();


@info "Assembling combined IPM ensemble on uninformative priors..."
isfile(string(combined_e_pth,'/',"ens")) ? (combined_e = deserialize(string(combined_e_pth,'/',"ens"))) :
    (combined_e = BioMotifInference.IPM_Ensemble(worker_pool, combined_e_pth, ensemble_size, combined_ui_sp, (falses(0,0), mixing_prior), combined_matrix, combined_obs, source_length_range, posterior_switch=true); serialize(string(combined_e_pth,'/',"ens"),combined_e))
combined_e=[]; Base.GC.gc();

rmprocs(worker_pool)

@info "Job done!"