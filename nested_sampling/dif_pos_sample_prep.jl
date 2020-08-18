using BioSequences, Distributed, DataFrames, nucpos, BioBackgroundModels, CSV, Serialization

sib_pos = "/bench/PhD/danpos_results/pooled/sib.Fnor.smooth.positions.xls"
rys_pos = "/bench/PhD/danpos_results/pooled/rys.Fnor.smooth.positions.xls"

danio_genome_path = "/bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna"
danio_gen_index_path = "/bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna.fai"
danio_gff_path = "/bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3"

refined_folders_path = "/bench/PhD/NGS_binaries/BBM/refined_folders"
selected_hmms = "/bench/PhD/NGS_binaries/BBM/selected_hmms"

sib_df_binary = "/bench/PhD/NGS_binaries/BMI/sib_diff_positions"
rys_df_binary = "/bench/PhD/NGS_binaries/BMI/rys_diff_positions"
combined_df_binary = "/bench/PhD/NGS_binaries/BMI/combined_diff_positions"

sib_code_binary = "/bench/PhD/NGS_binaries/BMI/sib_diff_codes"
rys_code_binary = "/bench/PhD/NGS_binaries/BMI/rys_diff_codes"
combined_code_binary = "/bench/PhD/NGS_binaries/BMI/combined_diff_codes"

sib_diff_bg = "/bench/PhD/NGS_binaries/BMI/sib_diff_bg"
rys_diff_bg = "/bench/PhD/NGS_binaries/BMI/rys_diff_bg"
combined_diff_bg = "/bench/PhD/NGS_binaries/BMI/combined_diff_bg"

sib_fa = "/bench/PhD/thicweed_results/sib_nuc_position_sequences.fa"
sib_arch = "/bench/PhD/thicweed_results/sib_nuc_position_sequences.fa_archs.txt"

rys_fa = "/bench/PhD/thicweed_results/rys_nuc_position_sequences.fa"
rys_arch = "/bench/PhD/thicweed_results/rys_nuc_position_sequences.fa_archs.txt"

@info "Reading from position.xls..."
sib_df=CSV.read(sib_pos)
rys_df=CSV.read(rys_pos)

@info "Mapping..."
nucpos.map_positions!(sib_df, rys_df)
nucpos.map_positions!(rys_df, sib_df)

@info "Making differential position dataframes..."
sib_diff_df = deepcopy(sib_df[findall(iszero,sib_df.mapped_pos),:])
sib_diff_df = sib_diff_df[findall(!isequal("MT"), sib_diff_df.chr),:]
rys_diff_df = deepcopy(rys_df[findall(iszero,rys_df.mapped_pos),:])
rys_diff_df = rys_diff_df[findall(!isequal("MT"), rys_diff_df.chr),:]

nucpos.add_position_sequences!(sib_diff_df, danio_genome_path, danio_gen_index_path)
nucpos.add_position_sequences!(rys_diff_df, danio_genome_path, danio_gen_index_path)

@info "Filtering ambiguous sequences..."
deleterows!(sib_diff_df, [hasambiguity(seq) for seq in sib_diff_df.seq])
deleterows!(rys_diff_df, [hasambiguity(seq) for seq in rys_diff_df.seq])

@info "Masking positions by genome partition and strand..."
BioBackgroundModels.add_partition_masks!(sib_diff_df, danio_gff_path, 500, (:chr,:seq,:start))
BioBackgroundModels.add_partition_masks!(rys_diff_df, danio_gff_path, 500, (:chr,:seq,:start))

@info "Obtaining cluster data..."
nucpos.get_cluster!(sib_diff_df, sib_fa, sib_arch)
nucpos.get_cluster!(rys_diff_df, rys_fa, rys_arch)

@info "Serializing dataframes..."
combined_diff_df = vcat(sib_diff_df, rys_diff_df)

serialize(sib_df_binary, sib_diff_df)
serialize(rys_df_binary, rys_diff_df)
serialize(combined_df_binary, combined_diff_df)

@info "Creating coded observation sets..."
sib_codes = nucpos.observation_setup(sib_diff_df, order=0, symbol=:seq)
rys_codes = nucpos.observation_setup(rys_diff_df, order=0, symbol=:seq)
combined_codes = nucpos.observation_setup(combined_diff_df, order=0, symbol=:seq)

@info "Serializing coded observation sets..."
serialize(sib_code_binary, Matrix(transpose(sib_codes)))
serialize(rys_code_binary, Matrix(transpose(rys_codes)))
serialize(combined_code_binary, Matrix(transpose(combined_codes)))

@info "Setting up for BBM likelihood calculations..."
BHMM_dict = Dict{String,BHMM}()
refined_folders=deserialize(refined_folders_path)
for (part, folder) in refined_folders
    BHMM_dict[part]=folder.partition_report.best_model[2]
end

serialize(selected_hmms,BHMM_dict)

@info "Performing calculations..."
sib_lh_matrix = BGHMM_likelihood_calc(sib_diff_df, BHMM_dict, symbol=:seq)
rys_lh_matrix = BGHMM_likelihood_calc(rys_diff_df, BHMM_dict, symbol=:seq)
combined_lh_matrix = hcat(sib_lh_matrix,rys_lh_matrix)

@info "Serializing background matrices..."
serialize(sib_diff_bg, sib_lh_matrix)
serialize(rys_diff_bg, rys_lh_matrix)
serialize(combined_diff_bg, combined_lh_matrix)

@info "Job done!"