using BioBackgroundModels, Serialization

hmm_output = "/bench/PhD/NGS_binaries/BBM/survey_chains"
sample_output = "/bench/PhD/NGS_binaries/BBM/survey_samples"

survey_folders = "/bench/PhD/NGS_binaries/BBM/survey_folders"

chains=deserialize(hmm_output)
sample_dfs = deserialize(sample_output)
training_sets, test_sets = split_obs_sets(sample_dfs)

report_folders=generate_reports(chains, test_sets)
serialize(survey_folders, report_folders)