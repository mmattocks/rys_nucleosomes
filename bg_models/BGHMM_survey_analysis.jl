#JOB FILEPATHS
sample_output = "/bench/PhD/NGS_binaries/BGHMM/BGHMM_samples"
hmm_output = "/bench/PhD/NGS_binaries/BGHMM/hmmchains"
selected_hmm_output = "/bench/PhD/NGS_binaries/BGHMM/selected_BGHMMs"
survey_results_path = "/bench/PhD/NGS_binaries/BGHMM/survey_chains"

#GENERAL SETUP
@info "Loading libraries..."
using BGHMM, BioSequences, DataFrames, Distributions, HMMBase, CLHMM, ProgressMeter, Serialization, Statistics, Plots

const replicates = 3 #repeat optimisation from this many seperately initialised samples from the prior
const Ks = [1,2,4,6] #mosaic class #s to test
const order_nos = [0,1,2] #DNA kmer order #s to test
const partitions = ["exon", "intergenic", "periexonic"]
const run_samples = 10000

#LOAD SAMPLES
@info "Loading samples from $sample_output..."
sample_dfs = deserialize(sample_output)

#LOAD HMM CHAINS
@info "Loading hmm chains from $hmm_output"
hmm_results_dict = deserialize(hmm_output)

#BUILD TRAINING AND TEST SETS FROM SAMPLES
training_sets, test_sets = BGHMM.split_obs_sets(sample_dfs)

naive_likelihood_dict = Dict()
naive_hmm = HMM(ones(1,1),[Categorical(4)])

#For all partitions, calculate the test set's probability given the naive model
@showprogress 1 "Calculating naive model test set likelihoods..." for partition in partitions
    naive_likelihood_dict[partition] = BGHMM.test_hmm(naive_hmm, test_sets[partition], 0)
end

#COMPOSE DICT OF TEST SET LIKELIHOODS GIVEN LAST HMM IN EACH CHAIN, BY JOBID
hmm_likelihoods_dict = Dict()
@showprogress 1 "Testing HMMs..." for (jobid, hmm_chain) in hmm_results_dict
    last_hmm = hmm_chain[end][2]
    partition_seqs = test_sets[jobid[1]]
    order = jobid[3]
    hmm_likelihoods_dict[jobid] = BGHMM.test_hmm(last_hmm, partition_seqs, order)
end

#COMPOSE DICT OF MAX STATE RUN LENGTH GIVEN LAST HMM DIAGONALS IN EACH CHAIN, BY JOBID
hmm_max_run_length_dict = Dict()
@showprogress 1 "Simulating run lengths..." for (jobid, hmm_chain) in hmm_results_dict
    last_hmm = hmm_chain[end][2]
    diagonal = BGHMM.get_diagonal_array(last_hmm)
    diagonal_mrls = BGHMM.sim_run_lengths(diagonal, run_samples)
    hmm_max_run_length_dict[jobid] = maximum(diagonal_mrls)
end

#SELECT BEST MODELS FOR PARTITIONS, COPY AND PARE RESULTS DICT TO CONTAIN ONLY BEST MODELS TRIPLICATES
best_dict = Dict{String,Tuple{HMM, Int64, Int64, Int64, Float64}}()
for (jobid, likelihood) in hmm_likelihoods_dict
    partition, classes, order, replicate = jobid
    if !haskey(best_dict, partition)
        best_dict[partition] = (hmm_results_dict[jobid][end][2], classes, order, replicate,likelihood)
    else
        if likelihood > best_dict[partition][4]
            best_dict[partition] = (hmm_results_dict[jobid][end][2], classes, order, replicate, likelihood)
        end
    end
end

pared_results=deepcopy(hmm_results_dict)
for (partition, (hmm, classes, order, replicate, likelihood)) in best_dict
    for (jobid, chain) in pared_results
        ipartition, iclasses, iorder, ireplicate = jobid
        if ipartition==partition
            if iclasses!=classes || iorder!=order
                delete!(pared_results,jobid)
            end
        end
    end
end

survey_results_dict=Dict() #cut down memory usage by making new dict
for (jobid, chain) in pared_results
    survey_results_dict[jobid]=chain
end

#INITIALIZE DATA MATRICES AND COMPOSE VALUES FOR PLOTTING
data_matrix_dict = Dict()
for order in order_nos
    data_matrix_dict[order] = Array{Union{Float64, Int64, Symbol}}(undef, length(Ks)*replicates, length(partitions), 5)
end
#Iterating over hmm_likelihoods_dict, compose the data matrices for plots
for (jobid, likelihood) in hmm_likelihoods_dict
    partition, K, order, replicate = jobid
    converged = hmm_results_dict[jobid][end][5]

    entry_coord = replicate + ((findfirst(isequal(K),Ks) - 1) * replicates)
    partition_series = findfirst(isequal(partition),partitions)

    data_matrix_dict[order][entry_coord,partition_series,1] = likelihood - naive_likelihood_dict[partition]
    data_matrix_dict[order][entry_coord,partition_series,2] = K
    data_matrix_dict[order][entry_coord,partition_series,3] = hmm_max_run_length_dict[jobid]

    if converged
        data_matrix_dict[order][entry_coord,partition_series,4] = :black
    else
        data_matrix_dict[order][entry_coord,partition_series,4] = :white
    end

    data_matrix_dict[order][entry_coord,partition_series,5] = length(hmm_results_dict[jobid])
end

#CALCULATE MEAN ITERATES TO CONVERGENCE
mit_dict = Dict()
for order in order_nos
    mit_dict[order] = Array{Union{Float64,Int64}}(undef, length(Ks), length(partitions), 3)
end

for order in order_nos, partition in partitions, K in Ks
    partition_series = findfirst(isequal(partition),partitions)
    entry_start = 1 + ((findfirst(isequal(K),Ks) - 1) * replicates)
    entry_range=entry_start:entry_start+replicates-1
    mit_dict[order][findfirst(isequal(K), Ks), partition_series, 1] = mean(data_matrix_dict[order][entry_range,partition_series,5])
    mit_dict[order][findfirst(isequal(K), Ks), partition_series, 2] = std(data_matrix_dict[order][entry_range,partition_series,5])
    mit_dict[order][findfirst(isequal(K), Ks), partition_series, 3] = K
end

#SETUP PLOTS
zeroOlh = scatter(Matrix{Int64}(data_matrix_dict[0][:,:,2]), title="0th order", label=["exon" "intergenic" "periexonic"], Matrix{Float64}(data_matrix_dict[0][:,:,1]),markercolors=data_matrix_dict[0][:,:,4], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="log likelihood vs naive")

zeroOrun = scatter(Matrix{Int64}(data_matrix_dict[0][(replicates+1):end,:,2]), title="0th order", label=["exon" "intergenic" "periexonic"], Matrix{Float64}(data_matrix_dict[0][(replicates+1):end,:,3]),markercolors=data_matrix_dict[0][:,:,4], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, ylims=[0, 1000], xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="model maximum mean(SRL)")

zeroOmit = plot(Matrix{Int64}(mit_dict[0][:,:,3]), Matrix{Float64}(mit_dict[0][:,:,1]), ribbon=Matrix{Float64}(mit_dict[0][:,:,2]), fillalpha=.5, title="0th order", label=["exon" "intergenic" "periexonic"], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, ylims=[0, 3000], xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="mean iterates to converge")

firstOlh = scatter(Matrix{Int64}(data_matrix_dict[1][:,:,2]), title="1st order", label=["exon" "intergenic" "periexonic"], Matrix{Float64}(data_matrix_dict[1][:,:,1]),markercolors=data_matrix_dict[1][:,:,4], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="log likelihood vs naive")

firstOrun = scatter(Matrix{Int64}(data_matrix_dict[1][(replicates+1):end,:,2]), title="1st order", label=["exon" "intergenic" "periexonic"], Matrix{Float64}(data_matrix_dict[1][(replicates+1):end,:,3]),markercolors=data_matrix_dict[1][:,:,4], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="model maximum mean(SRL)")

firstOmit = plot(Matrix{Int64}(mit_dict[1][:,:,3]), Matrix{Float64}(mit_dict[1][:,:,1]), ribbon=Matrix{Float64}(mit_dict[1][:,:,2]), fillalpha=.5, title="1st order", label=["exon" "intergenic" "periexonic"], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, ylims=[0, 1000], xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="mean iterates to converge")

secondOlh = scatter(Matrix{Int64}(data_matrix_dict[2][:,:,2]), title="2nd order", label=["exon" "intergenic" "periexonic"], Matrix{Float64}(data_matrix_dict[2][:,:,1]),markercolors=data_matrix_dict[2][:,:,4], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="log likelihood vs naive")

secondOrun = scatter(Matrix{Int64}(data_matrix_dict[2][(replicates+1):end,:,2]), title="2nd order", label=["exon" "intergenic" "periexonic"], Matrix{Float64}(data_matrix_dict[2][(replicates+1):end,:,3]),markercolors=data_matrix_dict[2][:,:,4], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="model maximum mean(SRL)")

secondOmit = plot(Matrix{Int64}(mit_dict[2][:,:,3]), Matrix{Float64}(mit_dict[2][:,:,1]), ribbon=Matrix{Float64}(mit_dict[2][:,:,2]), fillalpha=.5, title="2nd order", label=["exon" "intergenic" "periexonic"], markershape=[:circle :rect :utriangle], markersize=6, legend=:right, ylims=[0, 1000], xlims=[0.9,6.1], xticks=[1,2,4,6], xlabel="classes", ylabel="mean iterates to converge")

#SETUP FOR PARAMETRIC CONVERGENCE PLOTS
coord_dict=Dict{String,Vector{Vector{Tuple{Float64,Float64,Float64}}}}()
for (partition, (hmm, classes, order, replicate, lh)) in best_dict
    other_reps = deleteat!(collect(1:replicates), replicate)
    chains=Vector{Vector{Any}}()
    jobid=(partition, classes, order, replicate)
    chain=hmm_results_dict[jobid]
    push!(chains, chain)

    for rep in other_reps
        jobid=(partition, classes, order, rep)
        chain=hmm_results_dict[jobid]
        push!(chains, chain)
    end
    coord_dict[partition]=BGHMM.chain_3devo_coords(chains)
end


exon_traj=plot3d(coord_dict["exon"][1], lc=:purple, framestyle=:grid, title="exon convergence trajectories", legend=false)
plot3d!(coord_dict["exon"][1][end], markershape=[:circle], markercolors=[:purple])
plot3d!(coord_dict["exon"][2], lc=:green)
plot3d!(coord_dict["exon"][2][end], markershape=[:utriangle], markercolors=[:green])
plot3d!(coord_dict["exon"][3], lc=:cyan)
plot3d!(coord_dict["exon"][3][end], markershape=[:rect], markercolors=[:cyan])

inter_traj=plot3d(coord_dict["intergenic"][1], lc=:purple, framestyle=:grid, title="intergenic convergence trajectories", legend=false)
plot3d!(coord_dict["intergenic"][1][end], markershape=[:circle], markercolors=[:purple])
plot3d!(coord_dict["intergenic"][2], lc=:green)
plot3d!(coord_dict["intergenic"][2][end], markershape=[:utriangle], markercolors=[:green])
plot3d!(coord_dict["intergenic"][3], lc=:cyan)
plot3d!(coord_dict["intergenic"][3][end], markershape=[:rect], markercolors=[:cyan])

peri_traj=plot3d(coord_dict["periexonic"][1], lc=:purple, framestyle=:grid, title="periexonic convergence trajectories", legend=false)
plot3d!(coord_dict["periexonic"][1][end], markershape=[:circle], markercolors=[:purple])
plot3d!(coord_dict["periexonic"][2], lc=:green)
plot3d!(coord_dict["periexonic"][2][end], markershape=[:utriangle], markercolors=[:green])
plot3d!(coord_dict["periexonic"][3], lc=:cyan)
plot3d!(coord_dict["periexonic"][3][end], markershape=[:rect], markercolors=[:cyan])

#SETUP FOR DIAGONAL STABILITY PLOTS
selected_diagonals=Dict()

for (partition, (hmm, order, replicate, lh)) in best_dict
    chain=hmm_results_dict[(partition, length(hmm.D), order, replicate)]
    selected_diagonals[partition]=BGHMM.chain_diagonal_stability_matrix(chain)
end

xs=Dict()
xs["exon"]=zeros(Int64, size(selected_diagonals["exon"],1), size(selected_diagonals["exon"],2))
xs["intergenic"]=zeros(Int64, size(selected_diagonals["intergenic"],1), size(selected_diagonals["intergenic"],2))
xs["periexonic"]=zeros(Int64, size(selected_diagonals["periexonic"],1), size(selected_diagonals["periexonic"],2))

for (partition, x_matrix) in xs
    for state in 1:size(x_matrix,2)
        x_matrix[:,state]=collect(1:size(x_matrix,1))
    end
end

ex_stab=plot(xs["exon"], selected_diagonals["exon"], legend=false, title="exon model diagonal stability", xlabel="convergence iterate", ylabel="diagonal value")
int_stab=plot(xs["intergenic"], selected_diagonals["intergenic"], legend=false, title="intergenic model diagonal stability", xlabel="convergence iterate", ylabel="diagonal value")
peri_stab=plot(xs["periexonic"], selected_diagonals["periexonic"], legend=false, title="periexonic model diagonal stability", xlabel="convergence iterate", ylabel="diagonal value")

plots_to_print = Dict("zeroOlh"=>zeroOlh, "zeroOrun"=>zeroOrun, "zeroOmit"=>zeroOmit,"firstOlh"=>firstOlh, "firstOrun"=>firstOrun, "firstOmit"=>firstOmit,"secondOlh"=>secondOlh, "secondOrun"=>secondOrun,"secondOmit"=>secondOmit,"exontraj"=>exon_traj,"peritraj"=>peri_traj,"intertraj"=>inter_traj,"exonstab"=>ex_stab,"intergenicstab"=>int_stab,"perigenicstab"=>peri_stab)

for (filename, plot) in plots_to_print
    png(plot, filename)
end

#SERIALIZE SELECTED MODELS FOR LATER USE
serialize(selected_hmm_output, best_dict)
serialize(survey_results_path, survey_results_dict)