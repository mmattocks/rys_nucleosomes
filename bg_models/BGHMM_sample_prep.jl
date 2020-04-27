#JOB FILEPATHS
#GFF3 feature database, FASTA genome and index paths
danio_gff_path = "/bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3"
danio_genome_path = "/bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna"
danio_gen_index_path = "/bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna.fai"
#sample record and hmm serialisation output path
sample_output = "/bench/PhD/NGS_binaries/BGHMM/BGHMM_samples"
#non registered libs

#GENERAL SETUP
@info "Loading libraries..."
using BioSequences, Distributed, GenomicFeatures, ProgressMeter, Serialization

#JOB CONSTANTS
#CONSTANTS FOR GENOMIC SAMPLING
const sample_set_length = Int64(12e6)
const sample_window_min = 10
const sample_window_max = 3000
const perigenic_pad = 500
const partitions = 3 #exonic, periexonic, intragenic

#Setup sampling workers
@info "Spawning workers..."
pool_size = partitions                          #number of workers to use
worker_pool = addprocs(pool_size) # add processes up to the worker pool size + 1 control process
@everywhere using BGHMM, DataFrames, Random
@everywhere Random.seed!(1)

#GET SEQUENCE OBSERVATIONS TO TRAIN AND TEST MODELS - PARTITIONING GENOME INTO EXONIC, PERIEXONIC, INTRAGENIC SEQUENCE
@info "Building observation db..."
#generate the genomic sample dataframe
if isfile(sample_output) #if the sample DB has already been built for the current project, terminate
    @error "Existing sample dataframe at $sample_output"
else #otherwise, build it from scratch
    @info "Setting up sampling jobs.."
    #setup worker channels, input gets the genome partitions
    input_sample_channel, completed_sample_channel = BGHMM.setup_sample_jobs(danio_genome_path, danio_gen_index_path, danio_gff_path, sample_set_length, sample_window_min, sample_window_max, perigenic_pad)
    progress_channel = RemoteChannel(()->Channel{Tuple}(20))

    #send sampling jobs to workers
    if isready(input_sample_channel) > 0
        @info "Sampling.."
        #WORKERS SAMPLE
        for worker in worker_pool[1:partitions]
            remote_do(BGHMM.get_sample_set, worker, input_sample_channel, completed_sample_channel, progress_channel)
        end
    else
        @error "No sampling jobs!"
    end

    #progress meters for sampling
    sampling_meters=Dict{String, Progress}()
    overall_sampling_meter=Progress(partitions,"Overall sampling progress:")
    completed_counter = 0
    ProgressMeter.update!(overall_sampling_meter, completed_counter)
    sampling_offset = ones(Bool, partitions)

    #collect progress updates while waiting on completion of sampling jobs
    while completed_counter < partitions
        wait(progress_channel)
        partition_id, progress = take!(progress_channel)
        if haskey(sampling_meters, partition_id)
            ProgressMeter.update!(sampling_meters[partition_id], progress)
        else
            offset = findfirst(sampling_offset)[1]
            sampling_meters[partition_id] = Progress(sample_set_length, "Sampling partition $partition_id:", offset)
            ProgressMeter.update!(sampling_meters[partition_id], progress)
            sampling_offset[offset] = false
        end
        if progress == sample_set_length
            @info "job done"
            global completed_counter += 1
            ProgressMeter.update!(overall_sampling_meter, completed_counter)
        end
    end

    #collect sample dfs by partition id when ready
    sample_record_dfs = Dict{String,DataFrame}()
    collected_counter = 0
    while collected_counter < partitions
        wait(completed_sample_channel)
        partition_id, sample_df = take!(completed_sample_channel)
        sample_record_dfs[partition_id] = sample_df
        global collected_counter += 1
        @info "Partition $partition_id completed sampling..."
    end
    @info "Exporting sample dataframes to $sample_output..."
    serialize(sample_output, sample_record_dfs) #write the dataframes to file
end

@info "Done sampling!"
