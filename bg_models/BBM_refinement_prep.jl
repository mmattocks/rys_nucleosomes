#JOB FILEPATHS
#GFF3 feature database, FASTA genome and index paths
danio_gff_path = "/bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3"
danio_genome_path = "/bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna"
danio_gen_index_path = "/bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna.fai"
#sample record and hmm serialisation output path
sample_output = "/bench/PhD/NGS_binaries/BBM/refinement_samples"
#non registered libs

#GENERAL SETUP
@info "Loading libraries..."
using Distributed,Serialization

#JOB CONSTANTS
#CONSTANTS FOR GENOMIC SAMPLING
const sample_set_length = Int64(16e6)
const sample_window_min = 10
const sample_window_max = 3000
const perigenic_pad = 500
const partitions = 3 #exonic, periexonic, intragenic

#Setup sampling workers
@info "Spawning workers..."
pool_size = partitions                          #number of workers to use
worker_pool = addprocs(pool_size) # add processes up to the worker pool size + 1 control process
@everywhere using BioBackgroundModels, Random
@everywhere Random.seed!(1)

#GET SEQUENCE OBSERVATIONS TO TRAIN AND TEST MODELS - PARTITIONING GENOME INTO EXONIC, PERIEXONIC, INTRAGENIC SEQUENCE
@info "Building observation db..."
#generate the genomic sample dataframe
if isfile(sample_output) #if the sample DB has already been built for the current project, terminate
    @error "Existing sample dataframe at $sample_output"
else #otherwise, build it from scratch
    @info "Setting up sampling jobs..."
    #setup worker channels, input gets the genome partitions
    channels = setup_sample_jobs(danio_genome_path, danio_gen_index_path, danio_gff_path, sample_set_length, sample_window_min, sample_window_max, perigenic_pad)

    @info "Sampling genome..."
    sample_record_dfs=execute_sample_jobs(channels, worker_pool)

    @info "Serializing sample dataframes..."
    serialize(sample_output, sample_record_dfs) #write the dataframes to file
end

rmprocs(worker_pool)

@info "Done sampling jobs!"
