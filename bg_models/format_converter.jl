function convert_format(jobid, jobchain)
    id=jobid
    try
        id=BioBackgroundModels.Chain_ID(jobid...)
    catch e
        println("$jobid cannot be converted to valid Chain_ID")
        println(e)
    end

    chain=Vector{BioBackgroundModels.EM_step}()
    for step in jobchain
        old_hmm = step[2]
        new_hmm = BioBackgroundModels.HMM(old_hmm.a, old_hmm.A, old_hmm.B)
        em_step=BioBackgroundModels.EM_step(step[1],new_hmm,float(step[3]),float(step[4]),step[5])
        push!(chain, em_step)
    end

    return id,chain
end

function convert_results_dict(results_dict)
    new_dict=Dict{BioBackgroundModels.Chain_ID,Vector{BioBackgroundModels.EM_step}}()
    for entry in results_dict
        id,chain=convert_format(entry...)
        new_dict[id]=chain
    end
    return new_dict
end

