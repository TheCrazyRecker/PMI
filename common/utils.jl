using Statistics
using Random
using Distributions

#Used for challenger model to generate more data using a normal distribution with mean and variance
#based on the original 23 flight data
function sample_temps_from_data(temps::Vector{Int}, n_samples::Int;
                                rng::AbstractRNG=Random.default_rng())

    mean_temp = mean(temps)
    var_temp  = var(temps)

    dist = Normal(mean_temp, sqrt(var_temp))
    samples = rand(rng, dist, n_samples)

    samples_int = round.(Int, samples)

    return samples_int
end

#Used to show convergence to true value for continous variables (alpha, beta, lambda1, lambda2)
function running_mean(x)
    s = 0.0
    out = similar(x, Float64)
    for i in eachindex(x)
        s += x[i]
        out[i] = s / i
    end
    out
end

#Converts input data to choicemap
function vector_to_choicemap(data_vector; prefix::Symbol=:failure)
    cm = choicemap()
    for (i, val) in enumerate(data_vector)
        cm[(prefix, i)] = val
    end
    return cm
end


#Gets alpha and beta
function split_samples(samples)
    a = [s[1] for s in samples]
    b = [s[2] for s in samples]
    return a, b
end

#More generic method that gets certain param from tuple
function extract_param(traces, addr)
    return [tr[addr] for tr in traces]
end

#Next methods are for making the plots look nicer