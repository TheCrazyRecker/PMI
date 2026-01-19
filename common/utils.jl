using Statistics
using Random
using Distributions

function sample_temps_from_data(temps::Vector{Int}, n_samples::Int;
                                rng::AbstractRNG=Random.default_rng())

    mean_temp = mean(temps)
    var_temp  = var(temps)

    dist = Normal(mean_temp, sqrt(var_temp))
    samples = rand(rng, dist, n_samples)

    samples_int = round.(Int, samples)

    return samples_int
end

function running_mean(x)
    s = 0.0
    out = similar(x, Float64)
    for i in eachindex(x)
        s += x[i]
        out[i] = s / i
    end
    out
end