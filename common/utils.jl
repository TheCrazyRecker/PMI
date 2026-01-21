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

#more general
function extract_param(traces, addr)
    return [tr[addr] for tr in traces]
end

#Plotting helper
# Discrete PMF bar plot for tau (much nicer than histogram for an integer latent)
function tau_pmf_plot(tau_samples::Vector{Int}, N::Int; title="")
    counts = zeros(Int, N)
    for τ in tau_samples
        if 1 <= τ <= N
            counts[τ] += 1
        end
    end
    probs = counts ./ sum(counts)

    p = bar(1:N, probs;
        xlabel="tau",
        ylabel="Probability",
        title=title,
        legend=false,
        xlims=(1, N),
        ylims=(0, 1.05*maximum(probs))
    )
    vline!(p, [true_tau], linestyle=:dash, label=nothing)
    return p
end

# Consistent histogram for continuous lambdas
function lambda_hist_plot(x::Vector{<:Real}; title="", xlabel="", truth=nothing, xlims=nothing)
    p = histogram(x;
        bins=50,
        normalize=true,
        xlabel=xlabel,
        ylabel="Density",
        title=title,
        legend=false
    )
    if xlims !== nothing
        plot!(p, xlims=xlims)
    end
    if truth !== nothing
        vline!(p, [truth], linestyle=:dash, label=nothing)
    end
    return p
end

# Choose shared x-limits for lambda plots (IS vs PF comparable)
function shared_xlim(a::Vector{<:Real}, b::Vector{<:Real}; pad_frac=0.15)
    lo = min(minimum(a), minimum(b))
    hi = max(maximum(a), maximum(b))
    if hi == lo
        # handle extreme degeneracy (all samples identical)
        lo -= 1
        hi += 1
    else
        pad = pad_frac * (hi - lo)
        lo -= pad
        hi += pad
    end
    return (lo, hi)
end