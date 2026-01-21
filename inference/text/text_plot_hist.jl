include("text_inference_v2.jl")
include("../../models/text.jl")

n_iter = 10000

is_traces = run_is_text(n_iter)
pf_traces = run_pf_text(n_iter)

is_tau = extract_param(is_traces, :tau)
is_l1  = extract_param(is_traces, :lambda1)
is_l2  = extract_param(is_traces, :lambda2)

pf_tau = extract_param(pf_traces, :tau)
pf_l1  = extract_param(pf_traces, :lambda1)
pf_l2  = extract_param(pf_traces, :lambda2)

#Helpers for nicer plots

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

xl1 = shared_xlim(is_l1, pf_l1)
xl2 = shared_xlim(is_l2, pf_l2)

# ---------- Plots ----------

p_is_tau = tau_pmf_plot(is_tau, N; title="IS: τ PMF")
p_is_l1  = lambda_hist_plot(is_l1; title="IS: λ1 posterior", xlabel="lambda1", truth=true_l1, xlims=xl1)
p_is_l2  = lambda_hist_plot(is_l2; title="IS: λ2 posterior", xlabel="lambda2", truth=true_l2, xlims=xl2)

p_pf_tau = tau_pmf_plot(pf_tau, N; title="PF: τ PMF")
p_pf_l1  = lambda_hist_plot(pf_l1; title="PF: λ1 posterior", xlabel="lambda1", truth=true_l1, xlims=xl1)
p_pf_l2  = lambda_hist_plot(pf_l2; title="PF: λ2 posterior", xlabel="lambda2", truth=true_l2, xlims=xl2)

combined = plot(p_is_tau, p_is_l1, p_is_l2,
                p_pf_tau, p_pf_l1, p_pf_l2,
                layout=(2,3), size=(1100, 550), margin=5Plots.mm)

display(combined)
