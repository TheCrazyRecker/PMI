include("challenger_inference.jl")
using Measures

n_iter = 2000

is_samples  = run_is_challenger(n_iter)
pf_samples  = run_pf_challenger(n_iter)

function split_samples(samples)
    a = [s[1] for s in samples]
    b = [s[2] for s in samples]
    return a, b
end

is_alpha,  is_beta  = split_samples(is_samples)
pf_alpha,  pf_beta  = split_samples(pf_samples)

println("Generating histogram plots for IS and PF...")

# -------------------------
# Importance Sampling
# -------------------------

p_is_alpha = histogram(
    is_alpha,
    bins=40,
    normalize=:pdf,
    alpha=0.6,
    label="IS",
    xlabel="alpha",
    ylabel="density",
    title="IS: Posterior Density of Alpha",
    margin=10mm
)
vline!(p_is_alpha, [true_alpha], linestyle=:dash, label="true")

p_is_beta = histogram(
    is_beta,
    bins=40,
    normalize=:pdf,
    alpha=0.6,
    label="IS",
    xlabel="beta",
    ylabel="density",
    title="IS: Posterior Density of Beta",
    margin=10mm
)
vline!(p_is_beta, [true_beta], linestyle=:dash, label="true")

# display(plot(p_is_alpha, p_is_beta, layout=(1,2), size=(900,400)))

p_pf_alpha = histogram(
    pf_alpha,
    bins=40,
    normalize=:pdf,
    alpha=0.6,
    label="PF / SMC",
    xlabel="alpha",
    ylabel="density",
    title="PF: Posterior Density of Alpha"
)
vline!(p_pf_alpha, [true_alpha], linestyle=:dash, label="true")

p_pf_beta = histogram(
    pf_beta,
    bins=40,
    normalize=:pdf,
    alpha=0.6,
    label="PF / SMC",
    xlabel="beta",
    ylabel="density",
    title="PF: Posterior Density of Beta"
)
vline!(p_pf_beta, [true_beta], linestyle=:dash, label="true")

display(plot(p_pf_alpha, p_pf_beta, layout=(1,2), size=(900,400)))