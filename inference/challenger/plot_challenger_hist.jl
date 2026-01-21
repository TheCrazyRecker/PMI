include("challenger_inference.jl")
using Measures

n_iter = 2000

# is_samples  = run_is_challenger(n_iter)
pf_samples  = run_pf_challenger(n_iter)
cs_samples = run_custom_inference_challenger(n_iter)

# is_alpha,  is_beta  = split_samples(is_samples)
pf_alpha,  pf_beta  = split_samples(pf_samples)
cs_alpha, cs_beta = split_samples(cs_samples)

# p_is_alpha = histogram(
#     is_alpha,
#     bins=40,
#     normalize=:pdf,
#     alpha=0.6,
#     label="IS",
#     xlabel="alpha",
#     ylabel="density",
#     title="IS: Posterior Density of Alpha",
#     margin=10mm
# )
# vline!(p_is_alpha, [true_alpha], linestyle=:dash, label="true")

# p_is_beta = histogram(
#     is_beta,
#     bins=40,
#     normalize=:pdf,
#     alpha=0.6,
#     label="IS",
#     xlabel="beta",
#     ylabel="density",
#     title="IS: Posterior Density of Beta",
#     margin=10mm
# )
# vline!(p_is_beta, [true_beta], linestyle=:dash, label="true")

# display(plot(p_is_alpha, p_is_beta, layout=(1,2), size=(900,400)))

p_pf_alpha = histogram(
    pf_alpha,
    bins=40,
    normalize=:pdf,
    alpha=0.6,
    label="PF",
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
    label="PF",
    xlabel="beta",
    ylabel="density",
    title="PF: Posterior Density of Beta"
)
vline!(p_pf_beta, [true_beta], linestyle=:dash, label="true")

display(plot(p_pf_alpha, p_pf_beta, layout=(1,2), size=(900,400)))

# p_cs_alpha = histogram(
#     cs_alpha,
#     bins=40,
#     normalize=:pdf,
#     alpha=0.6,
#     label="Custom",
#     xlabel="alpha",
#     ylabel="density",
#     title="Custom: Posterior Density of Alpha"
# )
# vline!(p_cs_alpha, [true_alpha], linestyle=:dash, label="true")

# p_cs_beta = histogram(
#     cs_beta,
#     bins=40,
#     normalize=:pdf,
#     alpha=0.6,
#     label="Custom",
#     xlabel="beta",
#     ylabel="density",
#     title="Custom: Posterior Density of Beta"
# )
# vline!(cs_beta, [true_beta], linestyle=:dash, label="true")

# display(plot(p_cs_alpha, p_cs_beta, layout=(1,2), size=(900,400)))