include("text_inference_v2.jl")
include("../../common/utils.jl")

cs = custom_text_inference_mh_gibbs(observations; niter=20_000, burnin=5_000, thin=10)

# p_cs_tau = tau_pmf_plot(cs.taus, N; title="Custom: τ PMF")
# display(p_cs_tau)

tau_plot = plot(cs.taus, title="Running mean of tau", xlabel="Iteration / Sample Index", ylabel="E[tau]", label="Custom")
