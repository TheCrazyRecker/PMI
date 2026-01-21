include("text_inference_v2.jl")
include("../../common/utils.jl")

custom = custom_text_inference_mh_gibbs(observations; niter=20_000, burnin=5_000, thin=10)
n_iter = min(2000, length(custom.taus))

c_tau = custom.taus[1:n_iter]
c_l1  = custom.lambda1[1:n_iter]
c_l2  = custom.lambda2[1:n_iter]

hmc_traces = run_hmc_text(n_iter)
hmc_l1  = extract_param(hmc_traces, :lambda1)
hmc_l2  = extract_param(hmc_traces, :lambda2)

rm_c_l1  = running_mean(c_l1);  
rm_c_l2  = running_mean(c_l2);      

rm_hmc_l1  = running_mean(hmc_l1);   
rm_hmc_l2  = running_mean(hmc_l2);   

l1_plot = plot(rm_c_l1, title="Running mean of lambda1", xlabel="Iteration / Sample Index", ylabel="E[lambda1]", label="Custom")
plot!(l1_plot, rm_hmc_l1, label="HMC")
hline!(l1_plot, [true_lambda1], linestyle=:dash, label="true")


l2_plot = plot(rm_c_l2, title="Running mean of lambda2", xlabel="Iteration / Sample Index", ylabel="E[lambda2]", label="Custom")
plot!(l2_plot, rm_hmc_l2, label="HMC")
hline!(l2_plot, [true_lambda2], linestyle=:dash, label="true")

c_hist = histogram(
    c_tau;
    bins=tau_bins,
    normalize=true,
    xlabel="tau",
    ylabel="Probability",
    title="Custom: τ distribution",
    label=nothing
)

display(c_hist)

# combined = plot(l1_plot, l2_plot, layout=(2, 1), size=(900, 1200))
# display(combined)
