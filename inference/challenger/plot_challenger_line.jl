include("challenger_inference.jl")
include("../../common/utils.jl")

n_iter = 2000

cst_samples = run_custom_inference_challenger(n_iter)
mh_samples  = run_mh_rw_challenger(n_iter)
hmc_samples = run_hmc_challenger(n_iter)

cst_alpha, cst_beta = split_samples(cst_samples)
mh_alpha,  mh_beta  = split_samples(mh_samples)
hmc_alpha, hmc_beta = split_samples(hmc_samples)

rm_cst_alpha = running_mean(cst_alpha)
rm_cst_beta = running_mean(cst_beta)

rm_mh_alpha  = running_mean(mh_alpha)
rm_mh_beta  = running_mean(mh_beta)

rm_hmc_alpha = running_mean(hmc_alpha)
rm_hmc_beta = running_mean(hmc_beta)


mean_alpha_plot = plot(rm_cst_alpha, title="Running mean of alpha", xlabel="Iteration", ylabel="E[alpha]",label="Custom")
plot!(mean_alpha_plot, rm_mh_alpha,  label="MH")
plot!(mean_alpha_plot, rm_hmc_alpha, label="HMC")
hline!(mean_alpha_plot, [true_alpha], linestyle=:dash, label="true")

mean_beta_plot = plot(rm_cst_beta, title="Running mean of beta", xlabel="Iteration", ylabel="E[beta]",label="Custom")
plot!(mean_beta_plot, rm_mh_beta,  label="MH")
plot!(mean_beta_plot, rm_hmc_beta, label="HMC")
hline!(mean_beta_plot, [true_beta], linestyle=:dash, label="true")

combined_means_plot = plot(mean_alpha_plot, mean_beta_plot, layout=(2, 1), size=(900, 1000))
display(combined_means_plot)