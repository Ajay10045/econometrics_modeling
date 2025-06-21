using CSV
using DataFrames
using MixedModels

"""
    mixed_model_fn(data_path::String, formula::String)

Fit a mixed model using `MixedModels.jl` and return relevant statistics.
"""
function mixed_model_fn(data_path::String, formula::String)
    df = CSV.read(data_path, DataFrame)
    fm = eval(Meta.parse(formula))
    model = fit(MixedModel, fm, df)

    residuals = residuals(model)
    pred = predict(model)

    rand_eff = DataFrame(ranef(model))
    ct = coeftable(model)
    effect = ct.rownms
    estimate = ct.cols[1]
    stderr = ct.cols[2]
    z_value = ct.cols[3]
    p_value = ct.cols[4]

    var = VarCorr(model)
    dof = dof_residual(model)

    return residuals, pred, rand_eff, effect, estimate, stderr, z_value, p_value, var, dof
end
