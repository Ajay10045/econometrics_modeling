using CSV
using DataFrames
using MixedModels
using StatsModels
using CategoricalArrays

"""
    mixed_model_fn(data_path::String, formula_str::String)

Fits a mixed effects model using `MixedModels.jl` given a CSV path, a formula string, 
and a list of grouping variables (to be treated as categorical). Returns model components.
"""
function mixed_model_fn(data_path::String, formula_str::String)

    println("🔍 Reading data from: ", data_path)
    df = CSV.read(data_path, DataFrame)

    # println("📊 Converting grouping variables to categorical: ", group_vars)
    # for col in group_vars
    #     if col in names(df)
    #         df[!, col] = CategoricalArray(df[!, col])
    #     else
    #         error("❌ Column $(col) not found in dataset.")
    #     end
    # end

    println("🧮 Parsing formula string as raw formula expression")
    fm = eval(Meta.parse(formula_str))  # <-- formula_str must NOT include @formula(...)

    println("📐 Parsed formula: ", fm)

    println("🏗️  Fitting model...")
    model = fit(MixedModel, fm, df)
    println("✅ Model fit complete.")

    # Extract results
    residuals = residuals(model)
    predictions = predict(model)
    rand_eff = DataFrame(ranef(model))
    ct = coeftable(model)

    effect_names = ct.rownms
    estimates = ct.cols[1]
    std_errs = ct.cols[2]
    z_vals = ct.cols[3]
    p_vals = ct.cols[4]

    variance_components = VarCorr(model)
    dof = dof_residual(model)

    return residuals, predictions, rand_eff, effect_names, estimates, std_errs, z_vals, p_vals, variance_components, dof
end
