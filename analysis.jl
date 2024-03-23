# Load necessary packages
using DataFrames, CSV, GLM, ROCAnalysis, Statistics

# Read the CSV file
df = CSV.read("/Data.csv", DataFrame, normalizenames=true)

# Correcting the selection of columns by excluding some with Cols and Not
df = select(df, Not(Cols(1:3, 11, 15, 16, 26, 29, 35, 43, 44)))

# Creating a new variable based on a condition
df.SDNNCat = ifelse.(df.SDNN .< 32, 1, 0)

# Counting the occurrences of each category in SDNNCat
counts = combine(groupby(df, :SDNNCat), nrow => :count)

# Filtering rows based on a condition
dfob = filter(row -> row["WEIGHT_STATUS"] == "HIGHER", df)
dfthin = filter(row -> row["WEIGHT_STATUS"] == "NORMAL WEIGHT" || row["WEIGHT_STATUS"] == "LOWER", df)
dfthin = filter(row -> row["WEIGHT_STATUS"] == "NORMAL WEIGHT", df)

# Group dfs by the 'DIABETES' column
groups = groupby(dfob, :DIABETES)
groups1 = groupby(dfthin, :DIABETES)

# Calculate summary statistics for the 'SDNN' column for each group
grouped_summary_SDNN = combine(groups, 
                               :SDNN => mean => :mean_SDNN,
                               :SDNN => std => :std_SDNN,
                               :SDNN => median => :median_SDNN,
                               :SDNN => minimum => :min_SDNN,
                               :SDNN => maximum => :max_SDNN)

grouped_summary_SDNN1 = combine(groups1, 
                                :SDNN => mean => :mean_SDNN,
                                :SDNN => std => :std_SDNN,
                                :SDNN => median => :median_SDNN,
                                :SDNN => minimum => :min_SDNN,
                                :SDNN => maximum => :max_SDNN)

# Display the grouped summary statistics for SDNN
println(grouped_summary_SDNN)
println(grouped_summary_SDNN1)

# Fitting univariate generalized linear models
mod1 = glm(@formula(SDNNCat ~ DIABETES), dfob, Binomial(), LogitLink())
mod2 = glm(@formula(SDNNCat ~ DIABETES), dfthin, Binomial(), LogitLink())

# Print models' summary
println(mod1)
println(mod2)

# Get odds ratios
println(exp(coef(mod1)[2]))
println(exp(coef(mod2)[2]))

println(median(filter(row -> row.DIABETES == "YES", dfob).SDNN))
println(median(filter(row -> row.DIABETES == "NO", dfob).SDNN))
println(median(filter(row -> row.DIABETES == "YES", dfthin).SDNN))
println(median(filter(row -> row.DIABETES == "NO", dfthin).SDNN))

# Fitting generalized linear models adjusted for age
mod3 = glm(@formula(SDNNCat ~ DIABETES + AGE), dfob, Binomial(), LogitLink())
mod4 = glm(@formula(SDNNCat ~ DIABETES + AGE), dfthin, Binomial(), LogitLink())

# Print models' summary
println(mod3)
println(mod4)

# Get odds ratios
println(exp(coef(mod3)[2]))
println(exp(coef(mod4)[2]))

# Fitting generalized linear models adjusted for age and gender
mod5 = glm(@formula(SDNNCat ~ DIABETES + AGE + GENDER), dfob, Binomial(), LogitLink())
mod6 = glm(@formula(SDNNCat ~ DIABETES + AGE + GENDER), dfthin, Binomial(), LogitLink())

# Print models' summary
println(mod5)
println(mod6)

# Get odds ratios
println(exp(coef(mod5)[2]))
println(exp(coef(mod6)[2]))

# Predict probabilities for both models
probabilities_mod5 = predict(mod5, dfob)
probabilities_mod6 = predict(mod6, dfthin)

# Choose a threshold to classify predictions
threshold = 0.5
predictions_mod5 = ifelse.(probabilities_mod5 .> threshold, 1, 0)
predictions_mod6 = ifelse.(probabilities_mod6 .> threshold, 1, 0)

# Actual outcomes
actuals_mod5 = dfob.SDNNCat
actuals_mod6 = dfthin.SDNNCat

# Calculate confusion matrix components for mod5
TP_mod5 = sum((predictions_mod5 .== 1) .& (actuals_mod5 .== 1))
TN_mod5 = sum((predictions_mod5 .== 0) .& (actuals_mod5 .== 0))
FP_mod5 = sum((predictions_mod5 .== 1) .& (actuals_mod5 .== 0))
FN_mod5 = sum((predictions_mod5 .== 0) .& (actuals_mod5 .== 1))

# Calculate metrics for mod5
accuracy_mod5 = (TP_mod5 + TN_mod5) / length(predictions_mod5)
precision_mod5 = TP_mod5 / (TP_mod5 + FP_mod5)
sensitivity_mod5 = TP_mod5 / (TP_mod5 + FN_mod5)
specificity_mod5 = TN_mod5 / (TN_mod5 + FP_mod5)

println(accuracy_mod5)
println(precision_mod5)
println(sensitivity_mod5)
println(specificity_mod5)

# Calculate confusion matrix components for mod6
TP_mod6 = sum((predictions_mod6 .== 1) .& (actuals_mod6 .== 1))
TN_mod6 = sum((predictions_mod6 .== 0) .& (actuals_mod6 .== 0))
FP_mod6 = sum((predictions_mod6 .== 1) .& (actuals_mod6 .== 0))
FN_mod6 = sum((predictions_mod6 .== 0) .& (actuals_mod6 .== 1))

# Calculate metrics for mod6
accuracy_mod6 = (TP_mod6 + TN_mod6) / length(predictions_mod6)
precision_mod6 = TP_mod6 / (TP_mod6 + FP_mod6)
sensitivity_mod6 = TP_mod6 / (TP_mod6 + FN_mod6)
specificity_mod6 = TN_mod6 / (TN_mod6 + FP_mod6)

println(accuracy_mod6)
println(precision_mod6)
println(sensitivity_mod6)
println(specificity_mod6)

# AUC-ROC for mod5
roc_mod5 = roc(actuals_mod5, probabilities_mod5)
auc_roc_mod5 = auc(roc_mod5)

# AUC-ROC for mod6
roc_mod6 = roc(actuals_mod6, probabilities_mod6)
auc_roc_mod6 = auc(roc_mod6)

println("AUC-ROC for mod5: ", auc_roc_mod5)
println("AUC-ROC for mod6: ", auc_roc_mod6)

