# Uncomment this section the first time running the program
using Pkg
# Pkg.add("ADNLPModels")
# Pkg.add("NLPModels")
# Pkg.add("LinearAlgebra")
# Pkg.add("NLPModelsIpopt")
# Pkg.add("HTTP")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("Dates")
# Pkg.add("JSOSolvers")
# Pkg.add("SolverBenchmark")
# Pkg.add("Plots")

using LinearAlgebra
using ADNLPModels
using NLPModels
using NLPModelsIpopt
using JSOSolvers
using HTTP
using CSV
using DataFrames
using CSV
using Dates
using SolverBenchmark
using Plots


#**********************************************************************************************************************
# Function to evaluate the values in a given line in matrix A of the model
function solution(x)
    g = d -> x[1]+x[2]*d+x[3]*cos(2*pi*d/(365.25))+x[4]*sin(2*pi*d/(365.25))+x[5]*cos(2*pi*d/(10.7*365.25))+x[6]*sin(2*pi*d/(10.7*365.25))
    return g
end

#**********************************************************************************************************************
# Importation des données de températures dans un vecteur b
# URL of the data file
url = "https://vanderbei.princeton.edu/ampl/nlmodels/LocalWarming/McGuireAFB/data/McGuireAFB.dat"

# Download data from the URL
response = HTTP.get(url)
data = String(response.body)

# Parse the data into a DataFrame
df = CSV.File(IOBuffer(data), delim=' ', header=false, ignorerepeated=true) |> DataFrame

# Extract date and temperature columns
dates = df[!, 1]
temperatures = df[!, 2]

# Create a matrix with date and temperature
data_matrix = hcat(dates, temperatures)

# # Print the matrix
# println(data_matrix)

# Extract dates and temperatures vectors
dates_vector = data_matrix[:, 1]
temperatures_vector = data_matrix[:, 2]


# Imnport des données de Montréal
# Chemins des fichiers CSV
file_paths = ["climate-daily1.csv", "climate-daily2.csv", "climate-daily3.csv"]

# Initialisation des vecteurs de température et de date
temperatures_mtl = Float64[]
dates_mtl = Vector{String}()

# Parcourir chaque fichier CSV
for file_path in file_paths
    # Charger le fichier CSV dans un DataFrame Julia
    df = CSV.read(file_path, DataFrame)

    # Extraire les colonnes de température et de date
    temperature_column = df[:, "MEAN_TEMPERATURE"]
    date_column = df[:, "LOCAL_DATE"]

    # Filtrer les valeurs manquantes dans les colonnes de température et de date
    non_missing_indices = .!ismissing.(temperature_column) .& .!ismissing.(date_column)
    temperature_column_filtered = temperature_column[non_missing_indices]
    date_column_filtered = date_column[non_missing_indices]

    # Ajouter les valeurs aux vecteurs respectifs
    append!(temperatures_mtl, temperature_column_filtered)
    append!(dates_mtl, date_column_filtered)
end

# Formater les dates au format yyyymmdd
dates_mtl = Dates.format.(Date.(dates_mtl, "yyyy-mm-dd HH:MM:SS"), "yyyymmdd")

# Afficher les 10 premières valeurs des vecteurs (pour vérification)
println("10 premieres températures Montréal :", temperatures_mtl[1:10])
println("10 premieres dates Montréal :", dates_mtl[1:10])

b = temperatures_vector # temperature vector


#**********************************************************************************************************************
r = 6
n = length(b)

# Déclaration de la matrice A selon le nombre de données de températures
A = ones(n,r)

for i in 1:n
    A[i,2] = i
    A[i,3] = cos(2*pi*i/(365.25))
    A[i,4] = sin(2*pi*i/(365.25))
    A[i,5] = cos(2*pi*i/(10.7*365.25))
    A[i,6] = sin(2*pi*i/(10.7*365.25))
end

#**********************************************************************************************************************
# Déclaration des fonctions des problèmes ADNLP et ADNLS
F(x) = A*x-b        # residual function
f(x) = sum(A*x-b)         # Fonction
x0 = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1] # first guess

# Modèle ADNLS et résolution ipopt
nlps_lsquare = ADNLSModel(F, x0, n; constraints=Dict("C1" => λ -> abs(A * λ - b) <= A * λ - b))
stats_s = ipopt(nlps_lsquare)
print(stats_s.solution)

# Modèle ADNLP et résolution ipopt
nlpp_lsquare = ADNLPModel(f, x0; constraints=Dict("C1" => λ -> abs(A * λ - b) <= A * λ - b))
stats_p = ipopt(nlpp_lsquare)
print(stats_p.solution)

#**********************************************************************************************************************
# Représentation graphique des données de température vs les courbes obtenues
x = ones(n,1)
for i in 1:n
    x[i,1] = i
end
lp_mod = solution(stats_p.solution)
y_lp = ones(n,1)
for i in 1:n
    y_lp[i,1] = lp_mod(x[i])
end
ls_mod = solution(stats_s.solution)
y_ls = ones(n,1)
for i in 1:n
    y_ls[i,1] = ls_mod(x[i])
end


plot(x, b, label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x, y_lp, label="Modèle ADNLP", linecolor="red")
plot!(x, y_ls, label="Modèle ADNLS", linecolor="blue")
savefig("two_models_alldays_MTL.svg")
savefig("two_models_alldays_MTL.png")


start_plot = 10000
end_plot = 12000
plot(x[start_plot:end_plot], b[start_plot:end_plot], label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x[start_plot:end_plot], y_lp[start_plot:end_plot], label="Modèle ADNLP", linecolor="red")
plot!(x[start_plot:end_plot], y_ls[start_plot:end_plot], label="Modèle ADNLS", linecolor="blue")
savefig("two_models_day$(start_plot)_to_day$(end_plot)_MTL.svg")
savefig("two_models_day$(start_plot)_to_day$(end_plot)_MTL.png")



#**********************************************************************************************************************
# Résultats du modèle NLP avec d'autres solveurs et printing des performances
print("NLP Model results\n")
print("Performances ipopt\n")
resultP_ipopt = [stats_p.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; stats_p.elapsed_time]
print(resultP_ipopt)

print("\nSolution lbfgs\n")
statsP_lbfgs = lbfgs(nlpp_lsquare)
print(statsP_lbfgs.solution)
print("\nPeformances lbfgs\n")
resultP_lbfgs = [statsP_lbfgs.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; statsP_lbfgs.elapsed_time]
print(resultP_lbfgs)

print("\nSolution tron\n")
statsP_tron = tron(nlpp_lsquare)
print(statsP_tron.solution)
print("\nPeformances tron\n")
resultP_tron = [statsP_tron.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; statsP_tron.elapsed_time]
print(resultP_tron)

print("\nSolution trunk\n")
statsP_trunk = trunk(nlpp_lsquare)
print(statsP_trunk.solution)
print("\nPeformances trunk\n")
resultP_trunk = [statsP_trunk.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; statsP_trunk.elapsed_time]
print(resultP_trunk)

print("\nSolution R2\n")
statsP_R2 = R2(nlpp_lsquare)
print(statsP_R2.solution)
print("\nPeformances R2\n")
resultP_R2 = [statsP_R2.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; statsP_R2.elapsed_time]
print(resultP_R2)
print(statsP_R2.status)

#**********************************************************************************************************************
# Résultats du modèle NLS avec d'autres solveurs et printing des performances
print("\nNLS Model results\n")
print("\nPerformances ipopt\n")
resultS_ipopt = [stats_s.iter ; nlps_lsquare.counters.neval_obj ; nlps_lsquare.counters.neval_grad ; nlps_lsquare.counters.neval_hess ; stats_s.elapsed_time]
print(resultS_ipopt)

print("\nSolution tron\n")
statsS_tron = tron(nlps_lsquare)
print(statsS_tron.solution)
print("\nPerformances tron\n")
resultS_tron = [statsS_tron.iter ; nlps_lsquare.counters.neval_obj ; nlps_lsquare.counters.neval_grad ; nlps_lsquare.counters.neval_hess ; statsS_tron.elapsed_time]
print(resultS_tron)

print("\nSolution trunk\n")
statsS_trunk = trunk(nlps_lsquare)
print(statsS_trunk.solution)
print("\nPerformances trunk\n")
resultS_trunk = [statsS_trunk.iter ; nlps_lsquare.counters.neval_obj ; nlps_lsquare.counters.neval_grad ; nlps_lsquare.counters.neval_hess ; statsS_trunk.elapsed_time]
print(resultS_trunk)
print(statsP_R2.status)



#**********************************************************************************************************************
# MODÈLES AVEC DONNÉES "RÉDUITES"

# Résolution du modèle en utilisant seulement les 10 000 premières mesures de température
m = 10000
A = ones(m,r)
for i in 1:m
    A[i,2] = i
    A[i,3] = cos(2*pi*i/(365.25))
    A[i,4] = sin(2*pi*i/(365.25)) 
    A[i,5] = cos(2*pi*i/(10.7*365.25))
    A[i,6] = sin(2*pi*i/(10.7*365.25))
end

# Déclaration des fonctions des problèmes ADNLP et ADNLS
F(x) = A*x-b[1:m]        # residual function
f(x) = (1/2)*norm(A*x-b[1:m])^2          # Fonction
x0 = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1] # first guess

# Modèle ADNLS et résolution ipopt
nlps_red = ADNLSModel(F, x0, m)
stats_sred = ipopt(nlps_red)
print(stats_sred.solution)

# Modèle ADNLP et résolution ipopt
nlpp_red = ADNLPModel(f, x0)
stats_pred = ipopt(nlpp_red)
print(stats_pred.solution)

# Représentation graphique des données de température vs les courbes obtenues
x = ones(n,1)
for i in 1:n
    x[i,1] = i
end
lpr_mod = solution(stats_pred.solution)
y_lpr = ones(n,1)
for i in 1:n
    y_lpr[i,1] = lpr_mod(x[i])
end
lsr_mod = solution(stats_sred.solution)
y_lsr = ones(n,1)
for i in 1:n
    y_lsr[i,1] = lsr_mod(x[i])
end
plot(x, b, label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x, y_lpr, label="Modèle ADNLP", linecolor="red")
savefig("graph_donnees-vs-lpmodel_reduced_all_days_MTL.svg")
savefig("graph_donnees-vs-lpmodel_reduced_all_days_MTL.png")

plot(x, b, label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x, y_lsr, label="Modèle ADNLS", linecolor="blue")
savefig("graph_donnees-vs-lsmodel_reduced_all_days_MTL.svg")
savefig("graph_donnees-vs-lsmodel_reduced_all_days_MTL.png")




# Estimation a partir des données reduites
plot(x[19000:20000], b[19000:20000], label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x[19000:20000], y_lpr[19000:20000], label="Modèle ADNLP", linecolor="red")
plot!(x[19000:20000], y_lsr[19000:20000], label="Modèle ADNLS", linecolor="blue")
savefig("from_$(m)_MTL.svg")
savefig("from_$(m)_MTL.png")


