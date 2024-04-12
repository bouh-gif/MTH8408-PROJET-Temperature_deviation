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
# Pkg.add("Percival")

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
using Percival


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
f(x) = (1/2)*norm(A*x-b)^2          # Fonction
x0 = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1] # first guess

# Modèle ADNLS et résolution ipopt
nlps_lsquare = ADNLSModel(F, x0, n)
stats_s = ipopt(nlps_lsquare)
print(stats_s.solution)

# Modèle ADNLP et résolution ipopt
# nlpp_lsquare = ADNLPModel(f, x0)
# stats_p = ipopt(nlpp_lsquare)
# print(stats_p.solution)

#**********************************************************************************************************************
# Représentation graphique des données de température vs les courbes obtenues
x = ones(n,1)
for i in 1:n
    x[i,1] = i
end

# Données de l'article
article_mod = solution([52.6 1.2E-4 -20.3 -7.97 0.275 0.454])
y_article = ones(n,1)
for i in 1:n
    y_article[i,1] = article_mod(x[i])
end

# Modèle NLP
lp_mod = solution(stats_p.solution)
y_lp = ones(n,1)
for i in 1:n
    y_lp[i,1] = lp_mod(x[i])
end

# Modèle NLS
ls_mod = solution(stats_s.solution)
y_ls = ones(n,1)
for i in 1:n
    y_ls[i,1] = ls_mod(x[i])
end

start_plot = 1
end_plot = n
plot(x[start_plot:end_plot], b[start_plot:end_plot], label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x[start_plot:end_plot], y_lp[start_plot:end_plot], label="Modèle ADNLP", linecolor="red")
plot!(x[start_plot:end_plot], y_ls[start_plot:end_plot], label="Modèle ADNLS", linecolor="blue")
plot!(x[start_plot:end_plot], y_article[start_plot:end_plot], label="Modèle article", linecolor="green")
# savefig("two_models_day$(start_plot)_to_day$(end_plot).svg")
savefig("all_data_$(start_plot)_to_day$(end_plot).png")



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

lp_lbfgs = solution(statsP_lbfgs.solution)
y_lp_lbfgs = ones(n,1)
for i in 1:n
    y_lp_lbfgs[i,1] = lp_lbfgs(x[i])
end

print("\nSolution tron\n")
statsP_tron = tron(nlpp_lsquare)
print(statsP_tron.solution)
print("\nPeformances tron\n")
resultP_tron = [statsP_tron.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; statsP_tron.elapsed_time]
print(resultP_tron)

lp_tron = solution(statsP_tron.solution)
y_lp_tron = ones(n,1)
for i in 1:n
    y_lp_tron[i,1] = lp_tron(x[i])
end

print("\nSolution trunk\n")
statsP_trunk = trunk(nlpp_lsquare)
print(statsP_trunk.solution)
print("\nPeformances trunk\n")
resultP_trunk = [statsP_trunk.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; statsP_trunk.elapsed_time]
print(resultP_trunk)

lp_trunk = solution(statsP_trunk.solution)
y_lp_trunk = ones(n,1)
for i in 1:n
    y_lp_trunk[i,1] = lp_trunk(x[i])
end

print("\nSolution R2\n")
statsP_R2 = R2(nlpp_lsquare)
print(statsP_R2.solution)
print("\nPeformances R2\n")
resultP_R2 = [statsP_R2.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; statsP_R2.elapsed_time]
print(resultP_R2)
print(statsP_R2.status)

lp_R2 = solution(statsP_R2.solution)
y_lp_R2 = ones(n,1)
for i in 1:n
    y_lp_R2[i,1] = lp_R2(x[i])
end

print("\nSolution percival\n")
statsP_perci = percival(nlpp_lsquare)
print(statsP_perci.solution)
print("\nPeformances percival\n")
resultP_perci = [statsP_perci.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; statsP_perci.elapsed_time]
print(resultP_perci)

lp_perci = solution(statsP_perci.solution)
y_lp_perci = ones(n,1)
for i in 1:n
    y_lp_perci[i,1] = lp_perci(x[i])
end

# REPRÉSENTATION GRAPHIQUE

start_plot = 10000
end_plot = 10500
plot(x[start_plot:end_plot], b[start_plot:end_plot], label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x[start_plot:end_plot], y_lp[start_plot:end_plot], label="Ipopt", linecolor="red")
plot!(x[start_plot:end_plot], y_lp_lbfgs[start_plot:end_plot], label="Lbfgs", linecolor="blue")
plot!(x[start_plot:end_plot], y_lp_tron[start_plot:end_plot], label="Tron", linecolor="green")
plot!(x[start_plot:end_plot], y_lp_trunk[start_plot:end_plot], label="Trunk", linecolor="yellow")
plot!(x[start_plot:end_plot], y_lp_R2[start_plot:end_plot], label="R2", linecolor="fuchsia")
plot!(x[start_plot:end_plot], y_lp_perci[start_plot:end_plot], label="Percival", linecolor="orange")
# savefig("two_models_day$(start_plot)_to_day$(end_plot).svg")
 savefig("solvers_$(start_plot)_to_day$(end_plot).png")


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

ls_tron = solution(statsS_tron.solution)
y_ls_tron = ones(n,1)
for i in 1:n
    y_ls_tron[i,1] = ls_tron(x[i])
end

print("\nSolution trunk\n")
statsS_trunk = trunk(nlps_lsquare)
print(statsS_trunk.solution)
print("\nPerformances trunk\n")
resultS_trunk = [statsS_trunk.iter ; nlps_lsquare.counters.neval_obj ; nlps_lsquare.counters.neval_grad ; nlps_lsquare.counters.neval_hess ; statsS_trunk.elapsed_time]
print(resultS_trunk)
print(statsP_R2.status)

ls_trunk = solution(statsS_trunk.solution)
y_ls_trunk = ones(n,1)
for i in 1:n
    y_ls_trunk[i,1] = ls_trunk(x[i])
end

print("\nSolution percival\n")
statsS_perci = percival(nlps_lsquare)
print(statsS_perci.solution)
print("\nPeformances percival\n")
resultS_perci = [statsS_perci.iter ; nlps_lsquare.counters.neval_obj ; nlps_lsquare.counters.neval_grad ; nlps_lsquare.counters.neval_hess ; statsS_perci.elapsed_time]
print(resultS_perci)

ls_perci = solution(statsS_perci.solution)
y_ls_perci = ones(n,1)
for i in 1:n
    y_ls_perci[i,1] = ls_perci(x[i])
end

start_plot = 10000
end_plot = 10500
plot(x[start_plot:end_plot], b[start_plot:end_plot], label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x[start_plot:end_plot], y_ls[start_plot:end_plot], label="Ipopt", linecolor="red")
plot!(x[start_plot:end_plot], y_ls_tron[start_plot:end_plot], label="Tron", linecolor="green")
plot!(x[start_plot:end_plot], y_ls_trunk[start_plot:end_plot], label="Trunk", linecolor="yellow")
plot!(x[start_plot:end_plot], y_ls_perci[start_plot:end_plot], label="Percival", linecolor="orange")
# savefig("two_models_day$(start_plot)_to_day$(end_plot).svg")
 savefig("solvers_ls_$(start_plot)_to_day$(end_plot).png")


#**********************************************************************************************************************
# MODÈLES AVEC DONNÉES "RÉDUITES"

# Déclaration des vecteurs initiaux

sol1 = []
sol2 = []
val = 2000
m = [val]
#81
for i in 1:33
    val = val+250
    push!(m, val)
end

for k in m
    A = ones(k,r)
    for i in 1:k
        A[i,2] = i
        A[i,3] = cos(2*pi*i/(365.25))
        A[i,4] = sin(2*pi*i/(365.25))
        A[i,5] = cos(2*pi*i/(10.7*365.25))
        A[i,6] = sin(2*pi*i/(10.7*365.25))
    end

    # Déclaration des fonctions des problèmes ADNLP et ADNLS
    F(x) = A*x-b[1:k]        # residual function
    x0 = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1] # first guess

    # Modèle ADNLS et résolution ipopt
    nlps_red = ADNLSModel(F, x0, k)
    stats_sred = ipopt(nlps_red, print_level = 0)

    push!(sol1, stats_sred.solution[1])
    push!(sol2, stats_sred.solution[2])

end

# Représentation graphique des données de température vs les courbes obtenues
plot(m, sol1, ylabel="Valeur de x0", xlabel="Nombre de données utilisé", linecolor="red", ylims=(minimum(sol1),maximum(sol1)))
savefig("donnees-red_x0.png")
plot(m, sol2, ylabel="Valeur de x1", xlabel="Nombre de données utilisé", linecolor="blue", ylims=(minimum(sol2),maximum(sol2)))
savefig("donnees-red_x1.png")


plot(x, b, label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x, y_lsr, label="Modèle ADNLS", linecolor="blue")
savefig("graph_donnees-vs-lsmodel.svg")
savefig("graph_donnees-vs-lsmodel.png")



# Estimation a partir des données reduites
plot(x[19000:20000], b[19000:20000], label="Données",ylabel="Température", xlabel="Jour", linecolor="lightgrey")
plot!(x[19000:20000], y_lpr[19000:20000], label="Modèle ADNLP", linecolor="red")
plot!(x[19000:20000], y_lsr[19000:20000], label="Modèle ADNLS", linecolor="blue")
savefig("from_$m.svg")
savefig("from_$m.png")

