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

# Modèle ADNLP et résolution ipopt
nlps_lsquare = ADNLSModel(F, x0, n)
stats_p = ipopt(nlps_lsquare)
print(stats_p.solution)

# Modèle ADNLS et résolution ipopt
nlpp_lsquare = ADNLPModel(f, x0)
stats_s = ipopt(nlpp_lsquare)
print(stats_s.solution)

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
plot(x, b, label="Données",ylabel="Température", xlabel="Jour")
#plot!(x[10000:12000], y_lp[10000:12000], label="Modèle ADNLP")
plot!(x, y_ls, label="Modèle ADNLS")
#**********************************************************************************************************************
print("\n")
result_ipopt = [stats_p.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; stats_p.elapsed_time]
print(result_ipopt)
print("\n")

stats_lbfgs = lbfgs(nlpp_lsquare)
print(stats_lbfgs.solution)
print("\n")
result_lbfgs = [stats_lbfgs.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; stats_lbfgs.elapsed_time]
print(result_lbfgs)
print("\n")

stats_tron = tron(nlpp_lsquare)
print(stats_tron.solution)
print("\n")
result_tron = [stats_tron.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; stats_tron.elapsed_time]
print(result_tron)
print("\n")

stats_trunk = trunk(nlpp_lsquare)
print(stats_trunk.solution)
print("\n")
result_trunk = [stats_trunk.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; stats_trunk.elapsed_time]
print(result_trunk)
print("\n")

stats_R2 = R2(nlpp_lsquare)
print(stats_R2.solution)
print("\n")
result_R2 = [stats_R2.iter ; nlpp_lsquare.counters.neval_obj ; nlpp_lsquare.counters.neval_grad ; nlpp_lsquare.counters.neval_hess ; stats_R2.elapsed_time]
print(result_R2)
print("\n")


# print(stats_lbfgs.solution)
# print(stats_tron.solution)
# print(stats_trunk.solution)
# print(stats_R2.solution)