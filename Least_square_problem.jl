# # Uncomment this section the first time running the program
# using Pkg
# Pkg.add("ADNLPModels")
# Pkg.add("NLPModels")
# Pkg.add("LinearAlgebra")
# Pkg.add("NLPModelsIpopt")
# Pkg.add("HTTP")
# Pkg.add("CSV")
# Pkg.add("DataFrames")


using LinearAlgebra
using ADNLPModels
using NLPModels
using NLPModelsIpopt
using HTTP
using CSV
using DataFrames


#**********************************************************************************************************************
# Function to evaluate the values in a given line in matrix A of the model
function line(d)
    l = [1 d cos(2*pi*d/(365.25)) sin(2*pi*d/(365.25)) cos(2*pi*d/(10.7*365.25)) sin(2*pi*d/(10.7*365.25))]
    return l
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
    A[i,6] = cos(2*pi*i/(10.7*365.25))
end

F(x) = A*x-b
f(x) = (1/2)*norm(A*x-b)^2
x0 = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1] # first guess

nlps_lsquare = ADNLSModel(F, x0, 2)

nlpp_lsquare = ADNLPModel(f, x0)

residual(nlps_lsquare, x0)
jac_residual(nlps_lsquare, x0)

stats_v1 = ipopt(nlps_lsquare)
stats_v2 = ipopt(nlpp_lsquare)