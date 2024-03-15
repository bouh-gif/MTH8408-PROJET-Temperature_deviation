# Uncomment this section the first time running the program
using Pkg
Pkg.add("ADNLPModels")
Pkg.add("NLPModels")
Pkg.add("LinearAlgebra")
Pkg.add("NLPModelsIpopt")
Pkg.add("HTTP")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Dates")


using LinearAlgebra
using ADNLPModels
using NLPModels
using NLPModelsIpopt
using HTTP
using CSV
using DataFrames
using CSV
using Dates


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