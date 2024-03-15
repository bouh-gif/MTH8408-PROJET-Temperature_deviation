import Pkg
Pkg.add("HTTP")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Dates")

using HTTP
using CSV
using DataFrames
using CSV
using Dates

# Chemins des fichiers CSV
file_paths = ["climate-daily1.csv", "climate-daily2.csv", "climate-daily3.csv"]

# Initialisation des vecteurs de température et de date
temperatures = Float64[]
dates = Vector{String}()

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
    append!(temperatures, temperature_column_filtered)
    append!(dates, date_column_filtered)
end

# Formater les dates au format yyyymmdd
dates = Dates.format.(Date.(dates, "yyyy-mm-dd HH:MM:SS"), "yyyymmdd")

# Afficher les 10 premières valeurs des vecteurs (pour vérification)
println("Températures :", temperatures[1:10])
println("Dates :", dates[1:10])
# Afficher les 10 dernières valeurs des vecteurs
println("Dix dernières températures :", temperatures[end-9:end])
println("Dix dernières dates :", dates[end-9:end])
# Afficher la longueur des vecteurs
println("Nombre total de températures :", length(temperatures))
println("Nombre total de dates :", length(dates))
