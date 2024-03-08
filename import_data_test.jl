import Pkg
# Pkg.add("HTTP")
# Pkg.add("CSV")
# Pkg.add("DataFrames")

using HTTP
using CSV
using DataFrames

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

# Print the vectors
println("Dates: ", dates_vector)
println("Temperatures: ", temperatures_vector)
