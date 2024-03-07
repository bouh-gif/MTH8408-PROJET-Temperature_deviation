using Pkg
# Pkg.add("ADNLPModels")
# Pkg.add("NLPModels")
# Pkg.add("LinearAlgebra")
# Pkg.add("NLPModelsIpopt")

using LinearAlgebra
using ADNLPModels #Pkg.add("ADNLPModels")
using NLPModels #Pkg.add("NLPModels")
using NLPModelsIpopt #Pkg.add("NLPModelsIpopt")

# Importation des données de températures dans un vecteur b

####################
# Déclaration de la matrice A selon le nombre de données de températures
# n = length(b)
function line(d)
    l = [1 d cos(2*pi*d/(365.25)) sin(2*pi*d/(365.25)) cos(2*pi*d/(10.7*365.25)) sin(2*pi*d/(10.7*365.25))]
    return l
end
r = 6
n = 2000

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
x0 = [0.1; 0.1]

nlps_lsquare = ADNLSModel(F, x0, 2)

nlpp_lsquare = ADNLPModel(f, x0)

residual(nlps_lsquare, x0)
jac_residual(nlps_lsquare, x0)

stats_v1 = ipopt(nlps_lsquare)
stats_v2 = ipopt(nlpp_lsquare)