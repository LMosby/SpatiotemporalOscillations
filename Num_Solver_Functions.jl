#########################################
# List of functions for Num_Solver.jl   #
#########################################

# Define struct for storing simulation data for individual system size (from L_Vary_Functions.jl)
struct SimData

	ssReached::Int                     # Store whether system reached steady-state for each system size and production rate
	NESS::Int                          # Store whether the system reached an NESS at each system size
    oscTimeAmpParas::Vector{Float64}   # Store the oscillation timescales and amplitudes with their associated errors for both the morphogen and expander
    tSS::Float64                       # Store the time taken for each system to reach steady-state
	ranges::Vector{Float64}            # Store the ranges of the morphogen and expander gradients
	# Sm::Vector{Float64}                # Store the scaling profile for the current and previous system lengths
	# ssDists::Matrix{Float64}           # Store the final steady-state morphogen and expander distributions for each system size and production rate

end

# Define struct for storing combined simulation data for all system sizes (from L_Vary_Functions.jl)
mutable struct CombSimData

	seed::UInt                         # Store the seeds for each set of random number generation
	paras::Vector{Float64}             # Store input parameters (p), including Lcrit as the final element
	initDists::Matrix{Float64}         # Store the steady-state morphogen and expander distributions at the initial system size
    path::Vector{SimData}              # Store all data for each individual simulation at each system size
    LCrit::Float64                     # Store the critical length of the system

end

# Define struct for storing combined numerical data for all system sizes
mutable struct CombNumData

	index::UInt                        # Store the index corresponding to the system position in the output arrays of simulations
	seed::UInt                         # Store the seeds for each set of random number generation
	paras::Vector{Float64}             # Store input parameters (p), including Lcrit as the final element
    diffTimes::Vector{Vector{Float64}} # Store the diffusive timescales at each system size
    eRanges::Vector{Float64}           # Store the expander ranges at each system size
	# distsNum::Vector{Matrix{Float64}}  # Store the steady-state morphogen and expander distributions at the initial system size
	tSSSim::Vector{Float64}            # Store the time taken for each system to reach steady-state in simulations
	maxReEigValsNum::Vector{Float64}   # Store the eigen-value with the maximum real-part at each tissue length
	mAmpNum::Vector{Float64}           # Store the total morphogen concentration obtained using numerics
	mRatioNum::Vector{Float64}         # Store the ratio between the maximum morphogen concentration and its amplitude
	oscAmpSim::Vector{Float64}         # Store the oscillation amplitude obtained using simulations
	oscFreqNum::Vector{Float64}        # Store the oscillation frequency calculated using the corresponding imaginary part of the eigen-value
	oscFreqSim::Vector{Float64}        # Store the oscillation frequency obtained using simulations
	oscPhaseDiffSim::Vector{Float64}   # Store the phase difference between morphogen and expander peaks obtained using simulations
	LCritNum::Float64                  # Store the critical length of the system obtained using numerics
    LCritSim::Float64                  # Store the critical length of the system obtained using simulations

end

# Define the discretized NL PDE (same as in L_Vary.jl but without the t argument)
function fNL!(du, u, p)
	dm, de, num, nue, nueThreshPow, k1, mu = p

	du[1, 1] = (2 * dm * u[2, 1]) - (((2 * dm) + (k1 / (u[1, 2] + delM))) * u[1, 1]) + num
	du[1, 2] = (2 * de * u[2, 2]) - (((2 * de) + (mu / (u[1, 2] + delE))) * u[1, 2]) + (nueThreshPow / (threshPow + (u[1, 1] ^ hillPow)))
	@inbounds for i in 2:sIndex
		du[i, 1] = (dm * (u[i - 1, 1] + u[i + 1, 1])) - (((2 * dm) + (k1 / (u[i, 2] + delM))) * u[i, 1]) + num
		du[i, 2] = (de * (u[i - 1, 2] + u[i + 1, 2])) - (((2 * de) + (mu / (u[i, 2] + delE))) * u[i, 2]) + (nueThreshPow / (threshPow + (u[i, 1] ^ hillPow)))
	end
	@inbounds for i in sIndex1:N1
		du[i, 1] = (dm * (u[i - 1, 1] + u[i + 1, 1])) - (((2 * dm) + (k1 / (u[i, 2] + delM))) * u[i, 1])
		du[i, 2] = (de * (u[i - 1, 2] + u[i + 1, 2])) - (((2 * de) + (mu / (u[i, 2] + delE))) * u[i, 2]) + (nueThreshPow / (threshPow + (u[i, 1] ^ hillPow)))
	end
	du[N, 1] = (2 * dm * u[N1, 1]) - (((2 * dm) + (k1 / (u[N, 2] + delM))) * u[N, 1])
	du[N, 2] = (2 * de * u[N1, 2]) - (((2 * de) + (mu / (u[N, 2] + delE))) * u[N, 2]) + (nueThreshPow / (threshPow + (u[N, 1] ^ hillPow)))

end

# DEBUG: Define function to calculate the du for any system state
function testSolve(u, p)
	dm, de, num, nue, nueThreshPow, k1, mu = p

    test = zeros(N, 2)
	test[1, 1] = (2 * dm * u[2, 1]) - (((2 * dm) + (k1 / (u[1, 2] + delM))) * u[1, 1]) + num
	test[1, 2] = (2 * de * u[2, 2]) - (((2 * de) + (mu / (u[1, 2] + delE))) * u[1, 2]) + (nueThreshPow / (threshPow + (u[1, 1] ^ hillPow)))
	@inbounds for i in 2:sIndex
		test[i, 1] = (dm * (u[i - 1, 1] + u[i + 1, 1])) - (((2 * dm) + (k1 / (u[i, 2] + delM))) * u[i, 1]) + num
		test[i, 2] = (de * (u[i - 1, 2] + u[i + 1, 2])) - (((2 * de) + (mu / (u[i, 2] + delE))) * u[i, 2]) + (nueThreshPow / (threshPow + (u[i, 1] ^ hillPow)))
	end
	@inbounds for i in sIndex1:N1
		test[i, 1] = (dm * (u[i - 1, 1] + u[i + 1, 1])) - (((2 * dm) + (k1 / (u[i, 2] + delM))) * u[i, 1])
		test[i, 2] = (de * (u[i - 1, 2] + u[i + 1, 2])) - (((2 * de) + (mu / (u[i, 2] + delE))) * u[i, 2]) + (nueThreshPow / (threshPow + (u[i, 1] ^ hillPow)))
	end
	test[N, 1] = (2 * dm * u[N1, 1]) - (((2 * dm) + (k1 / (u[N, 2] + delM))) * u[N, 1])
	test[N, 2] = (2 * de * u[N1, 2]) - (((2 * de) + (mu / (u[N, 2] + delE))) * u[N, 2]) + (nueThreshPow / (threshPow + (u[N, 1] ^ hillPow)))

    return test

end

# Define function to calculate an integral using the trapezium rule
@views trapezium(u) = u[1] + u[end] + (2 * sum(u[2:end - 1])) # Only used for in-code, dimensionless functions, to calculate actual integrals need factor of dx/2

# Define function to calculate the effective diffusive distance of the morphogen-expander interactions
function diffDist(u::Vector{Float64}, LCurr::Float64, dxCurr::Float64)

	# Calculate the width of the morphogen source region
	mWidth::Float64 = beta * LCurr

	# Calculate the width of the expander source region
	eWidth::Float64 = (dxCurr / 2.) * trapezium(threshPow ./ (threshPow .+ (u .^ hillPow)))

	# Calculate total effective diffusive distance
	dist::Float64 = LCurr - mWidth - eWidth
	if(dist < 0)
		return -1.
	else
		return dist
	end

end

# Define function to calculate eigen-values for an interaction matrix
function calcEigVals(u::Matrix{Float64}, p::Vector{Float64})
	dm_dx2, de_dx2, num, nue, nueThreshPow, k1, mu = p

	# Create interaction matrix to calculate eigen-values
	# mat = zeros(N2, N2)
	mat = spzeros(N2, N2)
	@inbounds for i in 1:N
		ni::Int = N + i
		
		mat[i, i] = -(2 * dm_dx2) - (k1 / (u[i, 2] + delM))
		mat[i, ni] = ((k1 * u[i, 1]) / ((u[i, 2] + delM) ^ 2))

		if(i == 1)
			mat[i, i + 1] = 2 * dm_dx2
		elseif(i == N)
			mat[i, i - 1] = 2 * dm_dx2
		else
			mat[i, i - 1] = dm_dx2
			mat[i, i + 1] = dm_dx2
		end

		mat[ni, ni] = -(2 * de_dx2) - ((mu * delE) / ((u[i, 2] + delE) ^ 2))
		mat[ni, i] = -((hillPow * nueThreshPow * (u[i, 1] ^ hillPow1)) / ((threshPow + (u[i, 1] ^ hillPow)) ^ 2))

		if(i == 1)
			mat[ni, ni + 1] = 2 * de_dx2
		elseif(i == N)
			mat[ni, ni - 1] = 2 * de_dx2
		else
			mat[ni, ni - 1] = de_dx2
			mat[ni, ni + 1] = de_dx2
		end

	end

	# Calculate eigen-values of matrix
	# Traditional functions only work for regular matrix, not sparse arrays
	#= eigVals = eigvals(mat)
	maxInd = findall(x -> x == maximum(real.(eigVals)), real.(eigVals))[1]
	# println([real(eigVals[maxInd]), imag(eigVals[maxInd])])
	return [real(eigVals[maxInd]), imag(eigVals[maxInd])] =#

	# Use new method involving partial Schur decomposition when using sparse arrays
	schurDecomp = partialschur(mat, nev = 1, tol = 1e-6, which = :LR, restarts = Int(1e8))
	maxEigVal = partialeigen(schurDecomp[1])[1][1]
	println([real(maxEigVal), imag(maxEigVal)])
	return [real(maxEigVal), imag(maxEigVal)]

end
