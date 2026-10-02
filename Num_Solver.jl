####################################
# Initialise solver parameters     #
####################################

# Import libraries
using DifferentialEquations, NonlinearSolve, LinearSolve, LinearAlgebra, Symbolics, StaticArrays, SparseArrays, ArnoldiMethod, Random, Statistics, LsqFit, JLD2, DelimitedFiles

# Check whether input arguments have been passed to code
inputs = ( if isempty(ARGS) ; ["0.2", # beta
							   "1.0", # delM
							   "1.0", # delE
							   "0.5", # thresh
							   "0",   # saveAllDists
							   "1"] ; # ID
							else ARGS ; end )

# Read in parameters from input arguments
const beta = parse(Float64, inputs[1])
const delM = parse(Float64, inputs[2])
const delE = parse(Float64, inputs[3])
const thresh = parse(Float64, inputs[4])
const saveAllDists = parse(Int, inputs[5])
const ID = parse(UInt, inputs[6])

# Define how many system solves (independent sets of input parameters) will be run
nSims::Int = 2

# Check lengths to save all distributions at
if(saveAllDists == 1)
	global LSave::Vector{Float64} = [5., 20., 35., 48.7]
end

# Initialise grid size and morphogen source region width
const N::Int = 201
const N1::Int = N - 1
const N2::Int = 2 * N
const sIndex::Int = Int(round(beta * N1))
const sIndex1::Int = sIndex + 1

# Define lengthscales to simulate
const lInit::Float64 = 5.
const dL::Float64 = 0.1
const nNESS::Int = 25 # Number of simulations (inclusive) following detection of NESS

# Initialise constants for expander production
const hillPow::Int = 4
const hillPow1::Int = hillPow - 1
const threshPow::Float64 = thresh ^ hillPow

# Inititalise tolerances for steady-state detection
const absTol::Float64 = 1e-8
const relTol::Float64 = 1e-6

# Initialise the folder name for saving and loading data
folder = @__DIR__
inputFolder = folder * "/Data_Sim/"
outputFolder = folder * "/Data_Num/"

# Import functions defined in other files
include("Num_Solver_Functions.jl")



####################################
# Initialise solver                #
####################################

# Define initial conditions and parameter set
u0 = ones(N, 2) * 1e-10
p = ones(Float64, 7)

# Initialise non-linear problem for full system of equations
probNL = NonlinearProblem(fNL!, u0, p; abstol = absTol, reltol = relTol)



####################################
# Read in NESS seed data           #
####################################

# Check which systems have been simulated successfully
dataFiles = filter(x -> endswith(x, "_AllSimData.jld2"), readdir(inputFolder));
# println(dataFiles)
nFiles::Int = length(dataFiles);
dataInds = sort([parse(Int, dataFiles[k][1:end - 16]) for k in 1:nFiles]);
# println(dataInds)
dataFiles = ["$(dataInds[k])_AllSimData.jld2" for k in 1:nFiles];
# println(dataFiles)

# Check which systems have been simulated successfully
#= dataFiles = filter(x -> endswith(x, "_ForwardSimData.jld2"), readdir(inputFolder));
# println(dataFiles)
nFiles::Int = length(dataFiles);
dataInds = sort([parse(Int, dataFiles[k][1:end - 20]) for k in 1:nFiles]);
# println(dataInds)
dataFiles = ["$(dataInds[k])_ForwardSimData.jld2" for k in 1:nFiles];
println(dataFiles) =#

# Check whether number of simulations needs to be modified and assign files to simulation
maxFileID::Int = maximum(dataInds)
maxSimID::Int = Int(ceil(maximum(dataInds) / nSims))
# println("$(maxFileID), $(maxSimID)")
if(ID == maxSimID)
	global currSimIDs = [i for i in ((ID - 1) * nSims) + 1:maxFileID]
	global nSims = length(currSimIDs)
elseif(ID > maxFileID)
	global currSimIDs = []
	global nSims = length(currSimIDs)
else
	global currSimIDs = [i for i in ((ID - 1) * nSims) + 1:ID * nSims]
	# Do not need to change number of simulations
end

# Remove seeds that were not outputted from simulations
# println(currSimIDs)
currSimInds = [findall(x -> x == currSimIDs[i], dataInds)[1] for i in eachindex(currSimIDs)]
println(currSimInds)
if(length(currSimInds) > 0)
	println("Seeds $(currSimInds[1])-$(currSimInds[end]) ($(length(currSimInds)) / $(nFiles))")
else
	println("No seeds detected for this ID")
end

# Store data from simulations
outputInds = Vector{Int}(undef, 0)
allSeeds = Vector{UInt}(undef, 0)
allParas = Vector{Vector{Float64}}(undef, 0)
allInitDists = Vector{Vector{Float64}}(undef, 0)
allTSS = Vector{Vector{Float64}}(undef, 0)
allOscAmp = Vector{Vector{Float64}}(undef, 0)
allOscFreq = Vector{Vector{Float64}}(undef, 0)
allOscPhaseDiff = Vector{Vector{Float64}}(undef, 0)
allLCrit = Vector{Vector{Float64}}(undef, 0)

# Load dataset from previous simulation (read in simulation data in a new environment so that input data is deleted in the clean-up step when exiting)
let simData::Vector{CombSimData} = [load_object(inputFolder * dataFiles[i]) for i in currSimInds]
	println("Read in $(length(simData)) data sets")
	# println([simData[i].seed for i in 1:length(simData)])

    # Remove data where the simulation failed or the initial system size caused NESS
    @inbounds for i in length(simData):-1:1

        if(simData[i].LCrit == -1.)
            deleteat!(simData, i)
            deleteat!(currSimInds, i)
        end

    end

    # Extract data from (remaining) simulations
	global nSims = length(currSimInds)
	println("$(nSims) solves to complete after reading in simulation data")
	global outputInds = dataInds[currSimInds]
	println([simData[i].seed for i in eachindex(simData)])
    global allSeeds = [simData[i].seed for i in eachindex(simData)]
    global allParas = [simData[i].paras for i in eachindex(simData)]
    global allInitDists = [simData[i].initDists for i in eachindex(simData)]
    global allTSS = [[simData[i].path[j].tSS for j in eachindex(simData[i].path)] for i in eachindex(simData)]
    global allOscAmp = [[simData[i].path[j].oscTimeAmpParas[1] for j in eachindex(simData[i].path)] for i in eachindex(simData)]
    global allOscFreq = [1. ./ [simData[i].path[j].oscTimeAmpParas[3] for j in eachindex(simData[i].path)] for i in eachindex(simData)]
    global allOscPhaseDiff = [[simData[i].path[j].oscTimeAmpParas[9] for j in eachindex(simData[i].path)] for i in eachindex(simData)]
    global allLCrit = [simData[i].LCrit for i in eachindex(simData)]

end



####################################
# Loop over solves                 #
####################################

# Loop over simulations/sets of independent input parameters
@inbounds for i in 1:nSims
	println("\nParameter set $(i) out of $(length(currSimInds)) (file $(outputInds[i]))")

	# Store set of simulation input parameters
	pNew = deepcopy(allParas[i])
	# println(pNew)

	# (Re-)Initialise initial distribution for solver
	global uInit = deepcopy(allInitDists[i])

    # Set systems sizes to solve for
    LVecCurr::Vector{Float64} = [lInit + ((i - 1) * dL) for i in 1:((allLCrit[i] - lInit) / dL) + nNESS] # i.e.: (((allLCrit[i] - lInit) / dL) + 1) + (nNESS - 1)
	nLVecCurr::Int = length(LVecCurr)

    # Store output data of simulations at each system size
    data = CombNumData(outputInds[i],                              # index
					   allSeeds[i],                                # seed
					   pNew,                                       # paras
					   Vector{Vector{Float64}}(undef, nLVecCurr),  # diffTimes
					   Vector{Float64}(undef, nLVecCurr),          # eRanges
					   # Vector{Matrix{Float64}}(undef, nLVecCurr),  # distsNum
					   allTSS[i],                                  # tSSSim
					   Vector{Float64}(undef, nLVecCurr),          # maxReEigValsNum
					   Vector{Float64}(undef, nLVecCurr),          # mAmpNum
					   Vector{Float64}(undef, nLVecCurr),          # mRatioNum
					   allOscAmp[i],                               # oscAmpSim
					   Vector{Float64}(undef, nLVecCurr),          # oscFreqNum
					   allOscFreq[i],                              # oscFreqSim
					   allOscPhaseDiff[i],                         # oscPhaseDiffSim
					   -1.,                                        # LCritNum
					   allLCrit[i])                                # LCritSim
	println("Seed $(allSeeds[i])")

	# Create an initial save for current system so that seed and random numbers can be extracted if simulation crashes
    save_object(outputFolder * "$(outputInds[i])_AllNumData.jld2", data)

    # Loop over system sizes
	@inbounds for j in eachindex(LVecCurr)
        println("\n   Tissue length $(LVecCurr[j]) / $(LVecCurr[end])")
		
		# Initialise grid spacing for current tissue length
		dx::Float64 = LVecCurr[j] / N1
		dm_dx2::Float64 = pNew[1] / (dx ^ 2)
		de_dx2::Float64 = pNew[2] / (dx ^ 2)
		pSol::Vector{Float64} = [dm_dx2, de_dx2, pNew[3:5]..., pNew[6] * delM, pNew[7] * delE]
		# println(pSol)

        # Solve system numerically
		sol = solve(remake(probNL, u0 = uInit, p = (pSol)),
                    NewtonRaphson(),
					abstol = absTol, 
					reltol = relTol)

		# DEBUG Print output to test solver is working
		#= if(j == 1)
			println("Max. before: du = $(maximum(testSolve(uInit, pSol)))")
			println("Max. after:  du = $(maximum(testSolve(sol.u, pSol)))")
		end =#

		# Output steady-state profile to file
		if((saveAllDists == 1) && (LVecCurr[j] in LSave))
        	println("   Saving distributions ($(LVecCurr[j]) in $(LSave))...")
			save_object(outputFolder * "$(allSeeds[i])_L$(Int(round((LVecCurr[j] - lInit) / dL)))_Dists.jld2", sol.u)
		end

		# Store solution data
        # push!(data.distsNum, sol.u)

		# Store diffusion timescales
		data.diffTimes[j] = [((LVecCurr[j] ^ 2) ./ (2. .* pNew[1:2]))...,
							 (((LVecCurr[j] * (1. - beta)) ^ 2) ./ (2. .* pNew[1:2]))...,
							 ((diffDist(sol.u[:, 1], LVecCurr[j], dx) .^ 2) ./ (2. .* pNew[1:2]))...]

		# Store expander range
		data.eRanges[j] = (sol.u[end, 2] - sol.u[sIndex1, 2]) / sol.u[end, 2]

		# Store total morphogen concentration and ratio between maximal concentration and amplitude
		data.mAmpNum[j] = (dx / 2.) * trapezium(sol.u[:, 1])
		data.mRatioNum[j] = maximum(sol.u[:, 1]) / sol.u[1, 1]

		# Calculate and store eigen-values for corresponding interaction matrix
		eigVals::Vector{Float64} = calcEigVals(sol.u, pSol)
		data.maxReEigValsNum[j] = eigVals[1]
		data.oscFreqNum[j] = -eigVals[2] / (2. * pi)

		# Store solution as initial distribution for next length
		global uInit = sol.u

	end

	# Identify critical length according to eigen-values
	posInds::Vector{Int} = findall(x -> x >= 0., data.maxReEigValsNum)
	if(length(posInds) > 0)
		data.LCritNum = copy(LVecCurr[posInds[1]])
	end
	println("Simulation LCrit = $(data.LCritSim)")
	println(" Numerical LCrit = $(data.LCritNum)")

	# DEBUG: Compare oscillation frequencies
	# println("$(allOscFreq[i][end - 24]), $(data.oscFreqNum[end - 24])")
	#= @inbounds for j in eachindex(LVecCurr)
		println("$(Int(round(LVecCurr[j] / dL)) * dL): $(data.oscFreqSim[j]) -> $(data.oscFreqNum[j])")
	end =#

	# Save data for current system after looping over all system sizes (until critical length was reached, or after first loop if steady-state was not reached)
    save_object(outputFolder * "$(outputInds[i])_AllNumData.jld2", data)

end
