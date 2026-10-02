####################################
# Initialise simulation parameters #
####################################

# Import libraries
using DifferentialEquations, LinearAlgebra, Symbolics, StaticArrays, Random, Statistics, LsqFit, JLD2, DelimitedFiles

# Check whether input arguments have been passed to code
inputs = ( if isempty(ARGS) ; ["100.0", # LBuff
							   "0.2",   # beta
							   "1.0",   # delM
							   "1.0",   # delE
							   "0.5",   # thresh
							   "0",     # saveAllDists
							   "0",     # compareSimNum
							   "0",     # shrinkingFlag
							   "1"] ;   # ID (2566 is maximum for data in Hopf_Seeds.jld2)
							else ARGS ; end )

# Read in parameters from input arguments
const LBuff = parse(Float64, inputs[1])
const beta = parse(Float64, inputs[2])
const delM = parse(Float64, inputs[3])
const delE = parse(Float64, inputs[4])
const thresh = parse(Float64, inputs[5])
const saveAllDists = parse(Int, inputs[6])
const compareSimNum = parse(Int, inputs[7])
const shrinkingFlag = parse(Int, inputs[8])
const ID = parse(UInt, inputs[9])

# Define how many simulations (independent sets of input parameters) will be run
const nSims::Int = 2

# Define how many maxima and minima must be within the small margin to define as an NESS (non-equilibrium steady-state)
const nPeakRepeats::Int = 10

# Define how many maxima and minima to use to calculate oscillation amplitudes and frequencies (if available)
const nOscs::Int = 10

# Initialise grid size and morphogen source region width
const N::Int = 201
const N1::Int = N - 1
const N2::Int = N - 2
const sIndex::Int = Int(round(beta * N1))
const sIndex1::Int = sIndex + 1

# Define lengthscales to simulate
const lInit::Float64 = 5.
const lSave::Float64 = 5.
const dL::Float64 = 0.1
const nL::Int = Int(ceil((LBuff - lInit + dL) / dL))
LVec::Vector{Float64} = [lInit + ((i - 1) * dL) for i in 1:nL]
# println(LVec)
nNESS::Int = 25 # Number of simulations (inclusive) following detection of NESS

# Reverse lengthscales to check for hysteresis
if(shrinkingFlag == 1)

	# LVec = reverse(LVec)
	LVecMax::Float64 = 56.1
	LVec = reverse(LVec[findall(x -> x <= LVecMax, LVec)])
	# println(LVec)
	global nNESS = length(LVec) # So the simulations are not cancelled prematurely
	# println(nNESS)

end

# Check lengths to save all distributions at
if(saveAllDists == 1)

	# global LSave::Vector{Float64} = deepcopy(LVec)
	global LSave::Vector{Float64} = [5., 20., 35., 48.7]

end

# Initialise indexes and positions corresponding to gene expression boundary positions
#= const nBIndexes::Int = 19
bIndexes = SVector{nBIndexes, Int}(sIndex1 .+ [i * Int(round(N - sIndex1) / (nBIndexes + 1)) for i in 1:nBIndexes])
# println(bIndexes)

# Initialise relative co-ordinates for scaling calculation
relX::Vector{Float64} = [(i - 1) / N1 for i in 1:N]
posL1 = Vector{Float64}((relX[bIndexes] .- relX[sIndex1]) ./ (relX[end] - relX[sIndex1]))
# println(posL1) =#

# Initialise constants for expander production
const hillPow::Int = 4
const threshPow::Float64 = thresh ^ hillPow

# Inititalise tolerances for steady-state detection
const absTol::Float64 = 1e-10
const relTol::Float64 = 1e-8

# Initialise tolerance for including successive concentrations in the measurement of the oscillation frequency
const eps::Float64 = 1e-6

# Initialise ranges and minimum values for diffusivity, degradation and production parameters
range::Vector{Float64} = [4., 4., 4., 4., 4., 4.] # Use same minimum and maximum values as for evolution simulations
minVal::Vector{Float64} = [-3., -3., -3., -3., -3., -3.]

# Initialise flag and time variables required to decide when system has reached steady state
flagSS::Int = 0
tAnySS::Float64 = 0. # Stores the time when derivatives fall below threshold closest to final simulation time
tPrevSS::Float64 = 0. # Stores the time when derivatives fall below threshold or system fails check which must be repeated closest to final simulation time
mTotPrevSS::Float64 = 0.
mTotDevMax::Float64 = 0.
const nSSCheck::Int = 50
const ssCheckTime = 5e2

# Initialise flag storing whether system reached a NESS and variables used to calculate when this occurs
nessFlag::Int = 0
tPeakMPrev::Float64 = 0.
uPeakMPrev::Float64 = 0.
typePeakMPrev::Int = -1
uPeakM1 = Vector{Float64}()
uPeakT1 = Vector{Float64}()
uPeakM2 = Vector{Float64}()
uPeakT2 = Vector{Float64}()

# Initialise the folder name for saving and loading data
folder = @__DIR__

# Import functions defined in other files
include("L_Vary_Functions.jl")



####################################
# Initialise solvers               #
####################################

# Define initial conditions and time-scales for solver
u0 = ones(N, 2) .* 1e-10
tSpan = (0., Inf)
const dtMin::Float64 = 1e-6

# Initialise parameters for calculating the Jacobian
du0 = copy(u0)
p = ones(Float64, 7)
dm0::Float64 = 1.
k0::Float64 = 1.

# Initialise Jacobian and problem for full system of equations
jacKFeedback = Symbolics.jacobian_sparsity((du, u) -> fKFeedback!(du, u, p, 0.), du0, u0)
fKFeedbackSparse = ODEFunction(fKFeedback! ; jac_prototype = float.(jacKFeedback))
probKFeedback = ODEProblem(fKFeedbackSparse, u0, tSpan, p)
#= jacKFeedbackAllSource = Symbolics.jacobian_sparsity((du, u) -> fKFeedbackAllSource!(du, u, p, 0.), du0, u0)
fKFeedbackAllSourceSparse = ODEFunction(fKFeedbackAllSource! ; jac_prototype = float.(jacKFeedbackAllSource))
probKFeedback = ODEProblem(fKFeedbackAllSourceSparse, u0, tSpan, p) =#

# Initialise combined callback functions for detecting steady state, maintaining positive concentrations, oscillations and linear expander growth
cbTerm = DiscreteCallback(newTerminateSteadyStateCondition, newTerminateSteadyStateAffect!)
cbOscCheckM = ContinuousCallback(oscCheckMCondition, oscCheckMStringentAffect!)
cbPosDomain = PositiveDomain()
cbSet = CallbackSet(cbTerm, cbOscCheckM, cbPosDomain)



####################################
# Loop over simulations            #
####################################

# Load random number seeds known to cause NESS
seeds = load_object(folder * "/Hopf_Seeds.jld2")
nSeeds::Int = length(seeds)
# println(nSeeds)
# 64133 seeds for L1

# Check whether number of simulations needs to be modified and assign seeds to simulation
maxSeedID::Int = Int(ceil(nSeeds / nSims))
if(ID == maxSeedID)
	global currSimInds = ((ID - 1) * nSims) + 1:nSeeds
	global nSims = length(currSimInds)
elseif(ID > maxSeedID)
	global currSimInds = []
	global nSims = length(currSimInds)
else
	global currSimInds = ((ID - 1) * nSims) + 1:ID * nSims
	# Do not need to change number of simulations
end
println("Seeds $(currSimInds[1])-$(currSimInds[end]) / $(nSeeds)")

# Loop over simulations/sets of independent input parameters
@inbounds for i in currSimInds # UInt.([1])
	println("\nParameter set $(i) out of $(currSimInds[1])-$(currSimInds[end])")

	# Re-initialise break counter (number of profiles that reach NESS before simulation ends)
	breakCounter::Int = 0;

	# (Re-)Initialise initial distribution for solver
	global uInit = copy(u0)

	# Seed random numbers for remaining input parameters
	nSeed::UInt = seeds[i]

	# Examples for plotting
	# nSeed::UInt = 1869544802 # Cleanest system
	# nSeed::UInt = 0x000000006ed66f9a # Sub-critical bifurcation
	# println(nSeed)
	
	# Use random number seed to generate random input parameters
	Random.seed!(nSeed)
	r = rand(6)
	@. r = 10 ^ ((range * r) + minVal)

	# Store current set of input parameters
	pNew = [r[1], r[2], r[3], r[4], r[4] * threshPow, r[5], r[6]]
	println(pNew)

    # Store output data of simulations at each system size
    data = CombSimData(nSeed, pNew, Matrix{Float64}(undef, N, 2), Vector{SimData}(), -1.)
	println("Seed $(data.seed)")

	# DEBUG: Create an initial save for current system so that seed and random numbers can be extracted if simulation crashes
    saveAllDists == 0 ? save_object(folder * "/Data_Sim/$(i)_AllSimData.jld2", data) : save_object(folder * "/Data_Sim/$(nSeed)_AllSimData.jld2", data)

    # Loop over system sizes
	@inbounds for j in LVec
        println("\n   Tissue length $(j) / $(LBuff)")

        # Solve system numerically
		push!(data.path, solveSystem(nSeed, pNew, probKFeedback, j))


		# Skip to next system if steady-state was not reached at initial length
		if((lSave[1] - eps < j < lSave[1] + eps) && (data.path[end].ssReached == 0))
			
            println("   Skipping to next system...")
			break

        elseif(lSave[1] - eps < j < lSave[1] + eps) # Save initial morphogen and expander distributions

            # End state of current simulation saved as initial condition for next simulation
            data.initDists = uInit

		end

		# Skip to next system if simulation did not finish correctly
		if(data.path[end].ssReached == -1)
			println("   Simulation not finished correctly...")
			break
		end

		# Check if simulation has been terminated by reaching an NESS, a linear growth state, or did not finish correctly
        if(data.path[end].NESS == 1)
			breakCounter += 1;
			println("   System reached NESS")
            if(breakCounter == 1)

				# Store critical length
				data.LCrit = j

			end
			if(breakCounter == nNESS)

                # Simulation is completed
				println("   System reached NESS for $(nNESS) simulations")
				break

			end
		else

			# Reset NESS counter, sometimes get strange observations at very small L that are rectified as the system grows
			breakCounter = 0

		end

	end

	# Save data for current system after looping over all system sizes (until critical length was reached, or after first loop if steady-state was not reached)
    saveAllDists == 0 ? save_object(folder * "/Data_Sim/$(i)_AllSimData.jld2", data) : save_object(folder * "/Data_Sim/$(nSeed)_AllSimData.jld2", data)

end
