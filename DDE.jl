####################################
# Initialise simulation parameters #
####################################

# Import libraries
using DifferentialEquations, LinearAlgebra, Symbolics, NonlinearSolve, Random, Statistics, LsqFit, JLD2

# Check whether input arguments have been passed to code
inputs = ( if isempty(ARGS) ; ["0.2",   # beta
							   "1.0",   # delM
							   "1.0",   # delE
							   "0.5",   # thresh
							   "1"] ;   # ID
							else ARGS ; end )

# Read in parameters from input arguments
const beta = parse(Float64, inputs[1])
const delM = parse(Float64, inputs[2])
const delE = parse(Float64, inputs[3])
const thresh = parse(Float64, inputs[4])
const ID = parse(UInt, inputs[5])

# Define how many simulations (independent sets of input parameters) will be run
const nSims::Int = 11

# Define how many maxima and minima must be within the small margin to define as an NESS
const nPeakRepeats::Int = 10

# Define how many maxima and minima to use to calculate oscillation amplitudes and frequencies (if available)
const nOscs::Int = 10

# Define number of simulations (inclusive) following detection of NESS before termination of simulation
const nNESS::Int = 25

# Define delay timescales to simulate
const tauMin::Float64 = 0.1
const tauMax::Float64 = 1000.
const dTau::Float64 = 0.1
const nTau::Int = Int(ceil((tauMax - tauMin + dTau) / dTau))
taus::Vector{Float64} = [tauMin + ((i - 1.) * dTau) for i in 1:nTau]
#= const logTauMin::Float64 = -2.
const logTauMax::Float64 = 3.
const dLogTau::Float64 = 0.1
const nTau::Int = Int(ceil((logTauMax - logTauMin + dLogTau) / dLogTau))
taus::Vector{Float64} = [10. ^ (logTauMin + ((i - 1.) * dLogTau)) for i in 1:nTau] =#
# println("taus = $(taus)")

# Initialise Hill constants
const hillPowH::Int = 4
const hillPowM::Int = 1
const hillPowN::Int = 1

# Inititalise tolerances for steady-state detection
const absTol::Float64 = 1e-10
const relTol::Float64 = 1e-8

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
tPeakIPrev::Float64 = 0.
uPeakIPrev::Float64 = 0.
typePeakIPrev::Int = -1
uPeakI1 = Vector{Float64}()
uPeakT1 = Vector{Float64}()
uPeakI2 = Vector{Float64}()
uPeakT2 = Vector{Float64}()

# Initialise the folder name for saving and loading data
folder = @__DIR__

# Import functions defined in other files
include("DDE_Functions.jl")



####################################
# Initialise solvers               #
####################################

# Define initial conditions and time-scales for solver
u0::Vector{Float64} = ones(Float64, 2)
tSpan = (0., Inf)

# Initialise parameters for calculating the Jacobian
du0::Vector{Float64} = deepcopy(u0)
p::Vector{Float64} = ones(Float64, 4)

# Initialise Jacobian and problem for system with constant delay (using delay-differential equation and history function)
h(p, t) = zeros(Float64, 2)
currTau::Float64 = taus[1]
lags::Vector{Float64} = [currTau]
prob = ODEProblem(f!, u0, tSpan, p)
probDDE = DDEProblem(fDDE!, u0, h, tSpan, p; constant_lags = lags)

# Initialise combined callback functions for detecting steady state, maintaining positive concentrations, oscillations and linear expander growth
cbTerm = DiscreteCallback(newTerminateSteadyStateCondition, newTerminateSteadyStateAffect!)
cbOscCheckM = ContinuousCallback(oscCheckMCondition, oscCheckMStringentAffect!)
cbPosDomain = PositiveDomain()
cbSet = CallbackSet(cbTerm, cbOscCheckM, cbPosDomain)



####################################
# Initialise simulation parameters #
####################################

# Initialise values for A, B, G and sigma
fullSimVars::Vector{Float64} = [0.022667631902975684, 4.192373918360967, 1.9489487858961354, 0.34071631568120886] # In order: nu_I, nu_A, k_I, k_A
A0::Float64 = 1. / (fullSimVars[2] / fullSimVars[4])
println("A0 = $(A0)")
AVec::Vector{Float64} = [A0 * (10. ^ i) for i in -0.5:(1. / (nSims - 1.)):0.5]
println("Values of A to be simulated: $(AVec)")
A::Float64 = AVec[1]
B::Float64 = 0.5 / (fullSimVars[1] / fullSimVars[3])
G::Float64 = 1. / (fullSimVars[2] / fullSimVars[4])
sigma::Float64 = fullSimVars[4] / fullSimVars[3]
println("Parameters = $([A, B, G, sigma])")



####################################
# Loop over simulations            #
####################################

# Set indexes to be simulated in current simulation
currSimInds::Vector{Int} = ((ID - 1) * nSims) + 1:ID * nSims
# seeds::Vector{UInt} = [UInt(round(time())) + ID + (1e6i) for i in 1:nSims]

# Loop over simulations/sets of independent input parameters
@inbounds for i in currSimInds
	println("\nParameter set $(i) out of $(currSimInds[1])-$(currSimInds[end])")

	# Re-initialise break counter (number of profiles that reach NESS before simulation ends)
	breakCounter::Int = 0;

	# Re-Initialise initial distribution for solver
	global uInit = deepcopy(u0)

	# Store new system parameters
	global A = AVec[i - currSimInds[1] + 1]
	global pNew = [A ^ hillPowM, B ^ hillPowH, G ^ hillPowN, sigma]

	# Solve for steady-state activator and inhibitor levels
	probNL = NonlinearSolve.NonlinearProblem(fNL!, [1., 1.], pNew; abstol = absTol, reltol = relTol, lb = [0.0, 0.0])
	solNL = NonlinearSolve.solve(probNL, NonlinearSolve.NewtonRaphson())
	# println("SS sols = $(solNL[1:2])")

	# Calculate analytical estimate for point of bifurcation for current parameters
	jac::Matrix{Float64}, wCrit::Float64, tauCrit::Float64 = analyticalBifurcation(solNL[1:2], pNew)

    # Store output data of simulations at each system size
    data = AllDDEData(pNew, Vector{DDEData}(), -1., -1., solNL[1:2], jac, wCrit, tauCrit)

	# DEBUG: Create an initial save for current system so that seed and random numbers can be extracted if simulation crashes
    save_object(folder * "/Data_DDE/$(i)_AllDDEData.jld2", data)

    # Loop over delay magnitudes
	@inbounds for j in taus
        # println("\n   Delay $(j) / $(taus[end])")

		# Define new delay
		global currTau = j
        global lags = [currTau]

        # Solve system
		# push!(data.path, solveSystem(pNew, prob))
		push!(data.path, solveSystem(pNew, DDEProblem(fDDE!, u0, h, tSpan, pNew; constant_lags = lags)))

		# Skip to next system if simulation did not finish correctly
		if(data.path[end].ssReached == -1)
			# println("   Simulation not finished correctly...")
			break
		end

		# Check if simulation has been terminated by reaching an NESS or did not finish correctly
        if(data.path[end].NESS == 1)
			breakCounter += 1;
			# println("   System reached NESS")
            if(breakCounter == 1)
				println("   System reached NESS at delay = $(j)")
				println("      Compare to ana. delay = $(tauCrit)")
				println("      Osc. freq. = $(1. / data.path[end].oscTimeAmpParas[1])")
				println("      Compare to ana. freq. = $(wCrit)")

				# Store critical delay magnitude and oscillation frequency at the critical point
				data.wCrit = 1. / data.path[end].oscTimeAmpParas[1]
				data.tauCrit = j
				break

			end
			#= if(breakCounter == nNESS)

                # Simulation is completed
				# println("   System reached NESS for $(nNESS) simulations")
				break

			end
		else

			# Reset NESS counter
			breakCounter = 0; =#

		end

	end

	# Save data for current system after looping over all delayu magnitudes (until critical delay was reached, or after first loop if steady-state was not reached)
    save_object(folder * "/Data_DDE/$(i)_AllDDEData.jld2", data)

end
