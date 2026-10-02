#####################################
# List of functions for L_Vary.jl   #
#####################################

# Define struct for storing simulation data for individual system size
struct SimData

	ssReached::Int                   # Store whether system reached steady-state for each system size and production rate
	NESS::Int                        # Store whether the system reached an NESS at each system size
    oscTimeAmpParas::Vector{Float64} # Store the oscillation timescales and amplitudes with their associated errors for both the morphogen and expander
    tSS::Float64                     # Store the time taken for each system to reach steady-state
	ranges::Vector{Float64}          # Store the ranges of the morphogen and expander gradients
	# Sm::Vector{Float64}              # Store the scaling profile for the current and previous system lengths
	# ssDists::Matrix{Float64}         # Store the final steady-state morphogen and expander distributions for each system size and production rate

end

# Define struct for storing combined simulation data for all system sizes
mutable struct CombSimData

	seed::UInt                       # Store the seeds for each set of random number generation
	paras::Vector{Float64}           # Store input parameters (p), including Lcrit as the final element
	initDists::Matrix{Float64}       # Store the steady-state morphogen and expander distributions at the initial system size
    path::Vector{SimData}            # Store all data for each individual simulation at each system size
    LCrit::Float64                   # Store the critical length of the system

end

# Define function that solves the full system for the input parameters defined in the vector pNew
function solveSystem(nSeed::UInt, pNew::Vector{Float64}, prob, LCurr::Float64)

	# Zero flags and arrays storing whether system reached a NESS or a linear expander growth state
	global nessFlag = 0
	global tPeakMPrev = 10.
	global uPeakMPrev = 0.
	global typePeakMPrev = -1
	empty!(uPeakM1)
	empty!(uPeakT1)
	empty!(uPeakM2)
	empty!(uPeakT2)

	# Zero flags and times used to decide whether system has reached steady state
	global flagSS = 0
    global tAnySS = 0.
	global tPrevSS = 0.
	global mTotPrevSS = 0. # These two definitions are technically not required here
	global mTotDevMax = 0.
		
    # Initialise grid spacing for current tissue length
    dx = LCurr / N1



	####################################
	# Solve problem                    #
	####################################

	# Solve ODE to output results
	# println([pNew[1] / (dx ^ 2), pNew[2] / (dx ^ 2), pNew[3:5]..., pNew[6] * delM, pNew[7] * delE])
	sol = solve(remake(prob, u0 = uInit, p = (pNew[1] / (dx ^ 2), pNew[2] / (dx ^ 2), pNew[3:5]..., pNew[6] * delM, pNew[7] * delE)),
				Rodas4P(),
				maxiters = 1e6, 
				save_everystep = false, 
				verbose = false, 
				dtmin = dtMin, 
				force_dtmin = false,
				callback = cbSet, # isoutofdomain = (y, p, t) -> any(x -> x < 0, y)
				abstol = absTol, 
				reltol = relTol)

	# Check if system was terminated before solver reached MaxIters
	if(sol.retcode != :MaxIters)

		# Store the time to reach steady-state
		tFin::Float64 = deepcopy(sol.t[end])
	
		sol = solve(remake(prob, u0 = uInit, p = (pNew[1] / (dx ^ 2), pNew[2] / (dx ^ 2), pNew[3:7]...), tspan = (0., sol.t[end])),
					Rodas4P(),
					maxiters = 1e6,
					save_everystep = true,
					verbose = false,
					dtmin = dtMin,
					force_dtmin = false,
					# callback = cbSet, # No callbacks necessary as simulation has fixed end time
					abstol = absTol,
					reltol = relTol)

		# Calculate average oscillation frequency, amplitude and phase shift
		@views oscTimeAmpParas::Vector{Float64} = calculateOscParas(deepcopy(sol.t), 
																	(dx / 2.) .* [trapezium(sol.u[i][:, 1]) for i in eachindex(sol.u)], 
																	(dx / 2.) .* [trapezium(sol.u[i][:, 2]) for i in eachindex(sol.u)], 
																	length(sol.u)) # ,
																	# nessFlag)
		# oscTimeAmpParas::Vector{Float64} = []

		# Check whether scaling can be calculated (L > LInit)
		#= Sm = Vector{Float64}(undef, 0)
		if((nessFlag == 0) && (LCurr > lSave + eps))
			Sm = calculateLocalScalingOriginal(uInit[:, 1], sol.u[end][:, 1], LCurr, LCurr / N1)
		end =#
		# println(Sm)
		
		# Store steady-state gradients as initial condition for next loop
		@views global uInit = deepcopy(sol.u[end]) # .+ (rand(N, 2) .* 1e2)

		# Save time-dependent distributions at regular intervals
		if((saveAllDists == 1) && (LCurr in LSave))
        	println("\n   Saving distributions...")

			# Remove any times and distributions that do not correspond to consistent timesteps
			nTSave::Int = length(sol.t)
			#= @inbounds for i in nTSave:-1:1

				if(abs((sol.t[i] / dtReg) - round(sol.t[i] / dtReg)) > 1e-2)
					deleteat!(sol.t, i)
					deleteat!(sol.u, i)
				end

			end

			# Remove any repeats
			nTSave = length(sol.t)
			dtReg::Float64 = 1e-4 # Reasonable limit for nearby data points
			@inbounds for i in nTSave:-1:2

				if(round(sol.t[i] / dtReg) == round(sol.t[i - 1] / dtReg))
					deleteat!(sol.t, i)
					deleteat!(sol.u, i)
				end

			end =#

			# Check whether corresponding numerical simulation is available
			# nTSave = length(sol.t)
			currSol = Vector{Matrix{Float64}}(undef, nTSave)
			if(compareSimNum == 1)

				# Load numerical solution from file
				currNumSol::Matrix{Float64} = load_object(folder * "/Data_Num/$(nSeed)_L$(Int(round((LCurr - lInit) / dL)))_Dists.jld2")
				
				# Calculate difference between time-dependent simulation solution and numerical scolution at the fixed point
				@inbounds for i in 1:nTSave
					currSol[i] = sol.u[i] .- currNumSol
				end

			else

				# Else just use output of simulation
				currSol = sol.u

			end

			# Locate maxima and minima of the morphogen and expander distributions
			mMaxPos::Vector{Float64} = - ones(nTSave)
			mMinPos::Vector{Float64} = - ones(nTSave)
			eMaxPos::Vector{Float64} = - ones(nTSave)
			eMinPos::Vector{Float64} = - ones(nTSave)
			@inbounds for i in 1:nTSave

				@inbounds for j in 3:N2

					# For each case look for concentrations larger or smaller than all points +- 2
					# Any vector elements that retain -1 did not exhibit a maximum/minimum
					if((currSol[i][j, 1] > currSol[i][j - 1, 1]) && (currSol[i][j - 1, 1] > currSol[i][j - 2, 1]) && (currSol[i][j, 1] > currSol[i][j + 1, 1]) && (currSol[i][j + 1, 1] > currSol[i][j + 2, 1]))
						mMaxPos[i] = j - 1
					end

					if((currSol[i][j, 1] < currSol[i][j - 1, 1]) && (currSol[i][j - 1, 1] < currSol[i][j - 2, 1]) && (currSol[i][j, 1] < currSol[i][j + 1, 1]) && (currSol[i][j + 1, 1] < currSol[i][j + 2, 1]))
						mMinPos[i] = j - 1
					end

					if((currSol[i][j, 2] > currSol[i][j - 1, 2]) && (currSol[i][j - 1, 2] > currSol[i][j - 2, 2]) && (currSol[i][j, 2] > currSol[i][j + 1, 2]) && (currSol[i][j + 1, 2] > currSol[i][j + 2, 2]))
						eMaxPos[i] = j - 1
					end

					if((currSol[i][j, 2] < currSol[i][j - 1, 2]) && (currSol[i][j - 1, 2] < currSol[i][j - 2, 2]) && (currSol[i][j, 2] < currSol[i][j + 1, 2]) && (currSol[i][j + 1, 2] < currSol[i][j + 2, 2]))
						eMinPos[i] = j - 1
					end

				end
			end

			# Convert indexes to positions
			mMaxPos .*= dx
			mMinPos .*= dx
			eMaxPos .*= dx
			eMinPos .*= dx

			# Save results
			save_object(folder * "/Data_Sim/$(nSeed)_L$(Int(round((LCurr - lInit) / dL)))_Dists.jld2", [reduce(hcat, [sol.t, 
																												(dx / 2.) .* [trapezium(sol.u[i][:, 1]) for i in 1:nTSave],
																												(dx / 2.) .* [trapezium(sol.u[i][:, 2]) for i in 1:nTSave],
																												mMaxPos,
																												mMinPos,
																												eMaxPos,
																												eMinPos]), 
																										currSol])
			# save_object(folder * "/Data_Sim/$(nSeed)_L$(Int(round((LCurr - lInit) / dL)))_Dists.jld2", [solSave.t, currSol, mMaxPos, mMinPos, eMaxPos, eMinPos])

		end

		# Return different values for steady-state and NESS flags depending on final simulation states
		if(nessFlag == 0) # Simulation completed normally

			# println("      Stan. dev. = $(std(uInit[:, 1]))")
			return SimData(1, 0, oscTimeAmpParas, tFin, [(uInit[1, 1] - uInit[end, 1]) / uInit[1, 1], (uInit[end, 2] - uInit[1, 2]) / uInit[end, 2]]) # , Sm

		elseif(nessFlag == 1) # Indicate NESS was reached (NESS and linear growth states are mutually exclusive)

			return SimData(0, 1, oscTimeAmpParas, tFin, [(uInit[1, 1] - uInit[end, 1]) / uInit[1, 1], (uInit[end, 2] - uInit[1, 2]) / uInit[end, 2]]) # , []

		end

	else

		# Did not finish simulation
		println("Reached max iters!")
		return SimData(-1, -1, [], -1., []) # , []

	end

end

# Define the discretized PDE
function fKFeedback!(du, u, p, t)
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
function fKFeedbackAllSource!(du, u, p, t)
	dm, de, num, nue, nueThreshPow, k1, mu = p

	du[1, 1] = (2 * dm * u[2, 1]) - (((2 * dm) + (k1 / (u[1, 2] + delM))) * u[1, 1]) + num
	du[1, 2] = (2 * de * u[2, 2]) - (((2 * de) + (mu / (u[1, 2] + delE))) * u[1, 2]) + (nueThreshPow / (threshPow + (u[1, 1] ^ hillPow)))
	@inbounds for i in 2:N1
		du[i, 1] = (dm * (u[i - 1, 1] + u[i + 1, 1])) - (((2 * dm) + (k1 / (u[i, 2] + delM))) * u[i, 1]) + num
		du[i, 2] = (de * (u[i - 1, 2] + u[i + 1, 2])) - (((2 * de) + (mu / (u[i, 2] + delE))) * u[i, 2]) + (nueThreshPow / (threshPow + (u[i, 1] ^ hillPow)))
	end
	du[N, 1] = (2 * dm * u[N1, 1]) - (((2 * dm) + (k1 / (u[N, 2] + delM))) * u[N, 1]) + num
	du[N, 2] = (2 * de * u[N1, 2]) - (((2 * de) + (mu / (u[N, 2] + delE))) * u[N, 2]) + (nueThreshPow / (threshPow + (u[N, 1] ^ hillPow)))

end

# Define callback condition function that checks for when the derivative of the total morphogen concentration is zero
@views oscCheckMCondition(u, t, integrator) = trapezium(integrator(t, Val{1})[:, 1])

# Define additional affect function with more stringent conditions for NESS
function oscCheckMStringentAffect!(integrator)

	# Extract the total morphogen concentration
	@views mCurr::Float64 = trapezium(integrator.u[:, 1])

	# Check if enough time has passed since last maximum/minimum detection and if the observed concentration has changed sufficiently
	# (Prevents errors since adaptive timestep -> 0 near morphogen maximum/minimum due to maximum rate of change of expander concentration)
	if((integrator.t > tPeakMPrev) && ((max(mCurr, uPeakMPrev) / min(mCurr, uPeakMPrev)) - 1. > relTol))

		# Ensure system is not being checked for steady-state
		if(flagSS == 0)

            # Julia does not have functionality to use second order derivative to identify maxima/minima due to adaptive timestepping
            # Instead alternate the arrays that the peaks are stored in, assuming inflexion points are not possible (but check for these manually)

            # Store current time and peak amplitude
            global tPeakMPrev = integrator.t
            global uPeakMPrev = mCurr

            # Store peak amplitude in alternating arrays
			if((length(uPeakM1) == 0) && (length(uPeakM2) == 0)) # Store first entry
                push!(uPeakM1, uPeakMPrev)
                push!(uPeakT1, tPeakMPrev)
				global typePeakMPrev = 0
				# println("START: $(uPeakMPrev), $(uPeakM1[end]), $(typePeakMPrev)")

			elseif(length(uPeakM2) == 0) # Store second entry
				push!(uPeakM2, uPeakMPrev)
				push!(uPeakT2, tPeakMPrev)
				global typePeakMPrev = 1
				# println("SECOND: $(uPeakMPrev), $(uPeakM1[end]), $(uPeakM2[end]), $(typePeakMPrev)")

			else

				if(uPeakM1[end] > uPeakM2[end])
					# println("$(uPeakMPrev), $(uPeakM1[end]), $(uPeakM2[end]), $(typePeakMPrev)")

					if((uPeakMPrev > uPeakM2[end]) && (typePeakMPrev == 1)) # Maxima
						push!(uPeakM1, uPeakMPrev)
                		push!(uPeakT1, tPeakMPrev)
						global typePeakMPrev = 0

					elseif((uPeakMPrev < uPeakM1[end]) && (typePeakMPrev == 0)) # Minima
						push!(uPeakM2, uPeakMPrev)
                		push!(uPeakT2, tPeakMPrev)
						global typePeakMPrev = 1

					elseif((uPeakMPrev > uPeakM1[end]) && (typePeakMPrev == 0)) # Maxima again
						push!(uPeakM1, uPeakMPrev)
                		push!(uPeakT1, tPeakMPrev)
						global typePeakMPrev = 0

					elseif((uPeakMPrev < uPeakM2[end]) && (typePeakMPrev == 1)) # Minima again
						push!(uPeakM2, uPeakMPrev)
                		push!(uPeakT2, tPeakMPrev)
						global typePeakMPrev = 1

					end

				elseif(uPeakM1[end] < uPeakM2[end])

					if((uPeakMPrev < uPeakM2[end]) && (typePeakMPrev == 1)) # Minima
						push!(uPeakM1, uPeakMPrev)
                		push!(uPeakT1, tPeakMPrev)
						global typePeakMPrev = 0

					elseif((uPeakMPrev > uPeakM1[end]) && (typePeakMPrev == 0)) # Maxima
						push!(uPeakM2, uPeakMPrev)
                		push!(uPeakT2, tPeakMPrev)
						global typePeakMPrev = 1

					elseif((uPeakMPrev < uPeakM1[end]) && (typePeakMPrev == 0)) # Minima again
						push!(uPeakM1, uPeakMPrev)
                		push!(uPeakT1, tPeakMPrev)
						global typePeakMPrev = 0

					elseif((uPeakMPrev > uPeakM2[end]) && (typePeakMPrev == 1)) # Maxima again
						push!(uPeakM2, uPeakMPrev)
                		push!(uPeakT2, tPeakMPrev)
						global typePeakMPrev = 1

					end

				end

			end
			# println("NEW:")
			# println(uPeakM1)
			# println(uPeakM2)
			# println(uPeakT1)
			# println(uPeakT2)

			# Increment stored time to prevent next peak being detected too close
            global tPeakMPrev += 0.1

			# Check if equal (non-zero) numbers of maxima and minima have been detected so far
			nPeakM1::Int = length(uPeakM1)
			nPeakM2::Int = length(uPeakM2)
			if((nPeakM1 > nPeakRepeats) && (nPeakM1 == nPeakM2))

				# Check if net translation occurred
				nPeakM1_repeats::Int = nPeakM1 - nPeakRepeats
				nPeakM2_repeats::Int = nPeakM2 - nPeakRepeats
				if(((uPeakM1[nPeakM1] - uPeakM1[nPeakM1_repeats] > 0.001) && (uPeakM2[nPeakM2] - uPeakM2[nPeakM2_repeats] > 0.001)) || 
					((uPeakM1[nPeakM1] - uPeakM1[nPeakM1_repeats] < -0.001) && (uPeakM2[nPeakM2] - uPeakM2[nPeakM2_repeats] < -0.001)))
					# println("Net translation detected\n")

					# Clear arrays containing stored maxima and minima times and magnitudes
					empty!(uPeakM1)
					empty!(uPeakT1)
					empty!(uPeakM2)
					empty!(uPeakT2)
					global typePeakMPrev = -1

				# Check if there was a large change in oscillation period
				elseif((uPeakT1[nPeakM1] - uPeakT1[nPeakM1 - 1] > 1.1 * (uPeakT1[nPeakM1_repeats + 1] - uPeakT1[nPeakM1_repeats])) ||
					   (uPeakT1[nPeakM1] - uPeakT1[nPeakM1 - 1] < (1. / 1.1) * (uPeakT1[nPeakM1_repeats + 1] - uPeakT1[nPeakM1_repeats])) ||
					   (uPeakT2[nPeakM2] - uPeakT2[nPeakM2 - 1] > 1.1 * (uPeakT2[nPeakM2_repeats + 1] - uPeakT2[nPeakM2_repeats])) ||
					   (uPeakT2[nPeakM2] - uPeakT2[nPeakM2 - 1] < (1. / 1.1) * (uPeakT2[nPeakM2_repeats + 1] - uPeakT2[nPeakM2_repeats])))
					# println("Large change in oscillation period occurred\n")

					# Clear arrays containing stored maxima and minima times and magnitudes
					empty!(uPeakM1)
					empty!(uPeakT1)
					empty!(uPeakM2)
					empty!(uPeakT2)
					global typePeakMPrev = -1

				else

					# Compare ratios between peak heights in time to check if system has reached NESS
					magCurr::Vector{Float64} = [abs(uPeakM1[i] - uPeakM2[i]) for i in 1:nPeakM1]
					for i in 1:nPeakM1_repeats
			
						if(abs(min(magCurr[end] / magCurr[i], magCurr[i] / magCurr[end]) - 1) < 0.005)
							# println("\nSystem reached inhibitor NESS\n")
							# println(uPeakM1)
							# println(uPeakM2)
							# println(uPeakT1)
							# println(uPeakT2)

							# Terminate solver
							global nessFlag = 1

							# Moved termination of integrator to newTerminateSteadyStateCondition() function

						end
			
					end

				end

			else

				# Check for inflexion points: number of detected maxima should always be within number of minima +/- 1
				if(abs(nPeakM1 - nPeakM2) > 1)
					# println("Inflexion point detected\n")

					# Clear arrays containing stored maxima and minima times and magnitudes
					empty!(uPeakM1)
					empty!(uPeakT1)
					empty!(uPeakM2)
					empty!(uPeakT2)
					global typePeakMPrev = -1

				end

			end

        end

	end

end

# Define updated function that checks for when simulation reaches steady state
function newTerminateSteadyStateCondition(u, t, integrator)

	# Extract distribution and derivative information
	intFromCache = first(get_tmp_cache(integrator))
    DiffEqBase.get_du!(intFromCache, integrator)
	# println("$(t), $(trapezium(integrator.u[:, 1])), $(trapezium(integrator.u[:, 2])) ($(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 1], u[:, 1]))), $(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 2], u[:, 2]))))")

	# First check if distributions have reached steady state
	@views if(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 1], u[:, 1])) ||
			  any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 2], u[:, 2])))

        # Check if system is in an NESS state
        if(nessFlag == 1)
            # println("\nSystem reached morphogen NESS at check 1 at t = $(t)\n")
            terminate!(integrator)
        end

		# Reset steady state flag and time
		#= if(flagSS > 0)
			println("\nSimulated $(flagSS) points before deviation at t = $(t), Mtot = $(trapezium(integrator.u[:, 1])) ($(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 1], u[:, 1]))), $(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 2], u[:, 2]))))\n")
		end =#
		# println("Check 1 at t = $(t) ($(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 1], u[:, 1]))), $(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 2], u[:, 2]))))")

		global flagSS = 0
		global tAnySS = 0.
		global tPrevSS = 0.
		global mTotPrevSS = 0. # These two definitions are technically not required here
		global mTotDevMax = 0.
		return false
		
	# If system reached steady state check if this is the first time
	elseif(flagSS == 0)

		# Change flag to indicate steady state has been reached and store current time
		# println("\nCheck 2 at t = $(t), Mtot = $(trapezium(integrator.u[:, 1])) ($(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 1], u[:, 1]))), $(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 2], u[:, 2]))))\n")
		global flagSS = 1
		global tAnySS = t
		global tPrevSS = t
		@views global mTotPrevSS = trapezium(integrator.u[:, 1])
		global mTotDevMax = copy(mTotPrevSS)
		return false

	# If system has reached steady state but not for long enough yet
	# elseif(t < tPrevSS + ssCheckTime)
	elseif((t < tPrevSS + ssCheckTime) || (flagSS < nSSCheck))

		# println("\nCheck 3 at t = $(t), Mtot = $(trapezium(integrator.u[:, 1])) ($(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 1], u[:, 1]))), $(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 2], u[:, 2]))))\n")
		global flagSS += 1
		@views if(abs(trapezium(integrator.u[:, 1]) - mTotPrevSS) > abs(mTotDevMax - mTotPrevSS))
			@views global mTotDevMax = trapezium(integrator.u[:, 1])
		end
		return false

	# If system has reached steady state for a prolonged period but has varied slowly during this time so as not to be detected
	elseif((t >= tPrevSS + ssCheckTime) && (flagSS >= nSSCheck) && (abs(mTotDevMax - mTotPrevSS) / mTotPrevSS > 0.01))

		# println("\nRestarting steady-state checks at t = $(t), Mtot = $(trapezium(integrator.u[:, 1])) ($(abs(mTotDevMax - mTotPrevSS) / mTotPrevSS) > 0.01)\n")
		# println("\nCheck 4 at t = $(t), Mtot = $(trapezium(integrator.u[:, 1])) ($(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 1], u[:, 1]))), $(any((abs(deriv) > absTol) && (abs(deriv) > relTol * abs(val)) for (deriv, val) in zip(intFromCache[:, 2], u[:, 2]))))\n")
		global flagSS = 1
		global tPrevSS = t
		@views global mTotPrevSS = trapezium(integrator.u[:, 1])
		global mTotDevMax = copy(mTotPrevSS)
		return false
	
	# If system has reached steady state for a prolonged period
	# elseif(t >= tPrevSS + ssCheckTime)
	elseif((t >= tPrevSS + ssCheckTime) && (flagSS >= nSSCheck))

		# println("\nCheck 5 at t = $(t) > $(tPrevSS + ssCheckTime), Mtot = $(trapezium(integrator.u[:, 1])) ($(abs(mTotDevMax - mTotPrevSS) / mTotPrevSS) < 0.01)\n")
		return true

	end

end

# Define updated function to terminate the solver when steady state has been reached
newTerminateSteadyStateAffect!(integrator) = terminate!(integrator)

# Define function to calculate an integral using the trapezium rule
@views trapezium(u) = u[1] + u[end] + (2 * sum(u[2:end - 1])) # Only used for in-code, dimensionless functions, to calculate actual integrals need factor of dx/2

# Define quadratic function for fitting
quadFitFunc(x, p) = (p[1] .* ((x .- p[4]) .^ 2)) .+ (p[2] .* (x .- p[4])) .+ p[3];

# Define function to calculate the oscillation frequency of a solution
function calculateOscParas(t::Vector{Float64}, solM::Vector{Float64}, solE::Vector{Float64}, nPoints::Int) # , nessFlag::Int)

    # Remove repeats generated when using adaptive time-stepping
    @inbounds for i in nPoints - 1:-1:1
        if(abs(solM[i] - solM[i + 1]) <= 1e-4)
            deleteat!(t, i)
            deleteat!(solM, i)
            deleteat!(solE, i)
        end
    end
    nPoints = length(solM)

	# Loop over all times and store the indexes corresponding to the last nOscs maxima and minima
	mMaxInds::Vector{Int} = -ones(Int, nOscs)
    nMMax::Int = 0
	mMinInds::Vector{Int} = -ones(Int, nOscs)
    nMMin::Int = 0
	eMaxInds::Vector{Int} = -ones(Int, nOscs)
    nEMax::Int = 0
	eMinInds::Vector{Int} = -ones(Int, nOscs)
    nEMin::Int = 0
    @inbounds for i in nPoints - 2:-1:3

		# Check for morphogen peaks
        if((solM[i] > solM[i - 1]) && (solM[i - 1] > solM[i - 2]) && (solM[i] > solM[i + 1]) && (solM[i + 1] > solM[i + 2]))
            if(nMMax < nOscs)
                nMMax += 1
                mMaxInds[nOscs - nMMax + 1] = i
            end
		elseif((solM[i] < solM[i - 1]) && (solM[i - 1] < solM[i - 2]) && (solM[i] < solM[i + 1]) && (solM[i + 1] < solM[i + 2]))
            if(nMMin < nOscs)
                nMMin += 1
                mMinInds[nOscs - nMMin + 1] = i
            end
		end

        # Repeat for expander peaks
        if((solE[i] > solE[i - 1]) && (solE[i - 1] > solE[i - 2]) && (solE[i] > solE[i + 1]) && (solE[i + 1] > solE[i + 2]))
            if(nEMax < nOscs)
                nEMax += 1
                eMaxInds[nOscs - nEMax + 1] = i
            end
        elseif((solE[i] < solE[i - 1]) && (solE[i - 1] < solE[i - 2]) && (solE[i] < solE[i + 1]) && (solE[i + 1] < solE[i + 2]))
            if(nEMin < nOscs)
                nEMin += 1
                eMinInds[nOscs - nEMin + 1] = i
            end
		end

        # Check if nOscs maxima and minima have been found for each condition
        if((nMMax == nOscs) && (nMMin == nOscs) && (nEMax == nOscs) && (nEMin == nOscs))
            break
        end

    end

    # Delete undefined vector elements
    #= println(mMaxInds)
    println(mMinInds)
    println(eMaxInds)
    println(eMinInds) =#
    @inbounds for i in nOscs:-1:1

        if(mMaxInds[i] < 0)
            deleteat!(mMaxInds, i)
        end
        if(mMinInds[i] < 0)
            deleteat!(mMinInds, i)
        end
        if(eMaxInds[i] < 0)
            deleteat!(eMaxInds, i)
        end
        if(eMinInds[i] < 0)
            deleteat!(eMinInds, i)
        end

    end
    #= println(mMaxInds)
    println(mMinInds)
    println(eMaxInds)
    println(eMinInds) =#

	# Store the amplitudes and timings of morphogen and expander maxima and minima
    mMaxConc = Vector{Float64}(undef, nMMax)
    mMinConc = Vector{Float64}(undef, nMMin)
    tMMax = Vector{Float64}(undef, nMMax)
    eMaxConc = Vector{Float64}(undef, nEMax)
    eMinConc = Vector{Float64}(undef, nEMin)
	tEMax = Vector{Float64}(undef, nEMax)

	# Check if enough morphogen maxima were observed
	if(nMMax > 1)

		# Loop over maxima indexes
		counter::Int = 0
		@inbounds for i in mMaxInds

			# Fit peak with quadratic
			solFit = LsqFit.curve_fit(quadFitFunc, t[i - 2:i + 2] .- t[i], solM[i - 2:i + 2], [(solM[i + 2] - solM[i]) / ((t[i + 2] - t[i]) ^ 2), 0., solM[i], 0.])

            # Store maxima parameters
			counter += 1
            mMaxConc[counter] = quadFitFunc(solFit.param[4], solFit.param)
            tMMax[counter] = t[i] + solFit.param[4]

		end

	end

	# Repeat for morphogen minima
	if(nMMin > 1)

		counter = 0
		@inbounds for i in mMinInds

			solFit = LsqFit.curve_fit(quadFitFunc, t[i - 2:i + 2] .- t[i], solM[i - 2:i + 2], [(solM[i + 2] - solM[i]) / ((t[i + 2] - t[i]) ^ 2), 0., solM[i], 0.])
			counter += 1
            mMinConc[counter] = quadFitFunc(solFit.param[4], solFit.param)

		end

	end

	# Repeat for expander maxima
	if(nEMax > 1)

		counter = 0
		@inbounds for i in eMaxInds

			solFit = LsqFit.curve_fit(quadFitFunc, t[i - 2:i + 2] .- t[i], solE[i - 2:i + 2], [(solE[i + 2] - solE[i]) / ((t[i + 2] - t[i]) ^ 2), 0., solE[i], 0.])
			counter += 1
            eMaxConc[counter] = quadFitFunc(solFit.param[4], solFit.param)
            tEMax[counter] = t[i] + solFit.param[4]

		end

	end

	# Repeat for expander minima
	if(nEMin > 1)

		# Loop over minima indexes
		counter = 0
		@inbounds for i in eMinInds

			solFit = LsqFit.curve_fit(quadFitFunc, t[i - 2:i + 2] .- t[i], solE[i - 2:i + 2], [(solE[i + 2] - solE[i]) / ((t[i + 2] - t[i]) ^ 2), 0., solE[i], 0.])
			counter += 1
            eMinConc[counter] = quadFitFunc(solFit.param[4], solFit.param)

		end

	end

	# Initialise parameter values
	meanMAmp::Float64 = -1.
	stdMAmp::Float64 = -1.
	meanTMPeak::Float64 = -1.
	stdTMPeak::Float64 = -1.
	meanEAmp::Float64 = -1.
	stdEAmp::Float64 = -1.
	meanTEPeak::Float64 = -1.
	stdTEPeak::Float64 = -1.
	meanTDelPeak::Float64 = -1.

	# Check whether to calculate average or output the final value
	# if(nessFlag == 1)

    # Calculate the average morphogen and expander oscillation amplitudes and periods
	if((nMMax > 1) && (nMMin > 1))
		meanMAmp = mean(mMaxConc[1:nMMax]) - mean(mMinConc[1:nMMin])
		stdMAmp = sqrt((std(mMaxConc[1:nMMax]) ^ 2) + (std(mMinConc[1:nMMin]) ^ 2))
		meanTMPeak = mean([tMMax[i] - tMMax[i - 1] for i in 2:nMMax])
		stdTMPeak = std([tMMax[i] - tMMax[i - 1] for i in 2:nMMax])
	end
	if((nEMax > 1) && (nEMin > 1))
		meanEAmp = mean(eMaxConc[1:nEMax]) - mean(eMinConc[1:nEMin])
		stdEAmp = sqrt((std(eMaxConc[1:nEMax]) ^ 2) + (std(eMinConc[1:nEMin]) ^ 2))
		meanTEPeak = mean([tEMax[i] - tEMax[i - 1] for i in 2:nEMax])
		stdTEPeak = std([tEMax[i] - tEMax[i - 1] for i in 2:nEMax])
	end

	# Calculate the expander peak lag behind the morphogen peak
	if((nMMax > 1) && (nEMax > 1))

        # For simplicity locate pairs of morphogen and expander maxima times where the expander time comes first
        if(tEMax[1] > tMMax[1])
            tPairs = [[tMMax[i + 1], tEMax[i]] for i in 1:min(nMMax - 1, nEMax)]
        else
            tPairs = [[tMMax[i], tEMax[i]] for i in 1:min(nMMax, nEMax)]
        end
        # println(tPairs)
        if(length(tPairs) > 0)
    		meanTDelPeak = mean([tPairs[i][1] - tPairs[i][2] for i in eachindex(tPairs)])
        end
	end

    # Output the average morphogen oscillation frequency, morphogen oscillation amplitude, and relative morphogen-expander phase-shift
	# println("Amplitude = $(meanMAmp) +/- $(stdMAmp)")
	# println("Frequency = $(1. / meanTMPeak) +/- $(abs((1. / (meanTMPeak + stdTMPeak)) - (1. / meanTMPeak)))")
    return [meanMAmp, stdMAmp, meanTMPeak, stdTMPeak, meanEAmp, stdEAmp, meanTEPeak, stdTEPeak, meanTDelPeak]

end

# Define function to calculate the average scaling around specific points
#= function calculateLocalScalingOriginal(arrL1::Vector{Float64}, arrL2::Vector{Float64}, L2::Float64, dxL2::Float64)

    # Initialise array to store the average local scaling around each pre-specified point
    delPos = Vector{Float64}(undef, nBIndexes)

    # Extract local concentrations pre-growth
    @views valL1::Vector{Float64} = arrL1[bIndexes]

    # Loop over pre-specified points
    @inbounds for i in 1:nBIndexes

        # Locate boundary position in new array
        indCurr::Int = 0
        @inbounds for j in sIndex1:N1
            if(((arrL2[j] >= valL1[i]) && (arrL2[j + 1] <= valL1[i])) || ((arrL2[j] <= valL1[i]) && (arrL2[j + 1] >= valL1[i])))
                indCurr = j
                break
            end
        end
        # println(indCurr)

        # Perform linear interpolation
        if(indCurr != 0)

            m = (arrL2[indCurr + 1] - arrL2[indCurr]) / dxL2
            c = ((arrL2[indCurr] + arrL2[indCurr + 1]) - (m * (((2. * indCurr) - 1) * dxL2))) / 2.

            # Calculate absolute difference between positions before and after change
            delPos[i] = (valL1[i] - c) / m
            # println(delPos[i])
			# println((delPos[i] - (beta * L2)) / (L2 * (1 - beta)))
			# println(posL1[i])
			delPos[i] = abs(((delPos[i] - (beta * L2)) / (L2 * (1 - beta))) - posL1[i])
            # println(delPos[i])

        else
            
            # System cannot scale
            delPos[i] = -1.

        end

    end

	# Return the local changes in position
	return delPos

end =#
