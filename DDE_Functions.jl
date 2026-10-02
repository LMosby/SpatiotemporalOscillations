#####################################
# List of functions for DDE.jl      #
#####################################

# Define struct for storing simulation data for individual delay magnitude
struct DDEData

	ssReached::Int                   # Store whether system reached steady-state for each system size and production rate
	NESS::Int                        # Store whether the system reached an NESS at each system size
    oscTimeAmpParas::Vector{Float64} # Store the oscillation timescale and amplitude for the inhibitor
    tSS::Float64                     # Store the time taken for each system to reach steady-state
	tVec::Vector{Float64}            # Store the full set of times simulated
	tDists::Vector{Vector{Float64}}  # Store the full time-dependent solution

end

# Define struct for storing combined simulation data for all delay magnitudes
mutable struct AllDDEData

	paras::Vector{Float64}           # Store input parameters (p)
    path::Vector{DDEData}            # Store all data for each individual simulation at each system size
    wCrit::Float64                   # Store the oscillation frequency at the critical point for the system
    tauCrit::Float64                 # Store the critical delay magnitude for the system
	numSol::Vector{Float64}          # Store the numerical solution at the fixed point
	jac::Matrix{Float64}             # Store the Jacobian used to calculate the analytical values of the critical delay and oscillation frequency
	wCritAna::Float64                # Store the analytical oscillation frequency at the critical point for the system
    tauCritAna::Float64              # Store the analytical critical delay magnitude for the system

end

# Define the non-linear equations used to calculate the steady-state
function fNL!(du, u, p)
	
	du[1] = ((p[3] * (u[2] - 1.)) ^ (1. / hillPowN)) - u[1]
	du[2] = ((p[2] * (((p[1] + (u[1] ^ hillPowM)) / (p[1] * u[1])) - 1.)) ^ (1. / hillPowH)) - u[2]

end

# Define the discretized ODE without delay
function f!(du, u, p, t)
	Am, Bh, Gn, sigma = p

	du[1] = sigma * (- ((Am * u[1]) / (Am + (u[1] ^ hillPowM))) + (Bh / (Bh + (u[2] ^ hillPowH))))
	du[2] = 1. - ((Gn * u[2]) / (Gn + (u[1] ^ hillPowN)))
	# println(du)

end

# Define the discretized ODE with delay
function fDDE!(du, u, h, p, t)
	Am, Bh, Gn, sigma = p

	hist = h(p, t - currTau)

	du[1] = sigma * (- ((Am * u[1]) / (Am + (u[1] ^ hillPowM))) + (Bh / (Bh + (hist[2] ^ hillPowH))))
	du[2] = 1. - ((Gn * u[2]) / (Gn + (hist[1] ^ hillPowN)))
	# println(du)

end

# Define function to calculate analytical estimate for bifurcation point
function analyticalBifurcation(ssVals::Vector{Float64}, p::Vector{Float64})
	ssA, ssI = ssVals
	Am, Bh, Gn, sigma = p

	# Store Jacobian
	jac::Matrix{Float64} = [sigma * Am * ((((hillPowM - 1.) * (ssA ^ hillPowM)) - Am) / ((Am + (ssA ^ hillPowM)) ^ 2));
							- sigma * ((hillPowH * Bh * (ssI ^ (hillPowH - 1))) / ((Bh + (ssI ^ hillPowH)) ^ 2));;
							(hillPowN * Gn * (ssA ^ (hillPowN - 1)) * ssI) / ((Gn + (ssA ^ hillPowN)) ^ 2);
							- Gn / (Gn + (ssA ^ hillPowN))]
	absJac::Matrix{Float64} = abs.(jac)
	
	# Calculate critical oscillation frequency
	w2Crit::Float64 = - (((absJac[1, 1] ^ 2) + (absJac[2, 2] ^ 2)) / 2.) + sqrt(((((absJac[1, 1] ^ 2) - (absJac[2, 2] ^ 2)) / 2.) ^ 2) + ((absJac[1, 2] ^ 2) * (absJac[2, 1] ^ 2)))
	if(w2Crit < 0.)
		println("System does not exhibit a real critical osc. freq.")
		return -1., -1.
	end
	wCrit::Float64 = sqrt(w2Crit)
	# println("Critical osc. freq. = $(wCrit / (2. * pi))")

	# Check if a Hopf bifurcation is possible
	# if(w2Crit < absJac[1, 1] * absJac[2, 2])
	# 	println("System cannot exhibit a Hopf bifurcation")
	# 	return -1., -1.
	# end

	# Calculate critical delay
	# Factor of 0.5 as this formula is for conversion from total delay
	tauCrit::Float64 = atan((wCrit * (absJac[1, 1] + absJac[2, 2])), (w2Crit - (absJac[1, 1] * absJac[2, 2]))) / (2. * wCrit)
	# println("Critical delay      = $(tauCrit)")

	return jac, wCrit  / (2. * pi), tauCrit

end

# Define function that solves the full system for the input parameters defined in the vector pNew
function solveSystem(pNew::Vector{Float64}, prob)

	# Zero flags and arrays storing whether system reached a NESS or a linear expander growth state
	global nessFlag = 0
	global tPeakIPrev = 10.
	global uPeakIPrev = 0.
	global typePeakIPrev = -1
	empty!(uPeakI1)
	empty!(uPeakT1)
	empty!(uPeakI2)
	empty!(uPeakT2)

	# Zero flags and times used to decide whether system has reached steady state
	global flagSS = 0
    global tAnySS = 0.
	global tPrevSS = 0.
	global aPrevSS = 0.
	global aDevMax = 0.
	global iPrevSS = 0.
	global iDevMax = 0.



	####################################
	# Solve problem                    #
	####################################

	# Solve ODE to output results
	sol = solve(remake(prob, u0 = uInit, p = pNew),
				# Rodas4P(),
				# MethodOfSteps(QNDF(), constrained = true),
				MethodOfSteps(Tsit5()),
				maxiters = 1e6, 
				save_everystep = false, 
				verbose = false, 
				force_dtmin = false,
				callback = cbSet,
				# isoutofdomain = (y, p, t) -> any(x -> x < 0, y),
				abstol = absTol, 
				reltol = relTol)
	# println("END:")
	# println(sol.t)
	# println("      t = $(sol.t[end]), u = $(sol.u[end])")
	#= for i in eachindex(sol.u)
		println(sol.u[i])
	end =#

	# Check if system was terminated before solver reached MaxIters
	if(sol.retcode != :MaxIters)

		# Store the time to reach steady-state
		tFin::Float64 = deepcopy(sol.t[end])
		
		# Store steady-state gradients as initial condition for next loop
		# @views global uInit = deepcopy(sol.u[end])

		# Calculate average oscillation frequency and amplitude
		meanOscT::Float64 = -1.
		meanOscAmp::Float64 = -1.
		nPeak1::Int = length(uPeakT1)
		nPeak2::Int = length(uPeakT2)
		if((nPeak1 > 1) && (nPeak2 > 1))
			minDelInd::Int = minimum([9, nPeak1 - 1, nPeak2 - 1])
			nessFlag == 1 && println(uPeakT1[nPeak1 - minDelInd:nPeak1])
			nessFlag == 1 && println(uPeakT2[nPeak2 - minDelInd:nPeak2])
			meanOscT = abs(mean(uPeakT1[nPeak1 - minDelInd + 1:nPeak1] .- uPeakT1[nPeak1 - minDelInd:nPeak1 - 1]))
			meanOscAmp = abs(mean(uPeakI1[nPeak1 - minDelInd:nPeak1] .- uPeakI2[nPeak2 - minDelInd:nPeak2]))
		end

		# Return different values for steady-state and NESS flags depending on final simulation states
		if(nessFlag == 0) # Simulation completed normally

			return DDEData(1, 0, [meanOscT, meanOscAmp], tFin, sol.t, sol.u)

		elseif(nessFlag == 1) # Indicate NESS was reached (NESS and linear growth states are mutually exclusive)

			return DDEData(0, 1, [meanOscT, meanOscAmp], tFin, sol.t, sol.u)

		end

	else

		# Did not finish simulation
		println("Reached max iters!")
		return DDEData(-1, -1, [], -1., [], Vector{Matrix{Float64}}(undef, 0))

	end

end

# Define callback condition function that checks for when the derivative of the total inhibitor concentration is zero
@views oscCheckMCondition(u, t, integrator) = integrator(t, Val{1})[2]

# Define additional affect function with more stringent conditions for NESS
function oscCheckMStringentAffect!(integrator)

	# Extract the total inhibitor concentration
	@views iCurr::Float64 = deepcopy(integrator.u[2])

	# Check if enough time has passed since last maximum/minimum detection and if the observed concentration has changed sufficiently
	# (Prevents errors since adaptive timestep -> 0 near inhibitor maximum/minimum due to maximum rate of change of expander concentration)
	if((integrator.t > tPeakIPrev) && ((max(iCurr, uPeakIPrev) / min(iCurr, uPeakIPrev)) - 1. > relTol))

		# Ensure system is not being checked for steady-state
		if(flagSS == 0)

            # Julia does not have functionality to use second order derivative to identify maxima/minima due to adaptive timestepping
            # Instead alternate the arrays that the peaks are stored in, assuming inflexion points are not possible (but check for these manually)

            # Store current time and peak amplitude
            global tPeakIPrev = integrator.t
            global uPeakIPrev = iCurr

            # Store peak amplitude in alternating arrays
			if((length(uPeakI1) == 0) && (length(uPeakI2) == 0)) # Store first entry
                push!(uPeakI1, uPeakIPrev)
                push!(uPeakT1, tPeakIPrev)
				global typePeakIPrev = 0
				# println("START: $(uPeakIPrev), $(uPeakI1[end]), $(typePeakIPrev)")

			elseif(length(uPeakI2) == 0) # Store second entry
				push!(uPeakI2, uPeakIPrev)
				push!(uPeakT2, tPeakIPrev)
				global typePeakIPrev = 1
				# println("SECOND: $(uPeakIPrev), $(uPeakI1[end]), $(uPeakI2[end]), $(typePeakIPrev)")

			else

				if(uPeakI1[end] > uPeakI2[end])
					# println("$(uPeakIPrev), $(uPeakI1[end]), $(uPeakI2[end]), $(typePeakIPrev)")

					if((uPeakIPrev > uPeakI2[end]) && (typePeakIPrev == 1)) # Maxima
						push!(uPeakI1, uPeakIPrev)
                		push!(uPeakT1, tPeakIPrev)
						global typePeakIPrev = 0

					elseif((uPeakIPrev < uPeakI1[end]) && (typePeakIPrev == 0)) # Minima
						push!(uPeakI2, uPeakIPrev)
                		push!(uPeakT2, tPeakIPrev)
						global typePeakIPrev = 1

					elseif((uPeakIPrev > uPeakI1[end]) && (typePeakIPrev == 0)) # Maxima again
						push!(uPeakI1, uPeakIPrev)
                		push!(uPeakT1, tPeakIPrev)
						global typePeakIPrev = 0

					elseif((uPeakIPrev < uPeakI2[end]) && (typePeakIPrev == 1)) # Minima again
						push!(uPeakI2, uPeakIPrev)
                		push!(uPeakT2, tPeakIPrev)
						global typePeakIPrev = 1

					end

				elseif(uPeakI1[end] < uPeakI2[end])

					if((uPeakIPrev < uPeakI2[end]) && (typePeakIPrev == 1)) # Minima
						push!(uPeakI1, uPeakIPrev)
                		push!(uPeakT1, tPeakIPrev)
						global typePeakIPrev = 0

					elseif((uPeakIPrev > uPeakI1[end]) && (typePeakIPrev == 0)) # Maxima
						push!(uPeakI2, uPeakIPrev)
                		push!(uPeakT2, tPeakIPrev)
						global typePeakIPrev = 1

					elseif((uPeakIPrev < uPeakI1[end]) && (typePeakIPrev == 0)) # Minima again
						push!(uPeakI1, uPeakIPrev)
                		push!(uPeakT1, tPeakIPrev)
						global typePeakIPrev = 0

					elseif((uPeakIPrev > uPeakI2[end]) && (typePeakIPrev == 1)) # Maxima again
						push!(uPeakI2, uPeakIPrev)
                		push!(uPeakT2, tPeakIPrev)
						global typePeakIPrev = 1

					end

				end

			end
			# println("NEW:")
			# println(uPeakI1)
			# println(uPeakI2)
			# println(uPeakT1)
			# println(uPeakT2)

			# Increment stored time to prevent next peak being detected too close
            global tPeakIPrev += 0.1

			# Check if equal (non-zero) numbers of maxima and minima have been detected so far
			nPeakI1::Int = length(uPeakI1)
			nPeakI2::Int = length(uPeakI2)
			if((nPeakI1 > nPeakRepeats) && (nPeakI1 == nPeakI2))

				# Check if net translation occurred
				nPeakI1_repeats::Int = nPeakI1 - nPeakRepeats
				nPeakI2_repeats::Int = nPeakI2 - nPeakRepeats
				if(((uPeakI1[nPeakI1] - uPeakI1[nPeakI1_repeats] > 0.001) && (uPeakI2[nPeakI2] - uPeakI2[nPeakI2_repeats] > 0.001)) || 
					((uPeakI1[nPeakI1] - uPeakI1[nPeakI1_repeats] < -0.001) && (uPeakI2[nPeakI2] - uPeakI2[nPeakI2_repeats] < -0.001)))
					# println("Net translation detected\n")

					# Clear arrays containing stored maxima and minima times and magnitudes
					empty!(uPeakI1)
					empty!(uPeakT1)
					empty!(uPeakI2)
					empty!(uPeakT2)
					global typePeakIPrev = -1

				# Check if there was a large change in oscillation period
				elseif((uPeakT1[nPeakI1] - uPeakT1[nPeakI1 - 1] > 1.1 * (uPeakT1[nPeakI1_repeats + 1] - uPeakT1[nPeakI1_repeats])) ||
					   (uPeakT1[nPeakI1] - uPeakT1[nPeakI1 - 1] < (1. / 1.1) * (uPeakT1[nPeakI1_repeats + 1] - uPeakT1[nPeakI1_repeats])) ||
					   (uPeakT2[nPeakI2] - uPeakT2[nPeakI2 - 1] > 1.1 * (uPeakT2[nPeakI2_repeats + 1] - uPeakT2[nPeakI2_repeats])) ||
					   (uPeakT2[nPeakI2] - uPeakT2[nPeakI2 - 1] < (1. / 1.1) * (uPeakT2[nPeakI2_repeats + 1] - uPeakT2[nPeakI2_repeats])))
					# println("Large change in oscillation period occurred\n")

					# Clear arrays containing stored maxima and minima times and magnitudes
					empty!(uPeakI1)
					empty!(uPeakT1)
					empty!(uPeakI2)
					empty!(uPeakT2)
					global typePeakIPrev = -1

				else

					# Compare ratios between peak heights in time to check if system has reached NESS
					magCurr::Vector{Float64} = [abs(uPeakI1[i] - uPeakI2[i]) for i in 1:nPeakI1]
					for i in 1:nPeakI1_repeats
			
						if(abs(min(magCurr[end] / magCurr[i], magCurr[i] / magCurr[end]) - 1) < 0.005)
							# println("\nSystem reached inhibitor NESS\n")
							# println(uPeakI1)
							# println(uPeakI2)
							# println(uPeakT1)
							# println(uPeakT2)

							# Terminate solver
							global nessFlag = 1

							# Moved termination of integrator to newTerminateSTeadyStateCondition() function

						end
			
					end

				end

			else

				# Check for inflexion points: number of detected maxima should always be within number of minima +/- 1
				if(abs(nPeakI1 - nPeakI2) > 1)
					# println("Inflexion point detected\n")

					# Clear arrays containing stored maxima and minima times and magnitudes
					empty!(uPeakI1)
					empty!(uPeakT1)
					empty!(uPeakI2)
					empty!(uPeakT2)
					global typePeakIPrev = -1

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
	# println("$(t), $(u[1]), $(u[2]) ($((abs(intFromCache[1]) > absTol) && (abs(intFromCache[1]) > relTol * u[1])), $((abs(intFromCache[2]) > absTol) && (abs(intFromCache[2]) > relTol * u[2]))")

	# First check if distributions have reached steady state
	@views if(((abs(intFromCache[1]) > absTol) && (abs(intFromCache[1]) > relTol * u[1])) ||
			  ((abs(intFromCache[2]) > absTol) && (abs(intFromCache[2]) > relTol * u[2])))

        # Check if system is in an NESS state
        if(nessFlag == 1)
            # println("\nSystem reached NESS at check 1 at t = $(t)\n")
            terminate!(integrator)
        end

		# Reset steady state flag and time
		#= if(flagSS > 0)
			println("\nSimulated $(flagSS) points before deviation at t = $(t), Mtot = $(u[1]) ($((abs(intFromCache[1]) > absTol) && (abs(intFromCache[1]) > relTol * u[1])), $((abs(intFromCache[2]) > absTol) && (abs(intFromCache[2]) > relTol * u[2]))")
		end =#
		# println("Check 1 at t = $(t) ($((abs(intFromCache[1]) > absTol) && (abs(intFromCache[1]) > relTol * u[1])), $((abs(intFromCache[2]) > absTol) && (abs(intFromCache[2]) > relTol * u[2])))")

		global flagSS = 0
		global tAnySS = 0.
		global tPrevSS = 0.
		global aPrevSS = 0. # These four definitions are technically not required here
		global aDevMax = 0.
		global iPrevSS = 0.
		global iDevMax = 0.
		return false
		
	# If system reached steady state check if this is the first time
	elseif(flagSS == 0)

		# Change flag to indicate steady state has been reached and store current time
		# println("\nCheck 2 at t = $(t), u = ($(u[1]), $(u[2])) ($((abs(intFromCache[1]) > absTol) && (abs(intFromCache[1]) > relTol * u[1])), $((abs(intFromCache[2]) > absTol) && (abs(intFromCache[2]) > relTol * u[2])))")
		global flagSS = 1
		global tAnySS = t
		global tPrevSS = t
		@views global aPrevSS = u[1]
		global aDevMax = copy(aPrevSS)
		@views global iPrevSS = u[2]
		global iDevMax = copy(iPrevSS)
		return false

	# If system has reached steady state but not for long enough yet
	# elseif(t < tPrevSS + ssCheckTime)
	elseif((t < tPrevSS + ssCheckTime) || (flagSS < nSSCheck))

		# println("\nCheck 3 at t = $(t), u = ($(u[1]), $(u[2])) ($((abs(intFromCache[1]) > absTol) && (abs(intFromCache[1]) > relTol * u[1])), $((abs(intFromCache[2]) > absTol) && (abs(intFromCache[2]) > relTol * u[2])))")
		global flagSS += 1
		@views if((u[1] - aPrevSS) > abs(aDevMax - aPrevSS))
			@views global aDevMax = u[1]
		end
		@views if((u[1] - iPrevSS) > abs(iDevMax - iPrevSS))
			@views global iDevMax = u[2]
		end
		return false

	# If system has reached steady state for a prolonged period but has varied slowly during this time so as not to be detected
	elseif((t >= tPrevSS + ssCheckTime) && (flagSS >= nSSCheck) && (abs(aDevMax - aPrevSS) / aPrevSS > 0.01) && (abs(iDevMax - iPrevSS) / iPrevSS > 0.01))

		# println("\nCheck 4 at t = $(t), u = ($(u[1]), $(u[2])) ($((abs(intFromCache[1]) > absTol) && (abs(intFromCache[1]) > relTol * u[1])), $((abs(intFromCache[2]) > absTol) && (abs(intFromCache[2]) > relTol * u[2])))")
		global flagSS = 1
		global tPrevSS = t
		@views global aPrevSS = u[1]
		global aDevMax = copy(aPrevSS)
		@views global iPrevSS = u[2]
		global iDevMax = copy(iPrevSS)
		return false
	
	# If system has reached steady state for a prolonged period
	# elseif(t >= tPrevSS + ssCheckTime)
	elseif((t >= tPrevSS + ssCheckTime) && (flagSS >= nSSCheck))

		# println("\nCheck 5 at t = $(t) > $(tPrevSS + ssCheckTime), u = ($(u[1]), $(u[2])) ($(abs(aDevMax - aPrevSS) / aPrevSS) < 0.01, $(abs(iDevMax - iPrevSS) / iPrevSS) < 0.01)\n")
		return true

	end

end

# Define updated function to terminate the solver when steady state has been reached
newTerminateSteadyStateAffect!(integrator) = terminate!(integrator)
