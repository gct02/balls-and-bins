#=
Exact MIP model for the Balls and Bins problem (JuMP + HiGHS).

A bin holding b balls is worth b(b + 1)/2, which is convex in b, so the model
uses one binary variable per ball slot above the lower bound:

    x[i, j] = 1  <=>  bin i holds at least lower[i] + j balls,   j = 1, ..., upper[i] - lower[i]

The (lower[i] + j)-th ball of bin i adds lower[i] + j to the profit, so

    max   sum_i lower[i](lower[i] + 1)/2 + sum_{i,j} (lower[i] + j) x[i, j]
    s.t.  x[i, j] <= x[i, j-1]                     slots fill in order
          sum_{i,j} x[i, j] <= n_balls - sum_i lower[i]   ball budget
          x[i, j] in {0, 1}

The ordering constraints are required: the marginal profit grows with j, so
without them the solver would pick the most valuable slots instead of a prefix.

Usage:
    julia --project=mip mip/balls_and_bins.jl INSTANCE [SEED=42] [TIME_LIMIT=3600]
=#

using JuMP
using HiGHS

struct Instance
    n_balls::Int
    lower::Vector{Int}
    upper::Vector{Int}
end

n_bins(inst::Instance) = length(inst.lower)
capacity(inst::Instance) = inst.upper .- inst.lower
triangular(n) = n * (n + 1) ÷ 2

"""
    read_instance(path) -> Instance

Read `n_bins`, `n_balls`, then one `lower upper` pair per bin (whitespace separated).
"""
function read_instance(path::AbstractString)
    tokens = parse.(Int, split(read(path, String)))
    n, n_balls = tokens[1], tokens[2]
    length(tokens) == 2 + 2n ||
        error("$path: expected $n (lower, upper) pairs, found $((length(tokens) - 2) / 2)")
    bounds = reshape(tokens[3:end], 2, n)
    inst = Instance(n_balls, bounds[1, :], bounds[2, :])
    all(0 .<= inst.lower .<= inst.upper) || error("$path: every bin needs 0 <= lower <= upper")
    sum(inst.lower) <= n_balls || error(
        "infeasible instance: the lower bounds need $(sum(inst.lower)) balls " *
        "but only $n_balls are available",
    )
    return inst
end

"""
    build_model(inst; seed=42, time_limit=3600.0, silent=false) -> (model, x)

Build the slot-based MIP for `inst`; `x[i, j]` is the j-th slot above bin i's lower bound.
"""
function build_model(inst::Instance; seed::Int = 42, time_limit::Real = 3600.0, silent::Bool = false)
    n, cap = n_bins(inst), capacity(inst)

    model = Model(HiGHS.Optimizer)
    set_attribute(model, "random_seed", seed)
    set_time_limit_sec(model, time_limit)
    silent && set_silent(model)

    @variable(model, x[i = 1:n, j = 1:cap[i]], Bin)
    @constraint(model, [i = 1:n, j = 2:cap[i]], x[i, j] <= x[i, j-1])
    @constraint(model, sum(x[i, j] for i in 1:n for j in 1:cap[i]) <= inst.n_balls - sum(inst.lower))
    @objective(model, Max, sum((inst.lower[i] + j) * x[i, j] for i in 1:n for j in 1:cap[i]))
    return model, x
end

"""
    solve(inst; kwargs...) -> NamedTuple

Solve the MIP and return the termination status, the balls per bin and profit of
the best solution found (`nothing` if none), the best bound and the solve time.
"""
function solve(inst::Instance; kwargs...)
    base_profit = sum(triangular, inst.lower; init = 0)
    if sum(capacity(inst)) == 0  # no slots, so no variables: the lower bounds are the solution
        return (; status = MOI.OPTIMAL, balls_per_bin = copy(inst.lower), profit = base_profit,
                bound = float(base_profit), time = 0.0)
    end

    model, x = build_model(inst; kwargs...)
    optimize!(model)

    balls_per_bin, profit = nothing, nothing
    if has_values(model)
        cap = capacity(inst)
        balls_per_bin = [
            inst.lower[i] + sum((round(Int, value(x[i, j])) for j in 1:cap[i]); init = 0)
            for i in 1:n_bins(inst)
        ]
        profit = sum(triangular, balls_per_bin; init = 0)
    end
    bound = base_profit + objective_bound(model)
    return (; status = termination_status(model), balls_per_bin, profit, bound,
            time = solve_time(model))
end

function main(args = ARGS)
    if !(1 <= length(args) <= 3)
        println(stderr, "usage: julia --project=mip mip/balls_and_bins.jl INSTANCE [SEED=42] [TIME_LIMIT=3600]")
        return 1
    end
    path = args[1]
    inst = try
        read_instance(path)
    catch err
        err isa Union{ErrorException, ArgumentError, SystemError} || rethrow()
        println(stderr, "error: ", sprint(showerror, err))
        return 1
    end
    seed = length(args) >= 2 ? parse(Int, args[2]) : 42
    time_limit = length(args) >= 3 ? parse(Float64, args[3]) : 3600.0

    result = solve(inst; seed, time_limit)

    println()
    println("Instance:       $path ($(n_bins(inst)) bins, $(inst.n_balls) balls)")
    println("Status:         $(result.status)")
    if result.profit === nothing
        println("No feasible solution found within the time limit.")
    else
        # The objective is integral, so the floor of the bound is still a valid bound.
        bound = isfinite(result.bound) ? floor(Int, result.bound + 1e-6) : result.bound
        gap = abs(bound - result.profit) / max(abs(result.profit), 1)
        println("Best profit:    $(result.profit)")
        println("Best bound:     $bound")
        println("Gap:            $(round(100gap; digits = 2))%")
        println("Balls per bin:  $(result.balls_per_bin)")
    end
    println("Solve time:     $(round(result.time; digits = 2)) s")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
