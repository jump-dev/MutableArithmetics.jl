struct MutatingStepRange{T,S} <: OrdinalRange{T,S}
    start::T
    step::S
    stop::T

    function MutatingStepRange{T,S}(start::T, step::S, stop::T) where {T,S}
        return new(start, step, Base.steprange_last(start, step, stop))
    end
end
function MutatingStepRange(start::T, step::S, stop::T) where {T,S}
    return MutatingStepRange{T,S}(start, step, stop)
end
Base.step(r::MutatingStepRange) = r.step
function Base.unsafe_length(r::MutatingStepRange)
    n = Integer(div((r.stop - r.start) + r.step, r.step))
    return isempty(r) ? zero(n) : n
end
Base.length(r::MutatingStepRange) = Base.unsafe_length(r)
function Base.isempty(r::MutatingStepRange)
    return (r.start != r.stop) & ((r.step > zero(r.step)) != (r.stop > r.start))
end

function Base.iterate(r::MutatingStepRange)
    if isempty(r)
        return nothing
    else
        state = copy_if_mutable(first(r))
        return (state, state)
    end
end

function Base.iterate(r::MutatingStepRange{T}, i) where {T}
    i == last(r) && return nothing
    next = convert(T, operate!!(+, i, step(r)))
    return (next, next)
end
