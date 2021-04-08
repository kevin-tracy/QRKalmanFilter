using LinearAlgebra


function kalman_filter(μ,Σ,u,y,kf_sys)

    μ̄, Σ̄ = kf_predict(μ,Σ,u,kf_sys)

    z, L = kf_innovate(μ̄,Σ̄,y,kf_sys)

    μ₊, Σ₊ = kf_update(μ̄,Σ̄,z,L,kf_sys)

    return μ₊, Σ₊
end

function kf_predict(μ,Σ,u,kf_sys)

    # get probem data from kf_sys
    A,B,Q = kf_sys.A, kf_sys.B, kf_sys.Q

    # predict one step
    μ̄ = A*μ + B*u
    Σ̄ = A*Σ*A' + Q

    return μ̄, Σ̄
end

function kf_innovate(μ̄,Σ̄,y,kf_sys)

    # get probem data from kf_sys
    A,C,R = kf_sys.A, kf_sys.C, kf_sys.R

    # innovation
    z = y - C*μ̄
    S = C*Σ̄*C' + R

    # kalman gain
    L = (Σ̄*C')/S

    return z, L
end
function kf_update(μ̄,Σ̄,z,L,kf_sys)

    # problem data
    R,C = kf_sys.R,kf_sys.C

    # update (joseph form for Σ)
    μ₊= μ̄ + L*z
    Σ₊= (I - L*C)*Σ̄*(I - L*C)' + L*R*L'

    return μ₊, Σ₊
end
