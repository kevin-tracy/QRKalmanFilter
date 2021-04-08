using LinearAlgebra

# square root stuff
function chol(A)
    # returns upper triangular Cholesky factorization of matrix A
    return cholesky(Symmetric(A)).U
end
function qrᵣ(A)
    # QR decomposition of A where only the upper triangular R is returned
    return qr(A).R
end

function sqrkalman_filter(μ,F,u,y,kf_sys)

    μ̄, F̄ = sqrkf_predict(μ,F,u,kf_sys)

    z, L = sqrkf_innovate(μ̄,F̄,y,kf_sys)

    μ₊, F₊ = sqrkf_update(μ̄,F̄,z,L,kf_sys)

    return μ₊, F₊
end

function sqrkf_predict(μ,F₋,u,kf_sys)

    # get probem data from kf_sys
    A, B, ΓQ = kf_sys.A, kf_sys.B, kf_sys.ΓQ

    # predict one step
    μ̄ = A*μ + B*u
    F̄ = qrᵣ([F₋*A';ΓQ])

    return μ̄, F̄
end

function sqrkf_innovate(μ̄,F̄,y,kf_sys)

    # get probem data from kf_sys
    A, C, ΓR = kf_sys.A, kf_sys.C, kf_sys.ΓR

    # innovation
    z = y - C*μ̄
    G = qrᵣ([F̄*C';ΓR])

    # kalman gain
    L = ((F̄'*F̄*C')/G)/(G')

    return z, L
end
function sqrkf_update(μ̄,F̄,z,L,kf_sys)

    # problem data
    ΓR, C = kf_sys.ΓR, kf_sys.C

    # update (Joseph form for Σ)
    μ₊= μ̄ + L*z
    F₊= qrᵣ([F̄*(I - L*C)';ΓR*L'])

    return μ₊, F₊
end
