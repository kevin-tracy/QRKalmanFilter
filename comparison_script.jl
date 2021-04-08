using LinearAlgebra

include(joinpath(@__DIR__,"kalman_filter.jl"))
include(joinpath(@__DIR__,"QRkalman_filter.jl"))

function run_example()

# double integrator discrete dynamics
dt = 0.1
n, m, nu = 6, 6, 3
A = I + [zeros(3,3) dt*I; zeros(3,6)]
B = Array([.5*dt^2*I(3);dt*I(3)])
C = Array(I(6))

# noise covariances
Q = Array(0.0001*I(6))
R = Array(.001*I(6))
kf_sys = (A = A, B = B, C = C, Q = Q, R = R)

# sqrt Kalman Filter
sqrkf_sys = (A = A, B = B, C = C, ΓQ = chol(Q), ΓR = chol(R))

# allocate
N = 1000
X = [zeros(n)   for i = 1:N]
Y = [zeros(m)   for i = 1:N]
U = [zeros(nu)  for i = 1:N]
μ = [zeros(n)   for i = 1:N]
Σ = [zeros(n,n) for i = 1:N]

# initial conditions
X[1] = [1;2;3;0;0;0]
Y[1] = C*X[1] + sqrt(R)*randn(6)

# KF initialization
μ[1] = X[1] + .00001*randn(6)
Σ[1] = .05*I(6)

# QRKF initialization
μsqr = deepcopy(μ)
F = deepcopy(Σ)
F[1] = chol(Σ[1])

# simulation
for i = 1:N-1

    # control input
    t = dt*(i-1)
    U[i] = [sin(t);cos(t);.5*sin(2*t)]

    # propagate dynamics forward one step
    X[i+1] = A*X[i] + B*U[i] + sqrt(Q)*randn(6)

    # take masurement
    Y[i+1] = C*X[i+1] + sqrt(R)*randn(6)

    # Kalman Filter
    μ[i+1], Σ[i+1] = kalman_filter(μ[i],Σ[i],U[i],Y[i+1],kf_sys)

    # sqrt Kalman Filter
    μsqr[i+1], F[i+1] = sqrkalman_filter(μsqr[i],F[i],U[i],Y[i+1],sqrkf_sys)
end

# @infiltrate
# error()
Xm = mat_from_vec(X)
μm = mat_from_vec(μ)
Ym = mat_from_vec(Y)


Ye = abs.(Xm - Ym)
μe = abs.(Xm - μm)

kf_errors = abs.(μm - mat_from_vec(μsqr))

mat"
figure
hold on
plot($kf_errors')
hold off
"

mat"
figure
hold on
title('Position Errors')
plot($Ye(1:3,:)','b')
plot($μe(1:3,:)','r')
"
mat"
figure
hold on
title('Velocity Errors')
plot($Ye(4:6,:)','b')
plot($μe(4:6,:)','r')
"
end

run_example()
