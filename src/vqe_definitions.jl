# define an ansatz based on the group of 2-qubit centralizers with different propagation coefficients
function vqe_Krylov_centralizer_ansatz(nqbt::Int, dpth::Int)
    Kij = getKij(n, XXbonds, YYbonds, ZZbonds) # get centralizers
    cirq = chain(nqbt)
    for d=1:dpth
        for b=1:length(Kij)
            push!(cirq, time_evolve(Kij[b], 0.0, tol=1e-5, check_hermicity=false)  )
        end
    end
    return cirq
end

function prepare_for_vqe(Lx,Lz,BC,J)
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    w, plaq_seq, vtot = getstabilizers(Lx, Lz, BC, n)
    LXverts, LZverts = getlogicstrings(plaq_seq,Lx,Lz)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    println("We consider n = ", n, " qubits")
    flush(Core.stdout)
    Jxyz = [J, J, J] # interaction constants for Heisenberg terms 
    hmagn = [0.0, 0.0, 0.0] # effective magnetic fields, for VQE set to zero
    h = kitaev(n, XXbonds, YYbonds, ZZbonds, Jxyz, hmagn) #creates hamiltonian
    num_vortices = 0 # fix the number of vortices to consider (we choose 4 for n=8, 0 for n=12,18,24)
    if n==8
        num_vortices = 4
    end
    print_initial = false # choose if want to print properties of initial state (takes time for large system)
    ψ = prepare_inital_state(n, num_vortices, plaq_seq, LXverts, LZverts, print_initial, GPU_enabled,Lx,Lz,BC)
    return ψ,n,h
end

# define an ansatz based on the group of 3-qubit centralizers with different propagation coefficients
function vqe_Krylov_centralizer_product_ansatz(nqbt::Int)
    Lijk, _ = getLijk(Lx, Lz, BC) # get centralizer products as 3-body operators
    cirq = chain(nqbt)
    for b=1:length(Lijk)
        push!(cirq, time_evolve(Lijk[b], 0.0, tol=1e-5, check_hermicity=false)  )
    end
    return cirq
end

function mygate(nqbt, thet)
    return chain(nqbt, put(7=>Rz(thet)), put(9=>Rz(thet)), put(11=>Rz(thet)))
end
# horizontal_ring(nqbt::Int,i::Int,j::Int,k::Int,ϕ::Float64) = chain(nqbt, put(nqbt, i=>Rz(ϕ)), put(nqbt, j=>Rz(ϕ)), put(nqbt, k=>Rz(ϕ)))
# ϕ = 0.0
# i = 1
# j = 2
# k = 3
# nqbt = 10

# horizontal_ring(nqbt::Int,i::Int,j::Int,k::Int,ϕ::Float64) = matblock(mat(chain(nqbt, put(nqbt, i=>Rz(ϕ)), put(nqbt, j=>Rz(ϕ)), put(nqbt, k=>Rz(ϕ)))))
# use centralizers with gate decompositions

function vqe_centralizer_ansatz(nqbt::Int, dpth::Int,Lx,Lz,BC)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    cirq = chain(nqbt)
    # h_verts = 1:2:(2*Lx+2)
    # v_verts = 2:(2*Lx+2):n
    # push!(cirq, horizontal_ring(nqbt, verts[1], verts[2], verts[3], 0.))
    for d=1:dpth
        # # push!(cirq, put(n, 1=>Rz(0.0))  )
        # # push!(cirq, put(n, 2=>Rz(0.0))  )
        # push!(cirq, put(n, 3=>Rz(0.0))  )
        # # push!(cirq, put(n, 4=>Rz(0.0))  )
        # push!(cirq, put(n, 5=>Rz(0.0))  )
        # # push!(cirq, put(n, 6=>Rz(0.0))  )
        # # push!(cirq, put(n, 7=>Rz(0.0))  )
        # push!(cirq, put(n, 9=>Rz(0.0))  )
        # push!(cirq, put(n, 11=>Rz(0.0))  )
        #
        # push!(cirq, put(n, 7=>Ry(0.0))  )
        # push!(cirq, put(n, 9=>Ry(0.0))  )
        # push!(cirq, put(n, 11=>Ry(0.0))  )
        # # push!(cirq, put(n, 8=>Rz(0.0))  )
        # # push!(cirq, repeat(Rz(0.0), [1, 7])  )
        # # push!(cirq, time_evolve(vort_swap_op(1, 2, 3, 4), 0.0, tol=1e-5, check_hermicity=false)  )
        # # push!(cirq, time_evolve(vort_swap_op(1, 3, 2, 4), 0.0, tol=1e-5, check_hermicity=false)  )
        #
        # push!(cirq, repeat(Rz(0.0), h_verts)  )
        # push!(cirq, repeat(Ry(0.0), v_verts)  )
        #
        # # push!(cirq, repeat(Rz(0.0), [1, 3, 5, 7])  )
        # # push!(cirq, repeat(Ry(0.0), [2, 6, 4, 8])  )

        # implement XX(θ) unitaries
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, chain(nqbt, put(XXbonds[i][2]=>Rz(0.)) for i=1:length(XXbonds))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # implement YY(θ) unitaries
        push!(cirq, repeat(S', collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, chain(nqbt, put(YYbonds[i][2]=>Rz(0.)) for i=1:length(YYbonds))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(S, collect(Iterators.flatten(YYbonds)))  )
        # implement ZZ(θ) unitaries
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
        push!(cirq, chain(nqbt, put(ZZbonds[i][2]=>Rz(0.)) for i=1:length(ZZbonds))  )
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
        #additional operators
    end
    # # push!(cirq, mygate(nqbt, 0.))
    # # push!(cirq, chain(nqbt, put(7=>Rz(0.0)), put(9=>Rz(0.0)), put(11=>Rz(0.0))))
    return cirq
end

        # push!(cirq, time_evolve(Sz_plq(1, 2), 0.0, tol=1e-5, check_hermicity=false))
        # push!(cirq, time_evolve(Sz_plq(2, 3), 0.0, tol=1e-5, check_hermicity=false))
        # push!(cirq, time_evolve(Sz_plq(3, 1), 0.0, tol=1e-5, check_hermicity=false))
        # push!(cirq, time_evolve(Sz_plq(4, 5), 0.0, tol=1e-5, check_hermicity=false))
        # push!(cirq, time_evolve(Sz_plq(5, 6), 0.0, tol=1e-5, check_hermicity=false))
        # push!(cirq, time_evolve(Sz_plq(6, 4), 0.0, tol=1e-5, check_hermicity=false))

        # push!(cirq, put(n, 8=>Ry(0.0))  )
        # push!(cirq, put(n, 10=>Ry(0.0))  )
        # push!(cirq, put(n, 12=>Ry(0.0))  )
        # push!(cirq, put(n, 2=>Ry(0.0))  )
        # push!(cirq, put(n, 4=>Ry(0.0))  )
        # push!(cirq, put(n, 6=>Ry(0.0))  )


        # # push!(cirq, repeat(Rz(0.0), [1, 7])  )
        # # push!(cirq, time_evolve(vort_swap_op(1, 2, 3, 4), 0.0, tol=1e-5, check_hermicity=false)  )
        # # push!(cirq, time_evolve(vort_swap_op(1, 3, 2, 4), 0.0, tol=1e-5, check_hermicity=false)  )
        #
        # push!(cirq, repeat(Rz(0.0), h_verts)  )
        # push!(cirq, repeat(Ry(0.0), v_verts)  )
        #
        # # push!(cirq, repeat(Rz(0.0), [1, 3, 5, 7])  )
        # # push!(cirq, repeat(Ry(0.0), [2, 6, 4, 8])  )


function vqe_centralizer_ansatz_vort_cnots(nqbt::Int, dpth::Int)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    cirq = chain(nqbt)
    # h_verts = 1:2:(2*Lx+2)
    # v_verts = 2:(2*Lx+2):n
    # push!(cirq, horizontal_ring(nqbt, verts[1], verts[2], verts[3], 0.))
    for d=1:dpth
        if n==12
            #single-plaquette rotations
            push!(cirq, put(n, 1=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(1, 2), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 1=>Rz(0.0))  )

            push!(cirq, put(n, 3=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(2, 3), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 3=>Rz(0.0))  )

            push!(cirq, put(n, 5=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(3, 1), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 5=>Rz(0.0))  )

            push!(cirq, put(n, 7=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(4, 5), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 7=>Rz(0.0))  )

            push!(cirq, put(n, 9=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(5, 6), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 9=>Rz(0.0))  )

            push!(cirq, put(n, 11=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(6, 4), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 11=>Rz(0.0))  )
        elseif n==8
            push!(cirq, put(n, 3=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(1, 2), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 3=>Rz(0.0))  )

            push!(cirq, put(n, 1=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(2, 1), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 1=>Rz(0.0))  )

            push!(cirq, put(n, 7=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(3, 4), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 7=>Rz(0.0))  )

            push!(cirq, put(n, 5=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(4, 3), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 5=>Rz(0.0))  )
        end
        # #another single-plaquette rotations
        # push!(cirq, put(n, 7=>Ry(0.0))  )
        # push!(cirq, time_evolve(Sz_plq(1, 4), 0.0, tol=1e-5, check_hermicity=false))
        # push!(cirq, put(n, 7=>Rz(0.0))  )
        #
        # push!(cirq, put(n, 9=>Rz(0.0))  )
        # push!(cirq, time_evolve(Sz_plq(2, 5), 0.0, tol=1e-5, check_hermicity=false))
        # push!(cirq, put(n, 9=>Rz(0.0))  )
        #
        # push!(cirq, put(n, 11=>Rz(0.0))  )
        # push!(cirq, time_evolve(Sz_plq(3, 6), 0.0, tol=1e-5, check_hermicity=false))
        # push!(cirq, put(n, 11=>Rz(0.0))  )


        # # implement XX(θ) unitaries
        # push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        # push!(cirq, chain(nqbt, put(XXbonds[i][2]=>Rz(0.)) for i=1:length(XXbonds))  )
        # push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        # push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # # implement YY(θ) unitaries
        # push!(cirq, repeat(S', collect(Iterators.flatten(YYbonds)))  )
        # push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        # push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        # push!(cirq, chain(nqbt, put(YYbonds[i][2]=>Rz(0.)) for i=1:length(YYbonds))  )
        # push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        # push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        # push!(cirq, repeat(S, collect(Iterators.flatten(YYbonds)))  )
        # # implement ZZ(θ) unitaries
        # push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
        # push!(cirq, chain(nqbt, put(ZZbonds[i][2]=>Rz(0.)) for i=1:length(ZZbonds))  )
        # push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )

        # implement XX(θ) unitaries
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, chain(nqbt, put(XXbonds[i][2]=>Rz(0.)) for i=1:length(XXbonds))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # implement YY(θ) unitaries
        push!(cirq, repeat(S', collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, chain(nqbt, put(YYbonds[i][2]=>Rz(0.)) for i=1:length(YYbonds))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(S, collect(Iterators.flatten(YYbonds)))  )
        # implement ZZ(θ) unitaries
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
        push!(cirq, chain(nqbt, put(ZZbonds[i][2]=>Rz(0.)) for i=1:length(ZZbonds))  )
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )

        if n == 12
            #additional operators
            push!(cirq, vort_cnot(1, 4, 5))
            push!(cirq, vort_cnot(5, 2, 3))
            push!(cirq, vort_cnot(3, 6, 4))
            push!(cirq, vort_cnot(4, 1, 2))
            push!(cirq, vort_cnot(2, 5, 6))
            push!(cirq, vort_cnot(6, 3, 1))
        elseif n == 8
            push!(cirq, vort_cnot(1, 2, 3))
            push!(cirq, vort_cnot(2, 3, 4))
            push!(cirq, vort_cnot(3, 4, 1))
            push!(cirq, vort_cnot(4, 1, 2))
        end
        # implement XX(θ) unitaries
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, chain(nqbt, put(XXbonds[i][2]=>Rz(0.)) for i=1:length(XXbonds))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # implement YY(θ) unitaries
        push!(cirq, repeat(S', collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, chain(nqbt, put(YYbonds[i][2]=>Rz(0.)) for i=1:length(YYbonds))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(S, collect(Iterators.flatten(YYbonds)))  )
        # implement ZZ(θ) unitaries
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
        push!(cirq, chain(nqbt, put(ZZbonds[i][2]=>Rz(0.)) for i=1:length(ZZbonds))  )
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )

    end
    # # push!(cirq, mygate(nqbt, 0.))
    # # push!(cirq, chain(nqbt, put(7=>Rz(0.0)), put(9=>Rz(0.0)), put(11=>Rz(0.0))))
    return cirq
end

function vqe_centralizer_ansatz_vort_cnots(nqbt::Int, dpth::Int)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    cirq = chain(nqbt)
    # h_verts = 1:2:(2*Lx+2)
    # v_verts = 2:(2*Lx+2):n
    # push!(cirq, horizontal_ring(nqbt, verts[1], verts[2], verts[3], 0.))
    for d=1:dpth
        if n == 12
            #additional operators
            push!(cirq, vort_cnot(6, 3, 1))
            push!(cirq, vort_cnot(2, 5, 6))
            push!(cirq, vort_cnot(4, 1, 2))
            push!(cirq, vort_cnot(3, 6, 4))
            push!(cirq, vort_cnot(5, 2, 3))
            push!(cirq, vort_cnot(1, 4, 5))

            # push!(cirq, vort_cnot(1, 2, 4))
            # push!(cirq, vort_cnot(2, 3, 5))
            # push!(cirq, vort_cnot(3, 1, 6))
            # push!(cirq, vort_cnot(4, 5, 1))
            # push!(cirq, vort_cnot(5, 6, 2))
            # push!(cirq, vort_cnot(6, 4, 3))
        end
        if n==12
            #single-plaquette rotations
            push!(cirq, put(n, 3=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(1, 2), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 3=>Rz(0.0))  )

            push!(cirq, put(n, 5=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(2, 3), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 5=>Rz(0.0))  )

            push!(cirq, put(n, 1=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(3, 1), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 1=>Rz(0.0))  )

            push!(cirq, put(n, 9=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(4, 5), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 9=>Rz(0.0))  )

            push!(cirq, put(n, 11=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(5, 6), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 11=>Rz(0.0))  )

            push!(cirq, put(n, 7=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(6, 4), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 7=>Rz(0.0))  )
        elseif n==8
            push!(cirq, put(n, 3=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(1, 2), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 3=>Rz(0.0))  )

            push!(cirq, put(n, 1=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(2, 1), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 1=>Rz(0.0))  )

            push!(cirq, put(n, 7=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(3, 4), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 7=>Rz(0.0))  )

            push!(cirq, put(n, 5=>Rz(0.0))  )
            push!(cirq, time_evolve(Sz_plq(4, 3), 0.0, tol=1e-5, check_hermicity=false))
            push!(cirq, put(n, 5=>Rz(0.0))  )
        end
#         # #another single-plaquette rotations
#         # push!(cirq, put(n, 7=>Ry(0.0))  )
#         # push!(cirq, time_evolve(Sz_plq(1, 4), 0.0, tol=1e-5, check_hermicity=false))
#         # push!(cirq, put(n, 7=>Rz(0.0))  )
#         #
#         # push!(cirq, put(n, 9=>Rz(0.0))  )
#         # push!(cirq, time_evolve(Sz_plq(2, 5), 0.0, tol=1e-5, check_hermicity=false))
#         # push!(cirq, put(n, 9=>Rz(0.0))  )
#         #
#         # push!(cirq, put(n, 11=>Rz(0.0))  )
#         # push!(cirq, time_evolve(Sz_plq(3, 6), 0.0, tol=1e-5, check_hermicity=false))
#         # push!(cirq, put(n, 11=>Rz(0.0))  )
#
#
#         # # implement XX(θ) unitaries
#         # push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
#         # push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
#         # push!(cirq, chain(nqbt, put(XXbonds[i][2]=>Rz(0.)) for i=1:length(XXbonds))  )
#         # push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
#         # push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
#         # # implement YY(θ) unitaries
#         # push!(cirq, repeat(S', collect(Iterators.flatten(YYbonds)))  )
#         # push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
#         # push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
#         # push!(cirq, chain(nqbt, put(YYbonds[i][2]=>Rz(0.)) for i=1:length(YYbonds))  )
#         # push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
#         # push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
#         # push!(cirq, repeat(S, collect(Iterators.flatten(YYbonds)))  )
#         # # implement ZZ(θ) unitaries
#         # push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
#         # push!(cirq, chain(nqbt, put(ZZbonds[i][2]=>Rz(0.)) for i=1:length(ZZbonds))  )
#         # push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
#
        # # implement XX(θ) unitaries
        # push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        # push!(cirq, chain(nqbt, put(XXbonds[i][2]=>Rz(0.)) for i=1:length(XXbonds))  )
        # push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        # push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # # implement YY(θ) unitaries
        # push!(cirq, repeat(S', collect(Iterators.flatten(YYbonds)))  )
        # push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        # push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        # push!(cirq, chain(nqbt, put(YYbonds[i][2]=>Rz(0.)) for i=1:length(YYbonds))  )
        # push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        # push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        # push!(cirq, repeat(S, collect(Iterators.flatten(YYbonds)))  )
        # # implement ZZ(θ) unitaries
        # push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
        # push!(cirq, chain(nqbt, put(ZZbonds[i][2]=>Rz(0.)) for i=1:length(ZZbonds))  )
        # push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )

        if n == 12
            #additional operators
            push!(cirq, vort_cnot(1, 4, 5))
            push!(cirq, vort_cnot(5, 2, 3))
            push!(cirq, vort_cnot(3, 6, 4))
            push!(cirq, vort_cnot(4, 1, 2))
            push!(cirq, vort_cnot(2, 5, 6))
            push!(cirq, vort_cnot(6, 3, 1))

            # push!(cirq, vort_cnot(1, 2, 4))
            # push!(cirq, vort_cnot(2, 3, 5))
            # push!(cirq, vort_cnot(3, 1, 6))
            # push!(cirq, vort_cnot(4, 5, 1))
            # push!(cirq, vort_cnot(5, 6, 2))
            # push!(cirq, vort_cnot(6, 4, 3))
        elseif n == 8
            push!(cirq, vort_cnot(1, 2, 3))
            push!(cirq, vort_cnot(2, 3, 4))
            push!(cirq, vort_cnot(3, 4, 1))
            push!(cirq, vort_cnot(4, 1, 2))
        end
        # implement XX(θ) unitaries
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, chain(nqbt, put(XXbonds[i][2]=>Rz(0.)) for i=1:length(XXbonds))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # implement YY(θ) unitaries
        push!(cirq, repeat(S', collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, chain(nqbt, put(YYbonds[i][2]=>Rz(0.)) for i=1:length(YYbonds))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(S, collect(Iterators.flatten(YYbonds)))  )
        # implement ZZ(θ) unitaries
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
        push!(cirq, chain(nqbt, put(ZZbonds[i][2]=>Rz(0.)) for i=1:length(ZZbonds))  )
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )

    end
    # # push!(cirq, mygate(nqbt, 0.))
    # # push!(cirq, chain(nqbt, put(7=>Rz(0.0)), put(9=>Rz(0.0)), put(11=>Rz(0.0))))
    return cirq
end


function pauli_string_exp(nqbt, dict, theta)  # example of dict is d = Dict(1=>"X", 2=>"X")
    cirq = chain(nqbt)                          # key - number of qubit, value - one of "X", "Y", "Z"
    dict_keys = [key for key in keys(dict)]        # if initially  several gates are on one qubit, should be combined to one
    push!(cirq, chain(nqbt, put(dict_keys[end]=>Rz(theta))) ) #To be deleted

    for key in dict_keys
        if dict[key] == "X"
            push!(cirq, put(nqbt, key=>H)  )
        elseif dict[key] == "Y"
            push!(cirq, put(nqbt, key=>S')  )
            push!(cirq, put(nqbt, key=>H)  )
        end
    end

    for i_key in 2:length(dict_keys)
        push!(cirq, cnot(nqbt, dict_keys[i_key - 1], dict_keys[i_key]))
    end

    # push!(cirq, chain(nqbt, put(dict_keys[end]=>Rz(0.))) )
    push!(cirq, chain(nqbt, put(dict_keys[end]=>Rz(theta))) )

    for i_key in length(dict_keys):-1:2
        push!(cirq, cnot(nqbt, dict_keys[i_key - 1], dict_keys[i_key]))
    end

    for key in dict_keys
        if dict[key] == "X"
            push!(cirq, put(nqbt, key=>H)  )
        elseif dict[key] == "Y"
            push!(cirq, put(nqbt, key=>H)  )
            push!(cirq, put(nqbt, key=>S)  )
        end
    end
    push!(cirq, chain(nqbt, put(dict_keys[end]=>Rz(theta))) ) #To be deleted

    return cirq
end
# d = Dict(1=>"X", 2=>"X");
# d[2]
# d_keys = [key for key in keys(d)]

# theta1 = 2.0
# cirq_temp_new = matblock(pauli_string_exp(n, d, 2*theta1))
# parameters(cirq_temp_new)
#
# mat_2 = mat(cirq_temp_new)
# cirq_temp = chain(n)
# push!(cirq_temp, time_evolve(chain(put(n, 1=>X), put(n, 2=>X)), theta1)  )
# mat_3 = exp(mat(chain(put(n, 1=>X), put(n, 2=>X))) * (-im*theta1))
# mat_1 = mat(cirq_temp)
#
# norm(mat_3 - mat_2)
# norm(mat_3 - mat_2)
#
# println()
# circuit = dispatch!(vqe_centralizer_ansatz(18, 8),:random)
# parameters(circuit)
# edges(Lx, Lz, BC)


# define circuits for 3-body centralizers
ZXY(n::Int,i::Int,j::Int,k::Int,ϕ::Float64) = chain(n, put(k=>S'), repeat(H, [j,k]), cnot(i,j), cnot(j,k), put(k=>Rz(2*ϕ)), cnot(j,k), cnot(i,j), repeat(H, [j,k]), put(k=>S))
XYZ(n::Int,i::Int,j::Int,k::Int,ϕ::Float64) = chain(n, put(j=>S'), repeat(H, [i,j]), cnot(i,j), cnot(j,k), put(k=>Rz(2*ϕ)), cnot(j,k), cnot(i,j), repeat(H, [i,j]), put(j=>S))
YZX(n::Int,i::Int,j::Int,k::Int,ϕ::Float64) = chain(n, put(i=>S'), repeat(H, [i,k]), cnot(i,j), cnot(j,k), put(k=>Rz(2*ϕ)), cnot(j,k), cnot(i,j), repeat(H, [i,k]), put(i=>S))

# initailize the 3-body centralizer anstaz
function vqe_centralizer_product_ansatz(nqbt::Int)
    _, Lijk_indices = getLijk(Lx, Lz, BC)
    cirq = chain(nqbt)
    h_verts = 1:2:(2*Lx+2)
    v_verts = 2:(2*Lx+2):n
    for k = 1:6:length(Lijk_indices)
        push!(cirq, repeat(Rz(0.0), h_verts))
        push!(cirq, repeat(Ry(0.0), v_verts))
        push!(cirq,  YZX(nqbt,Lijk_indices[k+0][1],Lijk_indices[k+0][2],Lijk_indices[k+0][3],0.))
        push!(cirq,  XYZ(nqbt,Lijk_indices[k+1][1],Lijk_indices[k+1][2],Lijk_indices[k+1][3],0.))
        push!(cirq,  ZXY(nqbt,Lijk_indices[k+2][1],Lijk_indices[k+2][2],Lijk_indices[k+2][3],0.))
        push!(cirq,  YZX(nqbt,Lijk_indices[k+3][1],Lijk_indices[k+3][2],Lijk_indices[k+3][3],0.))
        push!(cirq,  XYZ(nqbt,Lijk_indices[k+4][1],Lijk_indices[k+4][2],Lijk_indices[k+4][3],0.))
        push!(cirq,  ZXY(nqbt,Lijk_indices[k+5][1],Lijk_indices[k+5][2],Lijk_indices[k+5][3],0.))
    end
    return cirq
end

# same centralizers as before, but now with fixed angles for layers (as QAOA) --- much easier optimization
function qaoa_centralizer_ansatz(nqbt::Int, dpth::Int)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    cirq = chain(nqbt)
    for d=1:dpth
        # implement XX(θ) unitaries
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, repeat(Rz(0.), [XXbonds[i][2] for i=1:length(XXbonds)])  )
        push!(cirq, chain(nqbt, cnot(XXbonds[i][1],XXbonds[i][2]) for i=1:length(XXbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(XXbonds)))  )
        # implement YY(θ) unitaries
        push!(cirq, repeat(S', collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, repeat(Rz(0.), [YYbonds[i][2] for i=1:length(YYbonds)])  )
        push!(cirq, chain(nqbt, cnot(YYbonds[i][1],YYbonds[i][2]) for i=1:length(YYbonds))  )
        push!(cirq, repeat(H, collect(Iterators.flatten(YYbonds)))  )
        push!(cirq, repeat(S, collect(Iterators.flatten(YYbonds)))  )
        # implement ZZ(θ) unitaries
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
        push!(cirq, repeat(Rz(0.), [ZZbonds[i][2] for i=1:length(ZZbonds)])  )
        push!(cirq, chain(nqbt, cnot(ZZbonds[i][1],ZZbonds[i][2]) for i=1:length(ZZbonds))  )
    end
    return cirq
end

# prerape the ground state using imaginary time propagation
function trueground(ψin,h,n)
    τ = 1. * n;
    ψin |> time_evolve(h, -1im * τ, tol=1e-5, check_hermicity=false)
    normalize!(ψin)
    #if energies != "undefined so far"
    #    if abs(expect(h, ψin) - energies[1]) > 0.01
    #        @warn "Accuracy is not sufficient: imaginary time τ shall be increased"
    #    end
    #end
    return ψin
end
# circuit = dispatch!(vqe_centralizer_ansatz(n, 4),:random); # initialize ansatz with random parameters
# params = parameters(circuit)
# seed_val = abs(Random.rand(Int))
# Random.seed!(seed_val)



function perform_vqe(nqbt::Int, ψ0::ArrayReg, depth::Int, niter::Int, Rlearn::Float64, ansatz, choose_optimizer, print_vqe::Bool, gpu_flag::String, useparams,h,Lx,Lz,BC,psitg)
    #Random.seed!(10150);
    println("Starting VQE procedure...")
    if ansatz == 0 # use hardware-efficient
        circuit = dispatch!(variational_circuit(nqbt, depth), :random);
    elseif ansatz == 1 # use Krylov propagation for XX, YY, ZZ
        circuit = dispatch!(vqe_Krylov_centralizer_ansatz(nqbt, depth,Lx,Lz,BC),:random); # initialize ansatz with random parameters
    elseif ansatz == 2 # use gate decompositions
        if useparams == 1
            circuit = dispatch!(vqe_centralizer_ansatz(nqbt, depth,Lx,Lz,BC),params_0); # initialize ansatz with random parameters
        else
            seed_val = abs(Random.rand(Int))
            # seed_val = 5542374386626961393
            Random.seed!(seed_val)
            println("make it visible")
            println("  * * * * *   seed = $seed_val")
            #вернуть от 0 до 2пи
            circuit = dispatch!(vqe_centralizer_ansatz(nqbt, depth,Lx,Lz,BC),:random); # initialize ansatz with random parameters
            # params = parameters(circuit)
            params = Random.rand(length(parameters(circuit))).*(2*pi)
            # for i in 1:3
            #     params[i] = pi/8
            # end
            # for i in 4:6
            #     params[i] = 0.0
            # end
            # for i in 19:24
            #     params[i] = 0.0
            # end
            # for i in 37:42
            #     params[i] = 0.0
            # end
            # for i in 55:60
            #     params[i] = 0.0
            # end
            # for j in 0:18:(72-2)
            #     for i in (j+1):(j+3)
            #         params[i] = pi/8
            #     end
            #     for i in (j+4):(j+5)
            #         params[i] = 0.0
            #     end
            # end
            # println(length)
            circuit = dispatch!(vqe_centralizer_ansatz(nqbt, depth,Lx,Lz,BC),params); # initialize ansatz with random parameters
        end
    elseif ansatz == 21 # use gate decompositions
        if useparams == 1
            circuit = dispatch!(vqe_centralizer_ansatz_vort_cnots(nqbt, depth),params_0); # initialize ansatz with known parameters
            println("used known parameters")
        else
            # seed_val = abs(Random.rand(Int))
            seed_val = 5542374386626961393
            Random.seed!(seed_val)
            println("make it visible")
            println("  * * * * *   seed = $seed_val")
            circuit = dispatch!(vqe_centralizer_ansatz_vort_cnots(nqbt, depth), :random); # initialize ansatz with random parameters
            params = Random.rand(length(parameters(circuit))).*(2*pi)
            circuit = dispatch!(vqe_centralizer_ansatz_vort_cnots(nqbt, depth), params)
        end
    elseif ansatz == 3 # use 3-body products of centralizers
        circuit = dispatch!(vqe_Krylov_centralizer_product_ansatz(nqbt),:random); # initialize ansatz with random parameters
    elseif ansatz == 4 # use decomposed 3-body products of centralizers
        circuit = dispatch!(vqe_centralizer_product_ansatz(nqbt),:random); # initialize ansatz with random parameters
        # initialparameters = 0.1*randn(Float64, (nparameters(vqe_centralizer_product_ansatz(nqbt)))) #,depth
        # circuit = dispatch!(vqe_centralizer_product_ansatz(nqbt),initialparameters); # initialize ansatz with random parameters
        # circuit = dispatch!(vqe_centralizer_product_ansatz(nqbt),savedparams);
    elseif ansatz == 5 # use qaoa type ansatz
        circuit = dispatch!(qaoa_centralizer_ansatz(nqbt, depth),:random); # initialize ansatz with same angle for XX, YY, ZZ layer
        #initialparameters = [0.6269382968144052, -0.5949986008661095, -0.10270016332991791, 0.33318586433792124, 0.3819943845013352, 0.9703353633805286, 1.3156586239688357, 0.3008037082827733, 0.9144214425468532, 0.4425532298356348, 0.6887103451447525, 0.5143408924225501];
        #circuit = dispatch!(qaoa_centralizer_ansatz(nqbt, depth),initialparameters); # initialize ansatz with defined parameters
    else
        @warn "So far only few ansatz implementations are prepared"
    end
    # Full GPU support for VQE: currently only single rotation-based ansatz is supported
    if (gpu_flag == "yes") && (ansatz == 4 || ansatz == 2 || ansatz == 5)
        ψ0 |> cu # perform all calculations in CuYao with GPU
    end
    if choose_optimizer == "GD"
        for i = 1:niter
            _, grad = expect'(h, ψ0=>circuit)
            dispatch!(-, circuit, Rlearn * grad)
            if gpu_flag == "yes" # use GPU only for expectation value
                energy = real(expect(h, (ψ0 |> cu) =>circuit)) # plotting energy shall be optional
            elseif gpu_flag == "no"
                energy = real(expect(h, ψ0 =>circuit)) # plotting energy shall be optional
            else
                @warn "Incorrect GPU flag"
            end
            println("Step $i, energy = $energy")
            flush(Core.stdout)
        end
    elseif choose_optimizer == "ADAM"
        params = parameters(circuit)
        Energies_vqe = []
        Fidelities_vqe = []
        en_min = real(expect(h, ψ0 =>circuit))
        params_min = parameters(circuit)
        fid_en_min = Yao.fidelity(psitg, ψ0 =>circuit)
        opt = Optimisers.setup(Optimisers.ADAM(0.1), params)
        # magn_z = []
        # magn_y = []
        # magn_x = []
        # param1 = []
        # param2 = []
        # param3 = []
        # param4 = []
        # param5 = []
        # param6 = []

        println("Step 0, energy = $en_min, fidelity = $fid_en_min")
        for i = 1:niter
            _, grad_params = expect'(h, ψ0=>circuit)
            Optimisers.update!(opt, params, grad_params)
            dispatch!(circuit, params)
            if gpu_flag == "yes" # use GPU only for expectation value
                energy = real(expect(h, (ψ0 |> cu) =>circuit)) # plotting energy shall be optional
            elseif gpu_flag == "no"
                energy = real(expect(h, ψ0 =>circuit)) # plotting energy shall be optional
                fid_vqe = Yao.fidelity(psitg, ψ0 =>circuit)
            else
                @warn "Incorrect GPU flag"
            end
            append!(Energies_vqe, energy)
            append!(Fidelities_vqe, fid_vqe)
            if energy < en_min
                en_min = energy
                params_min = parameters(circuit)
                fid_en_min = fid_vqe
                print(" *** ")
            end
            println("Step $i, energy = $energy, fidelity = $fid_vqe")
            if (nqbt == 24) & (ansatz == 2)
                open("./N=$(n)_h=0.0_lr=$(learning_rate)_depth=$(depth)_niter=$(niter).dat", "a") do io
                    writedlm(io, [params_min])
                end
            end
            # append!(Energies_vqe, energy)
            #
            # append!(magn_z, expect(magnZ(n), ψ0 =>circuit))
            # append!(magn_y, expect(magnY(n), ψ0 =>circuit))
            # append!(magn_x, expect(magnX(n), ψ0 =>circuit))
            #
            # append!(param1, parameters(circuit)[1]+parameters(circuit)[19]+parameters(circuit)[37]+parameters(circuit)[55])
            # append!(param2, parameters(circuit)[2]+parameters(circuit)[20]+parameters(circuit)[38]+parameters(circuit)[56])
            # append!(param3, parameters(circuit)[3]+parameters(circuit)[21]+parameters(circuit)[39]+parameters(circuit)[57])
            # append!(param4, parameters(circuit)[4]+parameters(circuit)[22]+parameters(circuit)[40]+parameters(circuit)[58])
            # append!(param5, parameters(circuit)[5]+parameters(circuit)[23]+parameters(circuit)[41]+parameters(circuit)[59])
            # append!(param6, parameters(circuit)[6]+parameters(circuit)[24]+parameters(circuit)[42]+parameters(circuit)[60])
            flush(Core.stdout)
        end
    else
        @warn "Choose optimizer: only GD (gradient descent) and ADAM options are implemented"
    end
    println("VQE optimization is finished \r")
    if print_vqe
        printvqeresults(ψ0=>circuit, energies)
        flush(Core.stdout)
    end
    #clearconsole()
    return circuit, Energies_vqe, Fidelities_vqe, en_min, fid_en_min, params_min #, magn_z, magn_y, magn_x, param1, param2, param3, param4, param5, param6
end


magnZ(n::Int) = sum([put(n, i => Z) for i = 1:n]) # define total magnetization (unnormalized) in Z direction
magnY(n::Int) = sum([put(n, i => Y) for i = 1:n]) # define total magnetization (unnormalized) in Y direction
magnX(n::Int) = sum([put(n, i => X) for i = 1:n])

#---
#
#
# gatecount(dispatch!(variational_circuit(n, depth), :random))
#
# gatecount(dispatch!(vqe_Krylov_centralizer_ansatz(n, depth),:random))
#
# gatecount(dispatch!(vqe_centralizer_ansatz(n, depth),:random);)

# op = vort_swap_op(1,2,3,4)
# println(op)
# println(op)
function vort_cnot(a, b, c) # a - control, b and c - target
    idop = put(n, 1=>I2)
    return matblock(Sx_plq(b, c) * (idop - w[a]) / 2  + (idop + w[a]) / 2)
end

function vort_swap_op(a, b, c, d)
    Sx1 = Sx_plq(a,b)
    Sy1 = im * Sx1 * Sz_plq(a, b)
    #
    Sx2 = Sx_plq(c, d)
    Sy2 = im * Sx2 * Sz_plq(c, d)
    #
    Sp1 = (Sx1 + im * Sy1)/2
    Sm1 = (Sx1 - im * Sy1)/2

    Sp2 = (Sx2 + im * Sy2)/2
    Sm2 = (Sx2 - im * Sy2)/2

    idop = put(n, 1=>I2)
    prod2 = chain(w[c],w[d])
    proj2 = (idop + prod2) / 2
    prod1 = chain(w[a],w[b])
    proj1 = (idop + prod1) / 2
    # projpart = (Sp2*Sm1 + Sm2*Sp1 + Sp1*Sp2*Sm2*Sm1 + Sm1*Sm2*Sp2*Sp1)
    #
    # a = iscommute((Sp2*Sm1 + Sm2*Sp1)* proj1 * proj2 , (Sp1*Sp2*Sm2*Sm1)* proj1 * proj2)
    #
    # return projpart * proj1 * proj2 + (idop - proj1*proj2), a
    projpart = (Sp2*Sm1 + Sm2*Sp1 + Sp1*Sp2*Sm2*Sm1 + Sm1*Sm2*Sp2*Sp1)

    # println(iscommute(proj1, proj1*proj2 ))
    return (projpart - idop) * proj1 * proj2 + idop
end
function Sz_plq(a, b)  # creates Sz operator for plaquets with numbers a and b
    return ((w[a] + w[b]) / 2)
end
# println(Sx_plq(3, 4))
function Sx_plq(a, b)  # creates Sx operator for plaquets with numbers a and b for BC = 1 (periodic)
    lx = mod(mod(b, 1:(Lx+1)) - mod(a, 1:(Lx+1)), 0:(Lx+1-1))
    ly = mod(div(b-1, (Lx+1)) - div(a-1, (Lx+1)) , 0:(Lz+1-1))
    # println("lx, ly = ", lx, ", ", ly)
    chain_qb_seqZ = []
    qbn1 = plaq_seq[a][3]
    for i in qbn1:2:(qbn1+2*(lx-1))
        qb = mod(i, 1:Nx) + div((qbn1 - 1), Nx)*Nx
        append!(chain_qb_seqZ, qb)
    end
    # println(chain_qb_seqZ)
    chain_qb_seqY = []
    qbn2 = plaq_seq[mod(a+lx, 1:(Lx+1)) + div(a-1, (Lx+1))*(Lx+1)][5]
    for i in qbn2:Nx:(qbn2+Nx*(ly-1))
        qb = mod(i, 1:n)
        append!(chain_qb_seqY, qb)
    end
    # println(chain_qb_seqY)
    return chain(chain(put(n, qb=>Z) for qb in chain_qb_seqZ), chain(put(n, qb=>Y) for qb in chain_qb_seqY))
end


vecexpect(operator, s, eps::Float64) = real(chopcomplex(s' * mat(operator) * s, eps))
eps = 1e-12
