@const_gate S = [[1,0] [0, 1im]] # define phase gate (sqrt[Z]) as a constant gate

# take a register, and measure operators A_i A_j = {XX,YY,ZZ} at bond (i,j), postselecting on 0 outcome
function measure_selected!(reg, i::Int, j::Int, A)
    SELECT0 = (I2+Z)*0.5; # project on |0>
    SELECT1 = (I2-Z)*0.5; # project on |1>
    nq = nqubits(reg) # so far QND measurement requires ancilla qubit, but we can do without
    reg |> addbits!(1) |> chain(nq+1, put(nq+1=>H), control(nq+1, i=>A), control(nq+1, j=>A), put(nq+1=>H))
    reg |> put(nq+1, nq+1=>SELECT0) 
    #|> partial_tr(nq+1)
    measure!(RemoveMeasured(), reg, [nq+1])
    reg
end

# repeat measurements on each plaquette bond-by-bond, but never on adjacent -- need to shift by 2
# this can be done in parallel (it's serial so far), and can be improved
function stabilize_plaquettes!(reg, plaquettes)
    yxz = [Y, X, Z]
    for k = 1:length(plaquettes)
        for j = 1:2
            for i = j:2:6
                println("i=$(plaquettes[k][(i-1)%6+1]), j=$(plaquettes[k][(i)%6+1]), A=$(yxz[(i-1)%3+1])")
                measure_selected!(reg, plaquettes[k][(i-1)%6+1], plaquettes[k][(i)%6+1], yxz[(i-1)%3+1])
            end
        end
    end
    reg
end

function stabilize_all_plaquettes!(reg, plaquettes)
    SELECT0 = (I2+Z)*0.5; # project on |0>
    nq = nqubits(reg)
    for k = 1:length(plaquettes)
        reg |> addbits!(1) |> chain(nq+1, put(nq+1=>H))
        reg |> chain(nq+1, cnot(nq+1, plaquettes[k][1]))
        reg |> chain(nq+1, put(plaquettes[k][2]=>H), cnot(nq+1, plaquettes[k][2]), put(plaquettes[k][2]=>H))
        reg |> chain(nq+1, put(plaquettes[k][3]=>S'), cnot(nq+1, plaquettes[k][3]), put(plaquettes[k][3]=>S))
        reg |> chain(nq+1, cnot(nq+1, plaquettes[k][4]))
        reg |> chain(nq+1, put(plaquettes[k][5]=>H), cnot(nq+1, plaquettes[k][5]), put(plaquettes[k][5]=>H))
        reg |> chain(nq+1, put(plaquettes[k][6]=>S'), cnot(nq+1, plaquettes[k][6]), put(plaquettes[k][6]=>S))
        reg |> chain(nq+1, put(nq+1=>H))
        reg |> put(nq+1, nq+1=>SELECT0)
        #reg = partial_tr(density_matrix(reg),nq+1)
        measure!(RemoveMeasured(), reg, [nq+1])
    end
    reg
end

# stabilize the logic operators running along horizontal and vertical lattice directions
function stabilize_logic!(reg, LXvertices, LZvertices)
    SELECT0 = (I2+Z)*0.5; # project on |0>
    SELECT1 = (I2-Z)*0.5; # project on |1>
    nq = nqubits(reg) # so far QND measurement requires ancilla qubit, but we can do without
    # implement Z string stabilization
    reg |> addbits!(1)
    for i = 1:length(LXvertices)
        reg |> chain(nq+1, cnot(LXvertices[i], nq+1))
    end
    reg |> put(nq+1, nq+1=>SELECT0) 
    #|> partial_tr(nq+1)
    measure!(RemoveMeasured(), reg, [nq+1])
    # implement Y string stabilization
    reg |> addbits!(1)
    reg |> chain(nq+1, put(nq+1=>H))
    #S = phase(π/4)*Rz(π/2) # define S gate
    for i = 1:length(LZvertices)
        reg |> chain(nq+1, put(LZvertices[i]=>S'), cnot(nq+1, LZvertices[i]), put(LZvertices[i]=>S))
    end
    reg |> chain(nq+1, put(nq+1=>H))
    reg |> put(nq+1, nq+1=>SELECT0) 
    #|> partial_tr(nq+1)
    measure!(RemoveMeasured(), reg, [nq+1])
    reg
end

# populate lattice with vortices
function excite_vortices!(Nw, reg, plqsq, Lx,Lz,BC)
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    if BC == 0
        @warn "This function only works for toric boundaries (change to BC = 1)"
    elseif Nw > length(plqsq)
        @warn "Number of vortices is bigger than number of plaquettes! Reduce it to be less or equal to $(length(plqsq))."
    elseif isodd(Nw)
        @warn "Number of vortices must be even, as vortices come in pairs."
    else
        if iseven(Lz+1)
            icntr = 0
            for i = 1:cld(Nw,(2*(Lx+1)))
                istrt = 1 + 2*(Lx+1)*(i-1)
                for j=1:(Lx+1)
                    icntr += 1
                    if (2*icntr)>Nw
                        break
                    end
                    reg |> chain(n, put(plqsq[istrt+j-1][5]=>Y))
                    println(" >>exc>> ", plqsq[istrt+j-1][5], " Y ")
                end
            end
        else
            icntr = 0
            for i = 1:cld(Nw,(2*(Lx+1)))
                istrt = 1 + 2*(Lx+1)*(i-1)
                if istrt<=(Lz+1-1)*(Lx+1)
                    for j=1:(Lx+1)
                        icntr += 1
                        if (2*icntr)>Nw
                            break
                        end
                        reg |> chain(n, put(plqsq[istrt+j-1][5]=>Y))
                        println(" >>exc>> ", plqsq[istrt+j-1][5], " Y ")
                    end
                else
                    println("it is an extra row")
                    for j=1:2:(Lx+1)
                        icntr += 1
                        if (2*icntr)>Nw
                            break
                        end
                        reg |> chain(n, put(plqsq[istrt+j-1][4]=>Z))
                        println(" >>exc>> ", plqsq[istrt+j-1][5], " Z ")
                    end
                end
            end
        end
    end
    reg
end


# collect functions together and prepare the initial state for VQE
function prepare_inital_state(nbits::Int, num_vortices::Int, plaq_seq, LXverts, LZverts, ifprint::Bool, gpu_flag::String, Lx, Lz, BC)
    # num_vortices - fix the number of vortices to consider
    ψstab = zero_state(nbits)  # set a vacuum initial state
    if gpu_flag == "yes"
        ψstab = zero_state(nbits) |> cu  # set a vacuum initial state with GPU support
    elseif gpu_flag == "no"
        ψstab = zero_state(nbits)  # set a vacuum initial state
    else
        @warn "Incorrect GPU flag"
    end
    stabilize_all_plaquettes!(ψstab, plaq_seq) # convert it into stabilized state with zero vortices
    stabilize_logic!(ψstab, LXverts, LZverts)
    excite_vortices!(num_vortices, ψstab, plaq_seq, Lx,Lz, BC)
    if ifprint
        printinit(ψstab)
    end
    return ψstab
end
